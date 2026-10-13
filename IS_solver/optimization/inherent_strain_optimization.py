import json
from pathlib import Path

import meshio
import numpy as onp
import jax
import jax.numpy as np
from jax import value_and_grad, vmap
from scipy.optimize import OptimizeResult, minimize

from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol

from optimization.relative_contraction import build_relative_contraction
from optimization.stress_normalization import build_stress_normalization

from inherent_solver.boundary import (
    create_displacement_bc,
)
from inherent_solver.output import save_forward_result
from inherent_solver.native_solver import native_linear_solver_options
from inherent_solver.problem import InherentStrainPlasticity
from inherent_solver.schedule import (
    precompute_activation_schedule,
)
from inherent_solver.simulation import (
    create_solver_options,
    load_mesh,
)
from inherent_solver.toolpath import load_toolpath


def validate_optimization_config(config):
    required_sections = [
        "paths",
        "mesh",
        "geometry",
        "activation",
        "toolpath",
        "material",
        "optimization",
        "output",
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(f"缺少配置段：{section}")


def resolve_relative_path(base_dir, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def normalize_range(range_value):
    if range_value is None:
        return None
    x0, x1 = map(float, range_value)
    return (min(x0, x1), max(x0, x1))


def range_mask(values, range_value, tolerance=0.0):
    if range_value is None:
        return np.ones(values.shape[0], dtype=bool)
    lo, hi = normalize_range(range_value)
    return (values >= lo - tolerance) & (values <= hi + tolerance)


def vm_from_diag(sxx, syy, szz):
    return np.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
        )
        + 1.0e-12
    )


def expand_inherent_strain_parameters(parameters, tie_xy):
    """Expand optimizer variables to the three diagonal strain components."""
    if tie_xy:
        return np.stack((parameters[0], parameters[0], parameters[1]))
    return parameters


def build_base_cell_indices(mesh, config):
    opt_cfg = config["optimization"]
    base_cfg = opt_cfg.get("base_inherent_strain", {})

    scan_cfg = config["toolpath"]["scan"]
    length_m = float(scan_cfg["length_m"])
    width_m = float(scan_cfg.get("width_m", 0.0))
    layer_height = float(config["toolpath"]["layer_height_m"])

    height_m = float(base_cfg.get("height_m", 2.0 * layer_height))
    x_ratio = float(base_cfg.get("x_ratio", 0.8))
    y_ratio = float(base_cfg.get("y_ratio", 0.8))

    cells = onp.asarray(mesh.cells)
    points = onp.asarray(mesh.points)
    centroids = onp.mean(points[cells], axis=1)

    mask_x = (
        centroids[:, 0] >= -x_ratio * length_m / 2.0
    ) & (
        centroids[:, 0] <= x_ratio * length_m / 2.0
    )
    mask_y = (
        centroids[:, 1] >= -y_ratio * width_m / 2.0
    ) & (
        centroids[:, 1] <= y_ratio * width_m / 2.0
    )
    mask_z = (
        centroids[:, 2] > -height_m
    ) & (centroids[:, 2] <= 1.0e-5)

    return onp.where(mask_x & mask_y & mask_z)[0]


def build_roi_indices(mesh, config):
    roi_cfg = config["optimization"]["roi"]

    cells = onp.asarray(mesh.cells)
    points = onp.asarray(mesh.points)
    centroids = onp.mean(points[cells], axis=1)

    stress_cfg = roi_cfg["stress"]
    stress_mask = onp.ones(len(cells), dtype=bool)
    for axis, key in enumerate(("x_range_m", "y_range_m", "z_range_m")):
        stress_mask &= onp.asarray(
            range_mask(
                centroids[:, axis],
                stress_cfg.get(key),
                tolerance=1.0e-9,
            )
        )

    disp_cfg = roi_cfg["displacement"]
    point_mask = onp.ones(len(points), dtype=bool)
    point_mask &= onp.asarray(
        range_mask(points[:, 0], disp_cfg.get("x_range_m"), 1.0e-9)
    )
    point_mask &= onp.asarray(
        range_mask(points[:, 1], disp_cfg.get("y_range_m"), 1.0e-9)
    )

    if "z_range_m" in disp_cfg:
        point_mask &= onp.asarray(
            range_mask(points[:, 2], disp_cfg.get("z_range_m"), 1.0e-9)
        )
    else:
        z_target = float(disp_cfg["z_target_m"])
        z_tolerance = float(disp_cfg.get("z_tolerance_m", 1.0e-6))
        point_mask &= onp.abs(points[:, 2] - z_target) <= z_tolerance

    cell_indices = onp.where(stress_mask)[0]
    point_indices = onp.where(point_mask)[0]

    if len(cell_indices) == 0:
        raise ValueError("optimization.roi.stress 没有选中任何单元")
    if len(point_indices) == 0:
        raise ValueError("optimization.roi.displacement 没有选中任何节点")

    return cell_indices, point_indices


def load_reference_data(reference_path, mesh):
    ref_mesh = meshio.read(str(reference_path))

    required_cell_fields = ("stress_xx", "stress_yy", "stress_zz")
    missing_cell_fields = [
        name for name in required_cell_fields if name not in ref_mesh.cell_data
    ]
    if missing_cell_fields:
        raise KeyError(
            f"参考 VTU 缺少单元场：{', '.join(missing_cell_fields)}"
        )
    if "u" not in ref_mesh.point_data:
        raise KeyError("参考 VTU 缺少节点场：u")

    target_cells = onp.asarray(mesh.cells)
    matching_block_index = None
    for block_index, cell_block in enumerate(ref_mesh.cells):
        candidate_cells = onp.asarray(cell_block.data)
        if (
            candidate_cells.shape == target_cells.shape
            and onp.array_equal(candidate_cells, target_cells)
        ):
            matching_block_index = block_index
            break
    if matching_block_index is None:
        reference_cell_count = sum(len(block.data) for block in ref_mesh.cells)
        if reference_cell_count != len(mesh.cells):
            raise ValueError(
                "参考 VTU 与优化小网格的单元数量不一致："
                f"reference={reference_cell_count}, optimization={len(mesh.cells)}。"
                "AM 和固有应变优化必须使用相同的小网格及节点/单元顺序。"
            )
        raise ValueError(
            "参考 VTU 与优化小网格的单元连接或顺序不一致。"
            "两阶段必须使用同一个小网格。"
        )
    if len(ref_mesh.points) != len(mesh.points):
        raise ValueError(
            "参考 VTU 与优化小网格的节点数量不一致："
            f"reference={len(ref_mesh.points)}, optimization={len(mesh.points)}。"
            "AM 和固有应变优化必须使用相同的小网格。"
        )
    if not onp.allclose(
        onp.asarray(ref_mesh.points),
        onp.asarray(mesh.points),
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError(
            "参考 VTU 与优化小网格的节点坐标或顺序不一致。"
            "请同时检查 mesh.scale_to_meter。"
        )

    return {
        "stress_xx": np.asarray(
            ref_mesh.cell_data["stress_xx"][matching_block_index]
        ).reshape(-1),
        "stress_yy": np.asarray(
            ref_mesh.cell_data["stress_yy"][matching_block_index]
        ).reshape(-1),
        "stress_zz": np.asarray(
            ref_mesh.cell_data["stress_zz"][matching_block_index]
        ).reshape(-1),
        "u": np.asarray(ref_mesh.point_data["u"]),
    }


def build_problem_and_schedule(config, project_root, paths_base):
    mesh = load_mesh(config, project_root)
    toolpath = load_toolpath(
        config["toolpath"],
        paths_base=paths_base,
    )

    schedule = precompute_activation_schedule(
        mesh=mesh,
        toolpath=toolpath,
        base_plate_height=float(config["geometry"]["base_plate_height_m"]),
        z_tolerance=float(config["activation"].get("z_tolerance_m", 1.0e-5)),
    )
    if not schedule:
        raise ValueError("激活计划为空，无法创建优化 Problem。")

    geometry_config = config["geometry"]
    # Keep one independent FE/Problem object per increment.  The AD wrapper
    # captures its Problem, so changing one shared object's BCs between
    # increments can leak tracers across JAX transformation scopes.
    for step_info in schedule:
        displacement_bc_info = create_displacement_bc(
            quiet_point_indices=step_info["quiet_point_indices"],
            fixed_plane_z=geometry_config["fixed_plane_z_m"],
            fixed_plane_tolerance=geometry_config["fixed_plane_tolerance_m"],
        )
        step_info["problem"] = InherentStrainPlasticity(
            mesh,
            vec=3,
            dim=3,
            dirichlet_bc_info=displacement_bc_info,
            material_config=config["material"],
        )

    problem = schedule[-1]["problem"]

    return mesh, toolpath, schedule, problem


def run_simulation(
    problem,
    schedule,
    eps_params_global,
    config,
    base_cells_indices,
    save_vtk=False,
    vtk_directory=None,
):
    number_of_cells = len(problem.fe.cells)
    number_of_quads = problem.fe.num_quads

    displacement = np.zeros(
        (problem.fes[0].num_total_nodes, problem.fes[0].vec)
    )
    stress = np.zeros(
        (number_of_cells, number_of_quads, problem.dim, problem.dim)
    )
    plastic_strain = np.zeros((number_of_cells, number_of_quads))

    current_elastic_strain = np.zeros_like(stress)

    for step_info in schedule:
        step_idx = step_info["step_index"]
        active_mask_bool = step_info["active_mask"]
        new_cell_indices = step_info["new_cell_indices"]
        step_problem = step_info["problem"]

        active_mask_float = np.asarray(active_mask_bool, dtype=np.float64)[:, None, None]
        active_mask_float = np.repeat(active_mask_float, number_of_quads, axis=1)

        old_displacement_gradient = step_problem.compute_u_grad(displacement)

        eps_inh_inc = np.zeros((number_of_cells, number_of_quads, 3))
        cells_to_apply = new_cell_indices
        if step_idx == 1 and len(base_cells_indices) > 0:
            cells_to_apply = onp.unique(
                onp.concatenate([new_cell_indices, base_cells_indices]).astype(onp.int32)
            )

        if len(cells_to_apply) > 0:
            eps_inh_inc = eps_inh_inc.at[cells_to_apply, :, 0].set(eps_params_global[0])
            eps_inh_inc = eps_inh_inc.at[cells_to_apply, :, 1].set(eps_params_global[1])
            eps_inh_inc = eps_inh_inc.at[cells_to_apply, :, 2].set(eps_params_global[2])

        cut_mask = np.zeros((number_of_cells, number_of_quads, 1))
        relaxation_factor = np.zeros_like(cut_mask)

        params = [
            eps_inh_inc,
            active_mask_float,
            cut_mask,
            relaxation_factor,
            old_displacement_gradient,
            stress,
            plastic_strain,
        ]
        solver_config = config.get(
            "optimization",
            {},
        ).get("solver")

        if solver_config is None:
            solver_config = config.get(
                "solver",
                {},
            ).get(
                "forward",
                {
                    "backend": "spsolve",
                    "line_search": True,
                },
            )
        solver_options = create_solver_options(
            solver_config,
            displacement if step_idx == schedule[0]["step_index"] else None,
        )

        step_pred = ad_wrapper(
            step_problem,
            solver_options=solver_options,
            adjoint_solver_options=native_linear_solver_options(solver_config),
        )
        sol_list = step_pred(params)
        displacement = sol_list[0]

        (
            stress,
            plastic_strain,
            current_elastic_strain,
        ) = step_problem.update_stress_strain(displacement, params)

    if save_vtk and vtk_directory is not None:
        vtk_path = Path(vtk_directory) / "optimized_global_plastic_forward.vtu"
        save_forward_result(
            problem=problem,
            solution=displacement,
            stress=stress,
            plastic_strain=plastic_strain,
            elastic_strain=current_elastic_strain,
            active_mask=schedule[-1]["active_mask"],
            vtk_path=vtk_path,
            displacement_scale=float(
                config.get("output", {}).get("displacement_scale", 1.0)
            ),
        )

    return displacement, stress, plastic_strain


def run_optimization(config, project_root):
    validate_optimization_config(config)

    paths_base = Path(config["_config_dir"]).resolve()
    output_directory = resolve_relative_path(
        paths_base,
        config["paths"]["output_directory"],
    )
    vtk_directory = resolve_relative_path(
        paths_base,
        config["paths"]["vtk_directory"],
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    vtk_directory.mkdir(parents=True, exist_ok=True)

    mesh, toolpath, schedule, problem = build_problem_and_schedule(
        config,
        project_root,
        paths_base,
    )

    reference_path = resolve_relative_path(
        paths_base,
        config["paths"]["reference_vtu"],
    )
    reference = load_reference_data(reference_path, mesh)

    roi_cell_indices, uz_node_indices = build_roi_indices(mesh, config)
    base_cell_indices = build_base_cell_indices(mesh, config)

    sigma_ref_xx = reference["stress_xx"]
    sigma_ref_yy = reference["stress_yy"]
    sigma_ref_zz = reference["stress_zz"]
    reference_displacement_scale = float(
        config["optimization"].get(
            "reference_displacement_scale_to_m",
            1.0,
        )
    )
    u_ref = reference["u"] * reference_displacement_scale
    u_ref_z = u_ref[:, 2]

    sigma_ref_xx_roi = sigma_ref_xx[roi_cell_indices]
    sigma_ref_yy_roi = sigma_ref_yy[roi_cell_indices]
    sigma_ref_zz_roi = sigma_ref_zz[roi_cell_indices]
    sigma_ref_vm_roi = vm_from_diag(
        sigma_ref_xx_roi,
        sigma_ref_yy_roi,
        sigma_ref_zz_roi,
    )

    stress_normalization = build_stress_normalization(
        onp.column_stack((sigma_ref_xx_roi, sigma_ref_yy_roi, sigma_ref_zz_roi)),
        config["optimization"].get("stress_normalization"),
        onp.asarray(schedule[-1]["active_mask"])[roi_cell_indices],
        material_config=config["material"],
    )
    stress_scale_xx, stress_scale_yy, stress_scale_zz = stress_normalization.scales_mpa
    print("Stress normalization: " + json.dumps(stress_normalization.report()))
    stress_scale_vm = np.maximum(np.sqrt(np.mean(sigma_ref_vm_roi**2)), 1.0)

    u_ref_z_roi = u_ref_z[uz_node_indices]
    uz_scale = np.maximum(np.sqrt(np.mean(u_ref_z_roi**2)), 1.0e-12)

    opt_cfg = config["optimization"]
    contraction = build_relative_contraction(
        mesh.points, uz_node_indices, u_ref,
        opt_cfg.get("relative_contraction", {}),
    )
    contraction_weight = float(opt_cfg.get("loss_weights", {}).get("contraction", 1.0))
    if not onp.isfinite(contraction_weight) or contraction_weight < 0:
        raise ValueError("loss_weights.contraction must be finite and nonnegative")
    if contraction is not None:
        print("Relative contraction reference: " + json.dumps(
            contraction.report(u_ref), ensure_ascii=False))
    tie_xy = bool(opt_cfg.get("tie_xy", False))
    expected_parameter_count = 2 if tie_xy else 3
    initial_params = onp.asarray(opt_cfg["initial_params"], dtype=onp.float64)
    param_scale = onp.asarray(
        opt_cfg.get("param_scale", [1.0] * expected_parameter_count),
        dtype=onp.float64,
    )
    if initial_params.ndim != 1 or initial_params.size != expected_parameter_count:
        parameter_name = "eps_xx_yy, eps_zz" if tie_xy else "eps_xx, eps_yy, eps_zz"
        raise ValueError(
            "optimization.initial_params 必须包含 "
            f"{expected_parameter_count} 个变量 ({parameter_name})，"
            f"当前为 {initial_params.size} 个"
        )
    if param_scale.ndim != 1 or param_scale.size != expected_parameter_count:
        raise ValueError(
            "optimization.param_scale 的长度必须与 optimization.initial_params 一致"
        )
    if onp.any(param_scale <= 0.0):
        raise ValueError("optimization.param_scale 必须全部大于 0")
    bounds = opt_cfg.get("bounds")
    if bounds is None:
        raise KeyError("optimization.bounds 不能为空")
    if len(bounds) != expected_parameter_count:
        raise ValueError(
            "optimization.bounds 的长度必须与 optimization.initial_params 一致"
        )
    scaled_bounds = [
        (
            float(lo) / float(scale),
            float(hi) / float(scale),
        )
        for (lo, hi), scale in zip(bounds, param_scale)
    ]
    initial_scaled = onp.asarray(initial_params / param_scale, dtype=onp.float64)
    best_feasible = {
        "scaled_params": initial_scaled.copy(),
        "loss": onp.inf,
    }

    def evaluate_scaled(p):
        """Evaluate one optimizer point without aborting on a bad tangent.

        Plastic tangent matrices can become singular for trial strains that
        are still inside the box constraints.  Treat only the explicit
        PETSc convergence assertion as an infeasible objective point; coding
        errors and other numerical failures must remain visible.
        """
        p_np = onp.asarray(p, dtype=onp.float64)
        try:
            loss, grad = value_and_grad(compute_loss_scaled)(np.asarray(p_np))
            loss_value = float(loss)
            grad_value = onp.asarray(grad, dtype=onp.float64)
            if onp.isfinite(loss_value) and loss_value < best_feasible["loss"]:
                best_feasible["scaled_params"] = p_np.copy()
                best_feasible["loss"] = loss_value
            return loss_value, grad_value
        except AssertionError as error:
            if "PETSc linear solver failed to converge" not in str(error):
                raise
            delta = p_np - initial_scaled
            penalty = 1.0e8 + 1.0e6 * float(onp.dot(delta, delta))
            penalty_grad = 2.0e6 * delta
            eps = onp.asarray(p_np * param_scale, dtype=onp.float64)
            physical_eps = (
                onp.array([eps[0], eps[0], eps[1]])
                if tie_xy
                else eps
            )
            print(
                "PETSc trial solve failed; rejecting optimizer point: "
                f"Exx=Eyy={physical_eps[0]:.6e}, "
                f"Ezz={physical_eps[2]:.6e}; {error}"
            )
            return penalty, penalty_grad

    def compute_loss_terms(u_sol, sigmas):
        sigma_xx_cells = np.mean(sigmas[:, :, 0, 0], axis=1) / 1.0e6
        sigma_yy_cells = np.mean(sigmas[:, :, 1, 1], axis=1) / 1.0e6
        sigma_zz_cells = np.mean(sigmas[:, :, 2, 2], axis=1) / 1.0e6
        final_active_mask = schedule[-1]["active_mask"]

        sigma_xx_roi = sigma_xx_cells[roi_cell_indices]
        sigma_yy_roi = sigma_yy_cells[roi_cell_indices]
        sigma_zz_roi = sigma_zz_cells[roi_cell_indices]
        roi_mask = final_active_mask[roi_cell_indices]

        r_xx = (sigma_xx_roi - sigma_ref_xx_roi) / stress_scale_xx
        r_yy = (sigma_yy_roi - sigma_ref_yy_roi) / stress_scale_yy
        r_zz = (sigma_zz_roi - sigma_ref_zz_roi) / stress_scale_zz
        sigma_vm_roi = vm_from_diag(sigma_xx_roi, sigma_yy_roi, sigma_zz_roi)
        r_vm = (sigma_vm_roi - sigma_ref_vm_roi) / stress_scale_vm

        diff = r_xx**2 + r_yy**2 + r_zz**2
        denom = np.maximum(np.sum(roi_mask), 1.0)
        loss_data = np.sum(diff * roi_mask) / denom
        loss_xx = np.sum(r_xx**2 * roi_mask) / denom
        loss_yy = np.sum(r_yy**2 * roi_mask) / denom
        loss_zz = np.sum(r_zz**2 * roi_mask) / denom
        # Evaluate the OLD objective on the same solution for valid comparisons.
        old_scales = stress_normalization.legacy_component_scales_mpa
        old_diff = ((sigma_xx_roi - sigma_ref_xx_roi) / old_scales[0])**2
        old_diff += ((sigma_yy_roi - sigma_ref_yy_roi) / old_scales[1])**2
        old_diff += ((sigma_zz_roi - sigma_ref_zz_roi) / old_scales[2])**2
        component_rms_stress = np.sum(old_diff * roi_mask) / denom
        loss_vm = np.sum((r_vm**2) * roi_mask) / denom

        u_z_sol_roi = u_sol[:, 2][uz_node_indices]
        r_uz = (u_z_sol_roi - u_ref_z_roi) / uz_scale
        loss_uz = np.sum(r_uz**2) / np.maximum(len(uz_node_indices), 1.0)

        weights = opt_cfg.get("loss_weights", {})
        loss_contraction = contraction.loss(u_sol) if contraction is not None else np.asarray(0.0)
        weighted_stress = float(weights.get("stress", 1.0)) * loss_data
        weighted_vm = float(weights.get("vm", 1.0)) * loss_vm
        weighted_uz = float(weights.get("uz", 1.0)) * loss_uz
        weighted_contraction = contraction_weight * loss_contraction
        return {
            "stress": loss_data, "vm": loss_vm, "uz": loss_uz,
            "stress_xx": loss_xx, "stress_yy": loss_yy, "stress_zz": loss_zz,
            "component_rms_stress": component_rms_stress,
            "component_rms_original_total": (
                float(weights.get("stress", 1.0)) * component_rms_stress
                + weighted_vm + weighted_uz
            ),
            "contraction": loss_contraction,
            "weighted_stress": weighted_stress, "weighted_vm": weighted_vm,
            "weighted_uz": weighted_uz, "weighted_contraction": weighted_contraction,
            "original_total": weighted_stress + weighted_vm + weighted_uz,
            "total": weighted_stress + weighted_vm + weighted_uz + weighted_contraction,
        }

    def compute_loss(eps_params_global):
        u_sol, sigmas, _ = run_simulation(
            problem=problem,
            schedule=schedule,
            eps_params_global=eps_params_global,
            config=config,
            base_cells_indices=base_cell_indices,
            save_vtk=False,
        )

        return compute_loss_terms(u_sol, sigmas)["total"]

    def compute_loss_scaled(p):
        return compute_loss(
            expand_inherent_strain_parameters(p * param_scale, tie_xy)
        )

    print("\n" + "=" * 60)
    print("Gradient check")
    loss_center, grad_ad = evaluate_scaled(initial_scaled)
    print(f"Loss center = {loss_center:.6e}")

    optimizer_cfg = opt_cfg.get("optimizer", {})
    evaluation_count = 0
    iteration_count = 0
    last_evaluation = {"scaled_params": None, "loss": None}
    accepted_losses = []
    verified_accepted_losses = []
    early_stop_reason = None
    early_stop_success = False

    def func_scaled(p):
        nonlocal evaluation_count
        loss_val, grad_val = evaluate_scaled(p)
        evaluation_count += 1
        scaled_params = onp.asarray(p, dtype=onp.float64)
        last_evaluation["scaled_params"] = scaled_params.copy()
        last_evaluation["loss"] = loss_val
        eps = scaled_params * param_scale
        physical_eps = (
            onp.array([eps[0], eps[0], eps[1]])
            if tie_xy
            else eps
        )
        print(
            f"Eval {evaluation_count}: Loss={loss_val:.6e}, "
            f"Exx=Eyy={physical_eps[0]:.6e}, "
            f"Ezz={physical_eps[2]:.6e}"
        )
        return loss_val, grad_val

    def report_iteration(p):
        nonlocal iteration_count, early_stop_reason, early_stop_success
        iteration_count += 1
        scaled_params = onp.asarray(p, dtype=onp.float64)
        eps = scaled_params * param_scale
        physical_eps = (
            onp.array([eps[0], eps[0], eps[1]])
            if tie_xy
            else eps
        )
        loss_val = last_evaluation["loss"]
        accepted_losses.append(float(loss_val))
        matches_last_evaluation = onp.array_equal(
            scaled_params,
            last_evaluation["scaled_params"],
        )
        if not matches_last_evaluation:
            loss_text = "n/a"
        else:
            loss_text = f"{loss_val:.6e}"
            if onp.isfinite(loss_val):
                verified_accepted_losses.append(float(loss_val))
        print(
            f"Iter {iteration_count} accepted: Loss={loss_text}, "
            f"Exx=Eyy={physical_eps[0]:.6e}, Ezz={physical_eps[2]:.6e}"
        )

        stagnation_window = int(optimizer_cfg.get("stagnation_window", 3))
        stagnation_tolerance = float(
            optimizer_cfg.get("stagnation_relative_tolerance", 1.0e-5)
        )
        if stagnation_window < 2:
            raise ValueError("optimization.optimizer.stagnation_window 必须至少为 2")
        if stagnation_tolerance < 0.0:
            raise ValueError(
                "optimization.optimizer.stagnation_relative_tolerance 不能小于 0"
            )
        if len(verified_accepted_losses) >= stagnation_window:
            recent = onp.asarray(
                verified_accepted_losses[-stagnation_window:],
                dtype=onp.float64,
            )
            relative_loss_spread = (
                (onp.max(recent) - onp.min(recent))
                / max(abs(float(recent[0])), 1.0)
            )
            if relative_loss_spread <= stagnation_tolerance:
                early_stop_reason = (
                    "converged by objective stagnation: relative loss spread "
                    f"{relative_loss_spread:.3e} <= {stagnation_tolerance:.3e} "
                    f"over {stagnation_window} accepted iterations"
                )
                early_stop_success = True
                raise RuntimeError(early_stop_reason)

        window = int(optimizer_cfg.get("oscillation_window", 6))
        min_improvement = float(
            optimizer_cfg.get("oscillation_min_relative_improvement", 1.0e-4)
        )
        if len(accepted_losses) >= window:
            recent = onp.asarray(accepted_losses[-window:], dtype=onp.float64)
            deltas = onp.diff(recent)
            signs = onp.sign(deltas[onp.abs(deltas) > 1.0e-14])
            sign_changes = int(onp.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
            relative_improvement = (
                (recent[0] - onp.min(recent))
                / max(abs(float(recent[0])), 1.0)
            )
            if sign_changes >= 2 and relative_improvement < min_improvement:
                early_stop_reason = (
                    "oscillation detected: "
                    f"{sign_changes} loss direction changes in the last {window} "
                    f"accepted steps, relative improvement={relative_improvement:.3e}"
                )
                raise RuntimeError(early_stop_reason)

    try:
        result = minimize(
            func_scaled,
            initial_scaled,
            method=optimizer_cfg.get("method", "L-BFGS-B"),
            jac=True,
            bounds=scaled_bounds,
            callback=report_iteration,
            options={
                "maxiter": int(optimizer_cfg.get("maxiter", 50)),
                "ftol": float(optimizer_cfg.get("ftol", 1.0e-12)),
                "gtol": float(optimizer_cfg.get("gtol", 1.0e-8)),
                "maxls": int(optimizer_cfg.get("maxls", 20)),
            },
        )
    except RuntimeError as error:
        if early_stop_reason is None or str(error) != early_stop_reason:
            raise
        best_scaled = onp.asarray(best_feasible["scaled_params"], dtype=onp.float64)
        result = OptimizeResult(
            x=best_scaled,
            fun=float(best_feasible["loss"]),
            jac=onp.zeros_like(best_scaled),
            nit=iteration_count,
            nfev=evaluation_count,
            success=early_stop_success,
            status=0 if early_stop_success else 99,
            message=early_stop_reason,
        )

    print(
        "Optimization stopped: "
        f"success={result.success}, status={result.status}, "
        f"iterations={result.nit}, evaluations={result.nfev}, "
        f"message={result.message}"
    )

    final_scaled = onp.asarray(result.x, dtype=onp.float64)
    if best_feasible["loss"] < onp.inf and (
        not onp.array_equal(final_scaled, best_feasible["scaled_params"])
        and best_feasible["loss"] < float(result.fun)
    ):
        print(
            "Using the best converged objective point because the optimizer "
            "ended on a rejected PETSc trial."
        )
        final_scaled = best_feasible["scaled_params"]
    final_optimizer_params = final_scaled * param_scale
    final_params = onp.asarray(
        expand_inherent_strain_parameters(
            np.asarray(final_optimizer_params), tie_xy
        ),
        dtype=onp.float64,
    )
    print("\n" + "=" * 60)
    print("Optimized parameters")
    if tie_xy:
        print(
            f"Exx=Eyy variable={final_optimizer_params[0]:.6e}, "
            f"Ezz={final_optimizer_params[1]:.6e} "
            f"(mapped to Exx=Eyy={final_params[0]:.6e})"
        )
    else:
        print(
            f"Exx={final_params[0]:.6e}, "
            f"Eyy={final_params[1]:.6e}, Ezz={final_params[2]:.6e}"
        )

    result_path_value = config.get("output", {}).get("epsilon_result_json")
    if result_path_value:
        result_path = resolve_relative_path(paths_base, result_path_value)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "inherent_strain": {
                "xx": float(final_params[0]),
                "yy": float(final_params[1]),
                "zz": float(final_params[2]),
            },
            "parameterization": {
                "tie_xy": tie_xy,
                "variables": ["eps_xx_yy", "eps_zz"]
                if tie_xy
                else ["eps_xx", "eps_yy", "eps_zz"],
            },
            "optimization": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "loss": float(result.fun),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            },
        }
    final_sol, final_sigmas, final_p = run_simulation(
        problem=problem,
        schedule=schedule,
        eps_params_global=final_params,
        config=config,
        base_cells_indices=base_cell_indices,
        save_vtk=True,
        vtk_directory=vtk_directory,
    )

    final_terms = {name: float(value) for name, value in compute_loss_terms(final_sol, final_sigmas).items()}
    print("Final objective terms: " + json.dumps(final_terms))
    if result_path_value:
        result_payload["optimization"]["loss"] = final_terms["total"]
        result_payload["objective"] = final_terms
        result_payload["stress_normalization"] = stress_normalization.report()
        result_payload["reference_vtu"] = str(reference_path)
        result_payload["material"] = config["material"]
        if contraction is not None:
            result_payload["relative_contraction"] = contraction.report(final_sol)
            result_payload["relative_contraction"]["weight"] = contraction_weight
        result_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Optimized inherent strain JSON: {result_path}")

    sigma_xx_final = onp.asarray(np.mean(final_sigmas[:, :, 0, 0], axis=1)) / 1.0e6
    sigma_yy_final = onp.asarray(np.mean(final_sigmas[:, :, 1, 1], axis=1)) / 1.0e6
    sigma_zz_final = onp.asarray(np.mean(final_sigmas[:, :, 2, 2], axis=1)) / 1.0e6
    sigma_xx_ref = onp.asarray(sigma_ref_xx)
    sigma_yy_ref = onp.asarray(sigma_ref_yy)
    sigma_zz_ref = onp.asarray(sigma_ref_zz)
    stress_xx_err = onp.abs(sigma_xx_final - sigma_xx_ref)
    stress_yy_err = onp.abs(sigma_yy_final - sigma_yy_ref)
    stress_zz_err = onp.abs(sigma_zz_final - sigma_zz_ref)
    stress_error_norm = onp.sqrt(
        stress_xx_err**2 + stress_yy_err**2 + stress_zz_err**2
    )

    u_sim = onp.asarray(final_sol)
    u_ref_output = onp.asarray(u_ref)
    u_error = onp.abs(u_sim - u_ref_output)
    u_error_norm = onp.linalg.norm(u_sim - u_ref_output, axis=1)

    save_sol(
        problem.fe,
        final_sol,
        str(Path(vtk_directory) / "optimized_global_plastic.vtu"),
        point_infos=[
            ("u", final_sol),
            ("u_sim", u_sim),
            ("u_ref", u_ref_output),
            ("u_err", u_error),
            ("u_err_norm", u_error_norm),
        ],
        cell_infos=[
            ("active_mask", onp.asarray(schedule[-1]["active_mask"], dtype=onp.int32)),
            ("stress_xx_sim", sigma_xx_final),
            ("stress_yy_sim", sigma_yy_final),
            ("stress_zz_sim", sigma_zz_final),
            ("stress_xx_ref", sigma_xx_ref),
            ("stress_yy_ref", sigma_yy_ref),
            ("stress_zz_ref", sigma_zz_ref),
            ("stress_xx_err", stress_xx_err),
            ("stress_yy_err", stress_yy_err),
            ("stress_zz_err", stress_zz_err),
            ("stress_error_norm", stress_error_norm),
        ],
    )

    forward_error_path = Path(vtk_directory) / "optimized_forward_error.vtu"
    save_sol(
        problem.fe,
        final_sol,
        str(forward_error_path),
        point_infos=[
            ("u_sim", u_sim),
            ("u_ref", u_ref_output),
            ("u_err", u_error),
            ("u_err_norm", u_error_norm),
        ],
        cell_infos=[
            ("active_mask", onp.asarray(schedule[-1]["active_mask"], dtype=onp.int32)),
            ("stress_xx_sim", sigma_xx_final),
            ("stress_yy_sim", sigma_yy_final),
            ("stress_zz_sim", sigma_zz_final),
            ("stress_xx_ref", sigma_xx_ref),
            ("stress_yy_ref", sigma_yy_ref),
            ("stress_zz_ref", sigma_zz_ref),
            ("stress_xx_err", stress_xx_err),
            ("stress_yy_err", stress_yy_err),
            ("stress_zz_err", stress_zz_err),
            ("stress_error_norm", stress_error_norm),
        ],
    )
    print(f"Forward error VTU: {forward_error_path}")
    print(
        "Forward errors: "
        f"u_rms={onp.sqrt(onp.mean(u_error_norm**2)):.6e} m, "
        f"u_max={onp.max(u_error_norm):.6e} m, "
        f"stress_rms={onp.sqrt(onp.mean(stress_error_norm**2)):.6e} MPa, "
        f"stress_max={onp.max(stress_error_norm):.6e} MPa"
    )

    return result
