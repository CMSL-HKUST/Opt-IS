import jax.numpy as np
import numpy as onp

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver

from inherent_solver.boundary import (
    create_displacement_bc,
    get_quiet_point_indices,
)
from inherent_solver.problem import (
    InherentStrainPlasticity,
)
from inherent_solver.native_solver import native_linear_solver_options
from inherent_solver.output import (
    save_cutting_result,
)


def normalize_range(range_value):
    if range_value is None:
        return None

    range_tuple = tuple(
        map(float, range_value)
    )

    if len(range_tuple) != 2:
        raise ValueError(
            "切割范围必须包含两个数值。"
        )

    return (
        min(range_tuple),
        max(range_tuple),
    )


def create_final_active_cell_mask(
    mesh,
    toolpath,
    geometry_config,
    activation_config,
):
    cells = onp.asarray(
        mesh.cells
    )

    points = onp.asarray(
        mesh.points
    )

    centroids = onp.mean(
        points[cells],
        axis=1,
    )

    base_plate_height = float(
        geometry_config[
            "base_plate_height_m"
        ]
    )

    z_tolerance = float(
        activation_config.get(
            "z_tolerance_m",
            1.0e-5,
        )
    )

    final_active_mask = onp.zeros(
        len(cells),
        dtype=bool,
    )

    final_active_mask[
        centroids[:, 2]
        <= base_plate_height
    ] = True

    top_z = (
        toolpath[-1, 5]
        + base_plate_height
    )

    final_active_mask[
        centroids[:, 2]
        <= top_z + z_tolerance
    ] = True

    return final_active_mask


def infer_cell_silence_z_range(
    cutting_config,
    toolpath,
):
    z_range = cutting_config.get(
        "z_range_m"
    )

    if z_range is not None:
        return normalize_range(
            z_range
        )

    if "z_cut_m" not in cutting_config:
        return None

    z_cut = float(
        cutting_config["z_cut_m"]
    )

    thickness = cutting_config.get(
        "cell_selection_thickness_m"
    )

    if thickness is None:
        z_values = onp.asarray(
            toolpath[:, 5],
            dtype=float,
        )

        layer_steps = onp.diff(
            onp.unique(z_values)
        )

        positive_steps = layer_steps[
            layer_steps > 0.0
        ]

        if len(positive_steps) == 0:
            thickness = float(
                cutting_config.get(
                    "interface_tolerance_m",
                    1.0e-6,
                )
            )
        else:
            thickness = float(
                positive_steps[0]
            )

    thickness = abs(
        float(thickness)
    )

    return normalize_range(
        (
            z_cut,
            z_cut + thickness,
        )
    )


def build_cell_selection_mask(
    mesh,
    cutting_config,
    toolpath,
):
    cells = onp.asarray(
        mesh.cells
    )

    points = onp.asarray(
        mesh.points
    )

    centroids = onp.mean(
        points[cells],
        axis=1,
    )

    tolerance = float(
        cutting_config.get(
            "cell_selection_tolerance_m",
            1.0e-9,
        )
    )

    ranges = {
        0: normalize_range(
            cutting_config.get(
                "x_range_m"
            )
        ),
        1: normalize_range(
            cutting_config.get(
                "y_range_m"
            )
        ),
        2: infer_cell_silence_z_range(
            cutting_config,
            toolpath,
        ),
    }

    selection_mask = onp.ones(
        len(cells),
        dtype=bool,
    )

    has_any_range = False

    for axis, axis_range in ranges.items():
        if axis_range is None:
            continue

        has_any_range = True

        selection_mask &= (
            (
                centroids[:, axis]
                >= axis_range[0] - tolerance
            )
            & (
                centroids[:, axis]
                <= axis_range[1] + tolerance
            )
        )

    if not has_any_range:
        raise ValueError(
            "cell_silence 模式需要至少设置 "
            "x_range_m、y_range_m、z_range_m "
            "或 z_cut_m。"
        )

    if bool(
        cutting_config.get(
            "invert",
            False,
        )
    ):
        selection_mask = ~selection_mask

    return selection_mask


def split_mesh_at_interface(
    points,
    cells,
    z_cut,
    tolerance=1.0e-6,
    x_range=None,
    gap=0.0,
):
    """
    Duplicate nodes at the z=z_cut interface.

    pairs[:, 0]: lower-side original nodes
    pairs[:, 1]: upper-side duplicate nodes
    """
    points = onp.asarray(
        points
    ).copy()

    cells = onp.asarray(
        cells
    ).copy()

    original_number_of_points = len(
        points
    )

    centroids = points[
        cells
    ].mean(axis=1)

    upper_cell_mask = (
        centroids[:, 2]
        > z_cut
    )

    if x_range is not None:
        x_min = min(x_range)
        x_max = max(x_range)

        upper_cell_mask &= (
            (centroids[:, 0] >= x_min)
            & (centroids[:, 0] <= x_max)
        )

    nodes_on_plane = onp.isclose(
        points[:, 2],
        z_cut,
        atol=tolerance,
    )

    upper_nodes = onp.unique(
        cells[upper_cell_mask]
    )

    lower_nodes = onp.unique(
        cells[~upper_cell_mask]
    )

    interface_nodes = onp.array(
        sorted(
            set(upper_nodes.tolist())
            & set(lower_nodes.tolist())
            & set(
                onp.where(
                    nodes_on_plane
                )[0].tolist()
            )
        ),
        dtype=onp.int64,
    )

    new_node_ids = onp.arange(
        original_number_of_points,
        original_number_of_points
        + len(interface_nodes),
    )

    duplicated_points = points[
        interface_nodes
    ].copy()

    duplicated_points[:, 2] += gap

    new_points = onp.vstack([
        points,
        duplicated_points,
    ])

    node_remap = onp.arange(
        original_number_of_points,
        dtype=onp.int64,
    )

    node_remap[
        interface_nodes
    ] = new_node_ids

    new_cells = cells.copy()

    new_cells[
        upper_cell_mask
    ] = node_remap[
        cells[upper_cell_mask]
    ]

    if len(interface_nodes) == 0:
        pairs = onp.empty(
            (0, 2),
            dtype=onp.int64,
        )
    else:
        pairs = onp.stack(
            [
                interface_nodes,
                new_node_ids,
            ],
            axis=1,
        )

    return (
        new_points,
        new_cells,
        pairs,
    )


def build_release_boundary_conditions(
    mesh_split,
    toolpath,
    geometry_config,
    activation_config,
):
    cells = onp.asarray(
        mesh_split.cells
    )

    points = onp.asarray(
        mesh_split.points
    )

    centroids = onp.mean(
        points[cells],
        axis=1,
    )

    base_plate_height = float(
        geometry_config[
            "base_plate_height_m"
        ]
    )

    z_tolerance = float(
        activation_config.get(
            "z_tolerance_m",
            1.0e-5,
        )
    )

    final_active_mask = onp.zeros(
        len(cells),
        dtype=bool,
    )

    final_active_mask[
        centroids[:, 2]
        <= base_plate_height
    ] = True

    top_z = (
        toolpath[-1, 5]
        + base_plate_height
    )

    final_active_mask[
        centroids[:, 2]
        <= top_z + z_tolerance
    ] = True

    quiet_point_indices = (
        get_quiet_point_indices(
            cells,
            points,
            final_active_mask,
        )
    )

    return create_displacement_bc(
        quiet_point_indices=(
            quiet_point_indices
        ),
        fixed_plane_z=geometry_config[
            "fixed_plane_z_m"
        ],
        fixed_plane_tolerance=(
            geometry_config[
                "fixed_plane_tolerance_m"
            ]
        ),
    )


def create_solver_options(
    solver_config,
    initial_guess,
):
    newton_options = {
        **native_linear_solver_options(solver_config),
        "line_search_flag": bool(
            solver_config.get(
                "line_search",
                True,
            )
        ),
        "initial_guess": [
            initial_guess
        ],
    }

    newton_options["tol"] = float(
        solver_config.get(
            "tolerance",
            1.0e-6,
        )
    )

    newton_options["rel_tol"] = float(
        solver_config.get(
            "relative_tolerance",
            1.0e-8,
        )
    )

    return newton_options


def run_cutting_by_release(
    original_mesh,
    toolpath,
    displacement,
    stress,
    plastic_strain,
    material_config,
    geometry_config,
    activation_config,
    cutting_config,
    solver_config,
    vtk_directory,
    displacement_scale=1000.0,
):
    z_cut = float(
        cutting_config["z_cut_m"]
    )

    x_range = cutting_config.get(
        "x_range_m"
    )

    if x_range is not None:
        x_range = tuple(
            map(float, x_range)
        )

    gap = float(
        cutting_config.get(
            "gap_m",
            0.0,
        )
    )

    tolerance = float(
        cutting_config.get(
            "interface_tolerance_m",
            1.0e-6,
        )
    )

    tag = cutting_config.get(
        "tag",
        "release",
    )

    print("\n" + "=" * 60)
    print(
        "✂️ Cutting by node release:"
    )
    print(
        f"   z_cut={z_cut}, "
        f"x_range={x_range}, gap={gap}"
    )
    print("=" * 60)

    (
        new_points,
        new_cells,
        node_pairs,
    ) = split_mesh_at_interface(
        points=original_mesh.points,
        cells=original_mesh.cells,
        z_cut=z_cut,
        tolerance=tolerance,
        x_range=x_range,
        gap=gap,
    )

    print(
        f"Duplicated interface nodes: "
        f"{len(node_pairs)}"
    )

    if len(node_pairs) == 0:
        raise ValueError(
            "没有找到可拆分的界面节点。请检查："
            "z_cut、x_range 和 interface_tolerance。"
        )

    split_mesh = Mesh(
        new_points,
        new_cells,
    )

    # Duplicate nodes inherit the corresponding original displacements to preserve pre-cut continuity.
    extended_displacement = onp.vstack([
        onp.asarray(displacement),
        onp.asarray(displacement)[
            node_pairs[:, 0]
        ],
    ])

    displacement_bc_info = (
        build_release_boundary_conditions(
            mesh_split=split_mesh,
            toolpath=toolpath,
            geometry_config=(
                geometry_config
            ),
            activation_config=(
                activation_config
            ),
        )
    )

    cutting_problem = (
        InherentStrainPlasticity(
            split_mesh,
            vec=3,
            dim=3,
            dirichlet_bc_info=(
                displacement_bc_info
            ),
            material_config=(
                material_config
            ),
        )
    )

    number_of_cells = len(
        new_cells
    )

    number_of_quads = (
        cutting_problem.fe.num_quads
    )

    # Allow stress redistribution in all cells during cutting.
    active_mask = np.ones(
        (
            number_of_cells,
            number_of_quads,
            1,
        )
    )

    cut_mask = np.zeros_like(
        active_mask
    )

    relaxation_factor = np.zeros_like(
        active_mask
    )

    zero_inherent_strain = np.zeros(
        (
            number_of_cells,
            number_of_quads,
            3,
        )
    )

    old_displacement_gradient = (
        cutting_problem.compute_u_grad(
            np.asarray(
                extended_displacement
            )
        )
    )

    params = [
        zero_inherent_strain,
        active_mask,
        cut_mask,
        relaxation_factor,
        old_displacement_gradient,
        stress,
        plastic_strain,
    ]

    cutting_problem.set_params(
        params
    )

    solver_options = (
        create_solver_options(
            solver_config,
            np.asarray(
                extended_displacement
            ),
        )
    )

    cutting_solution = solver(
        cutting_problem,
        solver_options=solver_options,
    )[0]

    print("Cutting release solved.")

    (
        cutting_stress,
        cutting_plastic_strain,
        cutting_elastic_strain,
    ) = cutting_problem.update_stress_strain(
        cutting_solution,
        params,
    )

    vtk_path = (
        vtk_directory
        / f"step_{tag}_final.vtu"
    )

    save_cutting_result(
        problem=cutting_problem,
        solution=cutting_solution,
        stress=cutting_stress,
        plastic_strain=(
            cutting_plastic_strain
        ),
        elastic_strain=(
            cutting_elastic_strain
        ),
        vtk_path=vtk_path,
        displacement_scale=(
            displacement_scale
        ),
    )

    print(
        f"Saved cutting result to: "
        f"{vtk_path}"
    )

    return (
        cutting_solution,
        cutting_stress,
        cutting_plastic_strain,
        cutting_problem,
    )


def run_cutting_by_cell_silence(
    original_mesh,
    toolpath,
    displacement,
    stress,
    plastic_strain,
    material_config,
    geometry_config,
    activation_config,
    cutting_config,
    solver_config,
    vtk_directory,
    retained_cell_mask=None,
    displacement_scale=1000.0,
):
    tag = cutting_config.get(
        "tag",
        "cell_silence",
    )

    number_of_substeps = int(
        cutting_config.get(
            "substeps",
            cutting_config.get(
                "n_sub",
                10,
            ),
        )
    )

    if number_of_substeps < 1:
        raise ValueError(
            "cell_silence 的 substeps 必须大于等于 1。"
        )

    final_active_mask = (
        create_final_active_cell_mask(
            original_mesh,
            toolpath,
            geometry_config,
            activation_config,
        )
    )

    if retained_cell_mask is None:
        retained_cell_mask = (
            final_active_mask.copy()
        )
    else:
        retained_cell_mask = onp.asarray(
            retained_cell_mask,
            dtype=bool,
        )

        if len(retained_cell_mask) != len(
            original_mesh.cells
        ):
            raise ValueError(
                "retained_cell_mask 的长度与当前网格单元数不一致。"
            )

        retained_cell_mask = (
            retained_cell_mask
            & final_active_mask
        )

    selected_cell_mask = (
        build_cell_selection_mask(
            original_mesh,
            cutting_config,
            toolpath,
        )
    )

    cut_cell_mask = (
        retained_cell_mask
        & selected_cell_mask
    )

    remaining_cell_mask = (
        retained_cell_mask
        & ~selected_cell_mask
    )

    print("\n" + "=" * 60)
    print(
        "✂️ Cutting by silent cells:"
    )
    print(
        f"   selected={int(cut_cell_mask.sum())}, "
        f"remaining={int(remaining_cell_mask.sum())}, "
        f"substeps={number_of_substeps}"
    )
    print("=" * 60)

    if int(cut_cell_mask.sum()) == 0:
        raise ValueError(
            "cell_silence 没有选中任何待静默单元。"
            "请检查 x/y/z 范围、z_cut_m、invert "
            "和 cell_selection_tolerance_m。"
        )

    if int(remaining_cell_mask.sum()) == 0:
        raise ValueError(
            "切除后没有剩余单元，请检查切除范围。"
        )

    quiet_point_indices = (
        get_quiet_point_indices(
            original_mesh.cells,
            original_mesh.points,
            remaining_cell_mask,
        )
    )

    displacement_bc_info = (
        create_displacement_bc(
            quiet_point_indices=(
                quiet_point_indices
            ),
            fixed_plane_z=geometry_config[
                "fixed_plane_z_m"
            ],
            fixed_plane_tolerance=(
                geometry_config[
                    "fixed_plane_tolerance_m"
                ]
            ),
        )
    )

    cutting_problem = (
        InherentStrainPlasticity(
            original_mesh,
            vec=3,
            dim=3,
            dirichlet_bc_info=(
                displacement_bc_info
            ),
            material_config=(
                material_config
            ),
        )
    )

    number_of_cells = len(
        original_mesh.cells
    )

    number_of_quads = (
        cutting_problem.fe.num_quads
    )

    active_mask = np.asarray(
        remaining_cell_mask,
        dtype=np.float64,
    )[:, None, None]

    active_mask = np.repeat(
        active_mask,
        number_of_quads,
        axis=1,
    )

    silent_cell_mask = (
        final_active_mask
        & ~remaining_cell_mask
    )

    cut_mask = np.asarray(
        silent_cell_mask,
        dtype=np.float64,
    )[:, None, None]

    cut_mask = np.repeat(
        cut_mask,
        number_of_quads,
        axis=1,
    )

    zero_inherent_strain = np.zeros(
        (
            number_of_cells,
            number_of_quads,
            3,
        )
    )

    frozen_stress = stress
    frozen_plastic_strain = (
        plastic_strain
    )

    remaining_for_stress = np.asarray(
        remaining_cell_mask
    )[:, None, None, None]

    remaining_for_plastic_strain = np.asarray(
        remaining_cell_mask
    )[:, None]

    current_stress_history = stress
    current_plastic_strain_history = (
        plastic_strain
    )
    current_displacement_gradient = (
        cutting_problem.compute_u_grad(
            displacement
        )
    )
    current_displacement = displacement

    for substep_index in range(
        1,
        number_of_substeps + 1,
    ):
        relaxation_value = (
            1.0
            - substep_index
            / float(number_of_substeps)
        )

        relaxation_factor = np.full(
            (
                number_of_cells,
                number_of_quads,
                1,
            ),
            relaxation_value,
        )

        params = [
            zero_inherent_strain,
            active_mask,
            cut_mask,
            relaxation_factor,
            current_displacement_gradient,
            current_stress_history,
            current_plastic_strain_history,
        ]

        cutting_problem.set_params(
            params
        )

        solver_options = (
            create_solver_options(
                solver_config,
                current_displacement,
            )
        )

        current_displacement = solver(
            cutting_problem,
            solver_options=solver_options,
        )[0]

        print(
            f"   substep {substep_index} / "
            f"{number_of_substeps} "
            f"(relax={relaxation_value:.3f}) solved."
        )

        (
            current_stress,
            current_plastic_strain,
            current_elastic_strain,
        ) = cutting_problem.update_stress_strain(
            current_displacement,
            params,
        )

        current_stress_history = np.where(
            remaining_for_stress,
            current_stress,
            frozen_stress,
        )

        current_plastic_strain_history = np.where(
            remaining_for_plastic_strain,
            current_plastic_strain,
            frozen_plastic_strain,
        )

        current_displacement_gradient = (
            cutting_problem.compute_u_grad(
                current_displacement
            )
        )

    vtk_path = (
        vtk_directory
        / f"step_{tag}_final.vtu"
    )

    save_cutting_result(
        problem=cutting_problem,
        solution=current_displacement,
        stress=current_stress,
        plastic_strain=(
            current_plastic_strain
        ),
        elastic_strain=(
            current_elastic_strain
        ),
        vtk_path=vtk_path,
        active_mask=(
            remaining_cell_mask
        ),
        displacement_scale=(
            displacement_scale
        ),
    )

    print(
        f"Saved cutting result to: "
        f"{vtk_path}"
    )

    return (
        current_displacement,
        current_stress,
        current_plastic_strain,
        cutting_problem,
        remaining_cell_mask,
    )
