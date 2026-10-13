import meshio
import numpy as onp

import jax
import jax.numpy as np

from jax_fem.generate_mesh import (
    Mesh,
    get_meshio_cell_type,
)
from jax_fem.solver import solver

from inherent_solver.config import (
    resolve_path,
    get_paths_base,
    validate_config,
    prepare_output_directories,
)
from inherent_solver.boundary import (
    create_displacement_bc,
    update_quiet_boundary_conditions,
)
from inherent_solver.toolpath import (
    load_toolpath,
    create_layerwise_inherent_strain,
)
from inherent_solver.schedule import (
    precompute_activation_schedule,
)
from inherent_solver.problem import (
    InherentStrainPlasticity,
)
from inherent_solver.native_solver import native_linear_solver_options
from inherent_solver.output import (
    save_forward_result,
)
from inherent_solver.cutting import (
    run_cutting_by_cell_silence,
    run_cutting_by_release,
)


def load_mesh(
    config,
    project_root,
):
    paths_base = get_paths_base(
        config,
        project_root,
    )

    mesh_path = resolve_path(
        paths_base,
        config["paths"]["mesh"],
    )

    if not mesh_path.exists():
        raise FileNotFoundError(
            f"找不到 Abaqus 网格：{mesh_path}"
        )

    element_type = config[
        "mesh"
    ].get(
        "element_type",
        "HEX8",
    )

    cell_type = get_meshio_cell_type(
        element_type
    )

    scale_to_meter = float(
        config["mesh"].get(
            "scale_to_meter",
            1.0,
        )
    )

    print(
        f"Loading Abaqus mesh: {mesh_path}"
    )

    meshio_mesh = meshio.read(
        str(mesh_path)
    )

    if cell_type not in meshio_mesh.cells_dict:
        raise KeyError(
            f"网格中没有单元类型 {cell_type}。"
            f"已有类型："
            f"{list(meshio_mesh.cells_dict.keys())}"
        )

    return Mesh(
        meshio_mesh.points
        * scale_to_meter,
        meshio_mesh.cells_dict[
            cell_type
        ],
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
    }

    # Do not put a missing or intentionally disabled guess into the options
    # closure.  In an AD scan, a previous increment's solution can be a JAX
    # tracer; capturing it here makes the next ad_wrapper call leak that
    # tracer across transformation scopes.
    if initial_guess is not None:
        newton_options["initial_guess"] = [initial_guess]

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


def get_cutting_jobs(cutting_config):
    if not cutting_config.get(
        "enabled",
        False,
    ):
        return []

    jobs = cutting_config.get("jobs")

    if not jobs:
        return [cutting_config]

    shared_config = {
        key: value
        for key, value in cutting_config.items()
        if key not in {"enabled", "jobs", "tag"}
    }

    return [
        {
            **shared_config,
            **job_config,
        }
        for job_config in jobs
    ]


def normalize_cutting_mode(mode):
    mode = str(
        mode
    ).lower()

    aliases = {
        "node_release": "node_release",
        "release": "node_release",
        "duplicate_nodes": "node_release",
        "node_duplicate": "node_release",
        "copy_nodes": "node_release",
        "cell_silence": "cell_silence",
        "silent_cells": "cell_silence",
        "quiet_cells": "cell_silence",
        "incremental": "cell_silence",
    }

    if mode not in aliases:
        raise ValueError(
            f"不支持的 cutting.mode：{mode}。"
            "可用模式：node_release 或 cell_silence。"
        )

    return aliases[mode]


def run_cutting_simulations(
    mesh,
    toolpath,
    displacement,
    stress,
    plastic_strain,
    config,
    cutting_config,
    output_paths,
    vtk_paths,
):
    cutting_jobs = get_cutting_jobs(
        cutting_config
    )

    if not cutting_jobs:
        return "forward"

    current_mesh = mesh
    current_displacement = displacement
    current_stress = stress
    current_plastic_strain = plastic_strain
    current_retained_cell_mask = None
    last_source = "forward"

    for index, job_config in enumerate(
        cutting_jobs,
        start=1,
    ):
        tag = job_config.get(
            "tag",
            f"release_{index}",
        )
        job_config = {
            **job_config,
            "tag": tag,
        }

        source_mesh = current_mesh
        source_displacement = (
            current_displacement
        )
        source_stress = current_stress
        source_plastic_strain = (
            current_plastic_strain
        )

        mode = normalize_cutting_mode(
            job_config.get(
                "mode",
                "node_release",
            )
        )

        if mode == "node_release":
            (
                cutting_solution,
                cutting_stress,
                cutting_plastic_strain,
                cutting_problem,
            ) = run_cutting_by_release(
                original_mesh=source_mesh,
                toolpath=toolpath,
                displacement=(
                    source_displacement
                ),
                stress=source_stress,
                plastic_strain=(
                    source_plastic_strain
                ),
                material_config=(
                    config["material"]
                ),
                geometry_config=(
                    config["geometry"]
                ),
                activation_config=(
                    config["activation"]
                ),
                cutting_config=job_config,
                solver_config=(
                    config["solver"]["cutting"]
                ),
                vtk_directory=(
                    output_paths[
                        "vtk_directory"
                    ]
                ),
                displacement_scale=float(
                    config["output"].get(
                        "displacement_scale",
                        1000.0,
                    )
                ),
            )

            current_retained_cell_mask = None
        else:
            (
                cutting_solution,
                cutting_stress,
                cutting_plastic_strain,
                cutting_problem,
                current_retained_cell_mask,
            ) = run_cutting_by_cell_silence(
                original_mesh=source_mesh,
                toolpath=toolpath,
                displacement=(
                    source_displacement
                ),
                stress=source_stress,
                plastic_strain=(
                    source_plastic_strain
                ),
                material_config=(
                    config["material"]
                ),
                geometry_config=(
                    config["geometry"]
                ),
                activation_config=(
                    config["activation"]
                ),
                cutting_config=job_config,
                solver_config=(
                    config["solver"]["cutting"]
                ),
                vtk_directory=(
                    output_paths[
                        "vtk_directory"
                    ]
                ),
                retained_cell_mask=(
                    current_retained_cell_mask
                ),
                displacement_scale=float(
                    config["output"].get(
                        "displacement_scale",
                        1000.0,
                    )
                ),
            )

        vtk_path = (
            output_paths["vtk_directory"]
            / f"step_{tag}_final.vtu"
        )
        vtk_paths[tag] = vtk_path
        vtk_paths[f"cutting_{index}"] = vtk_path
        vtk_paths["cutting"] = vtk_path
        last_source = "cutting"

        current_mesh = cutting_problem.fe
        current_displacement = (
            cutting_solution
        )
        current_stress = cutting_stress
        current_plastic_strain = (
            cutting_plastic_strain
        )

    return last_source


def create_problem(
    mesh,
    schedule,
    config,
):
    if len(schedule) == 0:
        raise ValueError(
            "激活计划为空，无法创建 Problem。"
        )

    initial_quiet_point_indices = (
        schedule[0][
            "quiet_point_indices"
        ]
    )

    geometry_config = config[
        "geometry"
    ]

    displacement_bc_info = (
        create_displacement_bc(
            quiet_point_indices=(
                initial_quiet_point_indices
            ),
            fixed_plane_z=(
                geometry_config[
                    "fixed_plane_z_m"
                ]
            ),
            fixed_plane_tolerance=(
                geometry_config[
                    "fixed_plane_tolerance_m"
                ]
            ),
        )
    )

    return InherentStrainPlasticity(
        mesh,
        vec=3,
        dim=3,
        dirichlet_bc_info=(
            displacement_bc_info
        ),
        material_config=config[
            "material"
        ],
    )


def run_forward_simulation(
    problem,
    schedule,
    layerwise_inherent_strain,
    config,
    vtk_directory,
):
    print("\n" + "=" * 60)
    print(
        "Starting forward inherent-strain simulation"
    )
    print("=" * 60)

    number_of_layers = len(
        schedule
    )

    number_of_increments = int(
        config["inherent_strain"][
            "increments_per_layer"
        ]
    )

    save_each_increment = bool(
        config["inherent_strain"].get(
            "save_each_increment",
            False,
        )
    )

    displacement_scale = float(
        config["output"].get(
            "displacement_scale",
            1000.0,
        )
    )

    forward_solver_config = (
        config["solver"]["forward"]
    )

    number_of_cells = len(
        problem.fe.cells
    )

    number_of_quads = (
        problem.fe.num_quads
    )

    displacement = np.zeros(
        (
            problem.fes[0].num_total_nodes,
            problem.fes[0].vec,
        )
    )

    stress = np.zeros(
        (
            number_of_cells,
            number_of_quads,
            problem.dim,
            problem.dim,
        )
    )

    plastic_strain = np.zeros(
        (
            number_of_cells,
            number_of_quads,
        )
    )

    for step_info in schedule:
        step_index = step_info[
            "step_index"
        ]

        print(
            f"\nLayer {step_index} / "
            f"{number_of_layers}"
        )

        active_mask_bool = step_info[
            "active_mask"
        ]

        new_cell_indices = step_info[
            "new_cell_indices"
        ]

        quiet_point_indices = step_info[
            "quiet_point_indices"
        ]

        update_quiet_boundary_conditions(
            problem,
            quiet_point_indices,
        )

        total_layer_strain = (
            layerwise_inherent_strain[
                step_index - 1
            ]
        )

        strain_increment = (
            total_layer_strain
            / float(number_of_increments)
        )

        print(
            "  Total layer strain: "
            f"{onp.asarray(total_layer_strain)}"
        )

        print(
            "  Strain per increment: "
            f"{onp.asarray(strain_increment)}"
        )

        print(
            "  New cells: "
            f"{len(new_cell_indices)}"
        )

        active_mask = np.asarray(
            active_mask_bool,
            dtype=np.float64,
        )[:, None, None]

        active_mask = np.repeat(
            active_mask,
            number_of_quads,
            axis=1,
        )

        cut_mask = np.zeros(
            (
                number_of_cells,
                number_of_quads,
                1,
            )
        )

        relaxation_factor = np.zeros_like(
            cut_mask
        )

        for increment_index in range(
            number_of_increments
        ):
            print(
                f"  Increment "
                f"{increment_index + 1} / "
                f"{number_of_increments}"
            )

            old_displacement_gradient = (
                problem.compute_u_grad(
                    displacement
                )
            )

            inherent_strain_increment = (
                np.zeros(
                    (
                        number_of_cells,
                        number_of_quads,
                        3,
                    )
                )
            )

            if len(new_cell_indices) > 0:
                inherent_strain_increment = (
                    inherent_strain_increment.at[
                        new_cell_indices,
                        :,
                        0,
                    ].set(
                        strain_increment[0]
                    )
                )

                inherent_strain_increment = (
                    inherent_strain_increment.at[
                        new_cell_indices,
                        :,
                        1,
                    ].set(
                        strain_increment[1]
                    )
                )

                inherent_strain_increment = (
                    inherent_strain_increment.at[
                        new_cell_indices,
                        :,
                        2,
                    ].set(
                        strain_increment[2]
                    )
                )

            params = [
                inherent_strain_increment,
                active_mask,
                cut_mask,
                relaxation_factor,
                old_displacement_gradient,
                stress,
                plastic_strain,
            ]

            problem.set_params(
                params
            )

            solver_options = (
                create_solver_options(
                    forward_solver_config,
                    displacement,
                )
            )

            solution = solver(
                problem,
                solver_options=(
                    solver_options
                ),
            )[0]

            (
                current_stress,
                current_plastic_strain,
                current_elastic_strain,
            ) = problem.update_stress_strain(
                solution,
                params,
            )

            # Commit the history state immediately after each subincrement.
            displacement = solution
            stress = current_stress
            plastic_strain = (
                current_plastic_strain
            )

            maximum_displacement = float(
                np.max(
                    np.abs(displacement)
                )
            )

            maximum_plastic_strain = float(
                np.max(
                    plastic_strain
                )
            )

            print(
                "    max|u| = "
                f"{maximum_displacement:.6e}, "
                "max p = "
                f"{maximum_plastic_strain:.6e}"
            )

            if save_each_increment:
                vtk_path = (
                    vtk_directory
                    / (
                        f"step_{step_index:03d}_"
                        f"inc_"
                        f"{increment_index + 1:03d}"
                        f".vtu"
                    )
                )

                save_forward_result(
                    problem=problem,
                    solution=displacement,
                    stress=stress,
                    plastic_strain=(
                        plastic_strain
                    ),
                    elastic_strain=(
                        current_elastic_strain
                    ),
                    active_mask=(
                        active_mask_bool
                    ),
                    vtk_path=vtk_path,
                    displacement_scale=(
                        displacement_scale
                    ),
                )

        # By default, output once after all subincrements of a layer are complete.
        vtk_path = (
            vtk_directory
            / f"step_{step_index:03d}.vtu"
        )

        save_forward_result(
            problem=problem,
            solution=displacement,
            stress=stress,
            plastic_strain=plastic_strain,
            elastic_strain=(
                current_elastic_strain
            ),
            active_mask=(
                active_mask_bool
            ),
            vtk_path=vtk_path,
            displacement_scale=(
                displacement_scale
            ),
        )

        print(
            f"  Saved layer result: "
            f"{vtk_path}"
        )

    print(
        "\nForward simulation completed."
    )

    return (
        displacement,
        stress,
        plastic_strain,
    )


def run_simulation(
    config,
    project_root,
):
    validate_config(config)

    print(
        "JAX backend:",
        jax.default_backend(),
    )

    print(
        "JAX devices:",
        jax.devices(),
    )

    output_paths = (
        prepare_output_directories(
            config,
            project_root,
        )
    )

    mesh = load_mesh(
        config,
        project_root,
    )

    toolpath = load_toolpath(
        config["toolpath"],
        paths_base=output_paths["paths_base"],
    )

    number_of_layers = (
        toolpath.shape[0] - 1
    )

    print(
        f"Detected number of layers: "
        f"{number_of_layers}"
    )

    schedule = (
        precompute_activation_schedule(
            mesh=mesh,
            toolpath=toolpath,
            base_plate_height=float(
                config["geometry"][
                    "base_plate_height_m"
                ]
            ),
            z_tolerance=float(
                config["activation"].get(
                    "z_tolerance_m",
                    1.0e-5,
                )
            ),
        )
    )

    problem = create_problem(
        mesh,
        schedule,
        config,
    )

    layerwise_inherent_strain = (
        create_layerwise_inherent_strain(
            config["inherent_strain"],
            number_of_layers,
        )
    )

    (
        final_displacement,
        final_stress,
        final_plastic_strain,
    ) = run_forward_simulation(
        problem=problem,
        schedule=schedule,
        layerwise_inherent_strain=(
            layerwise_inherent_strain
        ),
        config=config,
        vtk_directory=(
            output_paths["vtk_directory"]
        ),
    )

    forward_vtk_path = (
        output_paths["vtk_directory"]
        / f"step_{number_of_layers:03d}.vtu"
    )
    vtk_paths = {
        "forward": forward_vtk_path,
    }
    default_postprocess_source = "forward"

    cutting_config = config.get(
        "cutting",
        {},
    )

    default_postprocess_source = (
        run_cutting_simulations(
            mesh=mesh,
            toolpath=toolpath,
            displacement=(
                final_displacement
            ),
            stress=final_stress,
            plastic_strain=(
                final_plastic_strain
            ),
            config=config,
            cutting_config=cutting_config,
            output_paths=output_paths,
            vtk_paths=vtk_paths,
        )
    )

    postprocess_config = config.get(
        "postprocess",
        {},
    )
    if postprocess_config.get("enabled", False):
        from inherent_solver.postprocess import (
            run_line_postprocess,
        )

        run_line_postprocess(
            config=postprocess_config,
            paths_base=output_paths["paths_base"],
            vtk_paths=vtk_paths,
            default_source=(
                default_postprocess_source
            ),
        )

    print("\n" + "=" * 60)
    print(
        "Simulation finished successfully."
    )
    print("=" * 60)
