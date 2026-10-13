import csv

import meshio
import numpy as onp

import jax
import jax.numpy as np

from jax_fem import logger as jax_fem_logger
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.solver import solver

from thermo_mech.native_solver import native_linear_solver_options
from thermo_mech.config import (
    get_paths_base,
    resolve_path,
    prepare_output_directories,
    validate_config,
)
from thermo_mech.materials import MaterialDatabase
from thermo_mech.heat_source import GoldakHeatSource
from thermo_mech.toolpath import load_toolpath
from thermo_mech.boundary_conditions import (
    create_temperature_boundary_conditions,
    create_displacement_boundary_conditions,
    create_point_heat_analytical_boundary,
)
from thermo_mech.thermal_problem import Thermal
from thermo_mech.plasticity_problem import Plasticity
from thermo_mech.activation import (
    initialize_hash_map,
    update_hash_map,
    get_active_face_truth_table,
    get_quiet_point_mask,
)
from thermo_mech.output import save_step


def all_faces_fn(point):
    return True


def write_point_heat_vtu(path, points, cells, temperature_c, exact_surface_k, top_nodes):
    output_mesh = meshio.Mesh(
        points=points,
        cells={"hexahedron": cells},
    )
    temperature_c = onp.asarray(temperature_c).reshape(-1)
    temperature_k = temperature_c + 273.15
    output_mesh.point_data["temperature_c"] = temperature_c
    output_mesh.point_data["temperature_k"] = temperature_k
    output_mesh.point_data["analytical_surface_temperature_k"] = onp.where(
        top_nodes, exact_surface_k, onp.nan
    )
    output_mesh.write(str(path))


def create_mesh(config, project_root, mesh_output_path):
    mesh_config = config["mesh"]
    paths_base = get_paths_base(config, project_root)

    if mesh_config.get("mode") == "structured_box":
        geometry = config["geometry"]
        counts = [int(value) for value in mesh_config["cells_xyz"]]
        lengths = [
            float(geometry["length_x_m"]),
            float(geometry["length_y_m"]),
            float(geometry["length_z_m"]),
        ]
        meshio_mesh = box_mesh(*counts, *lengths)
        cell_type = get_meshio_cell_type(mesh_config.get("element_type", "HEX8"))
        mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
        meshio_mesh.write(str(mesh_output_path))
        return mesh

    element_type = mesh_config.get("element_type", "HEX8")

    cell_type = get_meshio_cell_type(element_type)

    abaqus_path = resolve_path(paths_base, config["paths"]["mesh"])

    if not abaqus_path.exists():
        raise FileNotFoundError(f"找不到网格文件：{abaqus_path}")

    meshio_mesh = meshio.read(str(abaqus_path))

    scale_to_meter = float(mesh_config.get("scale_to_meter", 1.0))

    mesh = Mesh(meshio_mesh.points * scale_to_meter, meshio_mesh.cells_dict[cell_type])

    scaled_meshio_mesh = meshio.Mesh(
        points=meshio_mesh.points * scale_to_meter,
        cells={cell_type: meshio_mesh.cells_dict[cell_type]},
    )

    scaled_meshio_mesh.write(str(mesh_output_path))

    return mesh


def create_solver_options(solver_config, initial_guess, mechanical=False):
    nonlinear_options = {
        **native_linear_solver_options(solver_config),
        "initial_guess": initial_guess,
        "line_search_flag": bool(solver_config.get("line_search", True)),
        "tol": float(solver_config.get("tolerance", 1.0e-6)),
        "rel_tol": float(solver_config.get("relative_tolerance", 1.0e-8)),
    }

    return nonlinear_options


def make_thermal_initial_guess(
    temperature,
    dt,
    laser_switch,
    initial_temperature,
    heat_source,
    materials,
    z_reference,
    predictor_config,
):
    """Build a temperature predictor for the next thermal increment.

    The estimate follows the legacy TMQE predictor: use the peak Goldak
    volumetric power and a reference volumetric heat capacity to produce a
    positive temperature increment. The predictor is only used while the
    laser is on; cooling keeps the existing ambient-temperature floor.
    """
    if not laser_switch:
        return np.maximum(temperature, initial_temperature)

    if not predictor_config.get("enabled", False):
        return temperature

    reference_heat_capacity = float(
        predictor_config.get("reference_heat_capacity_j_kgk", 500.0)
    )

    if reference_heat_capacity <= 0.0:
        raise ValueError(
            "solver.thermal.initial_guess_predictor.reference_heat_capacity_j_kgk "
            "必须大于 0"
        )

    predictor_scale = float(predictor_config.get("scale", 1.0))

    if predictor_scale < 0.0:
        raise ValueError("solver.thermal.initial_guess_predictor.scale 不能小于 0")

    # Keep the legacy normalization (6 / pi**1.5). The actual residual uses
    # the full Goldak expression; this predictor is only an initial estimate.
    normalization = 6.0 / (onp.pi**1.5)
    front_peak = heat_source.front_fraction / (
        heat_source.front_radius_x * heat_source.radius_y * heat_source.radius_z
    )
    rear_peak = heat_source.rear_fraction / (
        heat_source.rear_radius_x * heat_source.radius_y * heat_source.radius_z
    )
    peak_power_density = (
        heat_source.efficiency
        * heat_source.power
        * normalization
        * max(front_peak, rear_peak)
    )

    density = float(onp.asarray(materials.density(z_reference)))
    positive_small = (
        predictor_scale
        * float(dt)
        * peak_power_density
        / (density * reference_heat_capacity)
    )

    max_temperature_rise = predictor_config.get("max_temperature_rise_c")

    if max_temperature_rise is not None:
        max_temperature_rise = float(max_temperature_rise)

        if max_temperature_rise < 0.0:
            raise ValueError(
                "solver.thermal.initial_guess_predictor.max_temperature_rise_c "
                "不能小于 0"
            )

        positive_small = min(positive_small, max_temperature_rise)

    return np.maximum(temperature + positive_small, temperature)


def run_thermal_only_case(config, mesh, materials, heat_source, output_paths):
    """Run a thermal-only benchmark through the production AM Thermal problem."""
    geometry = config["geometry"]
    thermal_config = config["thermal"]
    benchmark = config["thermal_benchmark"]
    initial_temperature = float(thermal_config["initial_temperature_c"])
    lengths = [
        float(geometry["length_x_m"]),
        float(geometry["length_y_m"]),
        float(geometry["length_z_m"]),
    ]
    source = config["heat_source"]["analytical_reference"]
    analytical_boundary = create_point_heat_analytical_boundary(
        geometry, source, config["thermal_benchmark"]["material"], initial_temperature
    )
    temperature_bc_info = create_temperature_boundary_conditions(
        initial_temperature, analytical_boundary
    )
    thermal_config = {**thermal_config, "preserve_case_dirichlet": True}
    problem_temperature = Thermal(
        mesh,
        vec=1,
        dim=3,
        dirichlet_bc_info=temperature_bc_info,
        location_fns=[all_faces_fn],
        materials=materials,
        heat_source=heat_source,
        thermal_config=thermal_config,
        geometry_config=geometry,
    )

    active_cells = onp.ones(problem_temperature.fe.num_cells, dtype=bool)
    external_faces, _, _, _, _ = initialize_hash_map(problem_temperature, active_cells)
    top_z = lengths[2]
    active_faces = get_active_face_truth_table(
        problem_temperature, external_faces, laser_center_z=top_z
    )
    quiet_mask = get_quiet_point_mask(problem_temperature, active_cells)
    temperature = initial_temperature * np.ones((len(mesh.points), 1))
    center = np.array([lengths[0] / 2.0, lengths[1] / 2.0, top_z])

    solver_config = config["solver"]["thermal"]
    dt = float(benchmark["time_step_s"])
    final_time = float(benchmark["final_time_s"])
    if dt <= 0.0 or final_time <= 0.0:
        raise ValueError("thermal_benchmark 时间步和终止时间必须大于 0")
    step_ratio = final_time / dt
    nearest_steps = round(step_ratio)
    if onp.isclose(step_ratio, nearest_steps, rtol=1.0e-12, atol=1.0e-12):
        number_of_steps = max(1, int(nearest_steps))
    else:
        number_of_steps = max(1, int(onp.ceil(step_ratio)))
    time_epsilon = max(final_time, dt) * 1.0e-12
    output_interval = int(config.get("output", {}).get("temperature_interval_steps", 10))
    if output_interval <= 0:
        raise ValueError("output.temperature_interval_steps 必须大于 0")

    points = onp.asarray(problem_temperature.fe.points)
    cells = onp.asarray(problem_temperature.fe.cells)
    radial_squared = (points[:, 0] - lengths[0] / 2.0) ** 2 + (points[:, 1] - lengths[1] / 2.0) ** 2
    r0 = float(source["radius_m"])
    from scipy.special import i0

    exact_surface = float(source["initial_temperature_k"]) + (
        float(source["absorption"]) * float(source["power_w"])
        / (onp.sqrt(2.0 * onp.pi) * float(source["conductivity_w_mk"]) * r0)
        * onp.exp(-radial_squared / r0**2)
        * i0(radial_squared / r0**2)
    )
    top_nodes = onp.isclose(points[:, 2], top_z, atol=min(lengths) * 1.0e-8)

    options = create_solver_options(solver_config, [temperature])
    print(
        f"AM Thermal benchmark: cells={problem_temperature.fe.num_cells}, "
        f"dt={dt:.3e} s, steps={number_of_steps}, "
        f"temperature output interval={output_interval} steps"
    )

    previous_log_level = jax_fem_logger.level
    jax_fem_logger.setLevel("ERROR")
    try:
        for step in range(number_of_steps):
            remaining_time = final_time - step * dt
            if remaining_time <= time_epsilon:
                break
            step_dt = min(dt, remaining_time)
            problem_temperature.set_params([
                temperature,
                step_dt,
                center,
                1,
                quiet_mask,
                active_cells,
                active_faces,
            ])
            options["initial_guess"] = [temperature]
            temperature = solver(problem_temperature, solver_options=options)[0]

            if (step + 1) % output_interval == 0 or step + 1 == number_of_steps:
                current_time = min(final_time, (step + 1) * dt)
                maximum_temperature_k = float(np.max(temperature) + 273.15)
                snapshot_path = output_paths["vtk_dir"] / f"point_heat_step_{step + 1:05d}.vtu"
                write_point_heat_vtu(
                    snapshot_path,
                    points,
                    cells,
                    temperature,
                    exact_surface,
                    top_nodes,
                )
                print(
                    f"thermal step {step + 1}/{number_of_steps}, "
                    f"t={current_time:.6e} s, Tmax={maximum_temperature_k:.6g} K, "
                    f"VTU={snapshot_path.name}"
                )
    finally:
        jax_fem_logger.setLevel(previous_log_level)

    output_path = output_paths["vtk_dir"] / "point_heat_final.vtu"
    write_point_heat_vtu(
        output_path,
        points,
        cells,
        temperature,
        exact_surface,
        top_nodes,
    )
    temperature_c = onp.asarray(temperature).reshape(-1)
    temperature_k = temperature_c + 273.15
    error = temperature_k[top_nodes] - exact_surface[top_nodes]
    summary = {
        "problem": "am_solver.thermo_mech.thermal_problem.Thermal",
        "final_time_s": final_time,
        "time_step_s": dt,
        "time_steps": number_of_steps,
        "cell_count": int(problem_temperature.fe.num_cells),
        "center_temperature_k": float(temperature_k[onp.argmin(radial_squared + (points[:, 2] - top_z) ** 2)]),
        "center_analytical_temperature_k": float(exact_surface[onp.argmin(radial_squared + (points[:, 2] - top_z) ** 2)]),
        "surface_rmse_k": float(onp.sqrt(onp.mean(error**2))),
        "surface_max_absolute_error_k": float(onp.max(onp.abs(error))),
        "vtu": str(output_path),
    }
    import json

    with (output_paths["output_dir"] / "thermal_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Thermal benchmark summary: {summary}")
    return temperature


def solve_mechanics_and_save(
    problem_temperature,
    problem_mechanics,
    temperature,
    old_temperature_for_mechanics,
    displacement_solution,
    mechanics_params,
    active_cell_truth_table,
    current_time,
    output_step,
    vtk_directory,
    mechanics_solver_config,
):
    active_cell_quad = np.repeat(active_cell_truth_table[:, None], problem_mechanics.fe.num_quads, axis=1)

    # Use a fixed-length mask for mechanics as well to avoid compilation overhead from changing quiet-index array lengths.
    quiet_point_mask = get_quiet_point_mask(problem_mechanics, active_cell_truth_table)

    mechanics_params[6] = quiet_point_mask

    mechanics_params[7] = active_cell_quad

    mechanics_params = (
        problem_mechanics.update_dT_and_phase(
            (temperature - old_temperature_for_mechanics),
            temperature,
            mechanics_params,
        )
    )

    problem_mechanics.set_params(mechanics_params)

    mechanics_options = create_solver_options(mechanics_solver_config, displacement_solution, mechanical=True)

    displacement_solution = solver(problem_mechanics, solver_options=mechanics_options)

    print("Displacement solved")

    (
        mechanics_params,
        plastic_information,
    ) = problem_mechanics.update_stress_strain(displacement_solution[0], mechanics_params)

    vtk_path = vtk_directory / f"step_{output_step:05d}.vtu"

    save_step(
        problem_temperature=problem_temperature,
        temperature=temperature,
        displacement_solution=displacement_solution,
        plastic_information=plastic_information,
        mechanics_params=mechanics_params,
        active_cell_truth_table=(active_cell_truth_table),
        current_time=current_time,
        vtk_path=vtk_path,
    )

    return displacement_solution, mechanics_params, temperature, output_step + 1


def release_springback_constraints(problem_mechanics, mechanics_config):
    """Release configured thin-wall anchors while retaining rigid-body restraints."""
    springback_config = mechanics_config.get("springback", {})
    boundary_config = mechanics_config.get("boundary_conditions", {})

    if boundary_config.get("type") != "thin_wall":
        raise ValueError("Springback is currently configured only for thin_wall boundary conditions.")

    release_anchor_ids = springback_config.get("release_anchor_ids", [1, 3])
    released_location = lambda point: False

    # Entries 0--2 are quiet-element constraints and entry 3 is the y=0
    # symmetry constraint. Every following anchor contributes x, y, and z.
    for anchor_id in release_anchor_ids:
        anchor_id = int(anchor_id)

        if anchor_id not in (1, 2, 3):
            raise ValueError(f"Unsupported thin-wall springback anchor: {anchor_id}")

        first_index = 4 + 3 * (anchor_id - 1)

        for location_index in range(first_index, first_index + 3):
            problem_mechanics.fe.dirichlet_bc_info[0][location_index] = released_location

    problem_mechanics.fe.update_Dirichlet_boundary_conditions(problem_mechanics.fe.dirichlet_bc_info)


def write_springback_summary(summary_path, displacement_before, displacement_after, points):
    springback_displacement = onp.asarray(displacement_after - displacement_before)
    magnitudes = onp.linalg.norm(springback_displacement, axis=1)
    maximum_index = int(onp.argmax(magnitudes))
    maximum_displacement = springback_displacement[maximum_index]
    maximum_point = onp.asarray(points[maximum_index])

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "maximum_springback_mm",
                "ux_at_max_mm",
                "uy_at_max_mm",
                "uz_at_max_mm",
                "point_x_m",
                "point_y_m",
                "point_z_m",
                "point_index",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "maximum_springback_mm": float(magnitudes[maximum_index] * 1.0e3),
            "ux_at_max_mm": float(maximum_displacement[0] * 1.0e3),
            "uy_at_max_mm": float(maximum_displacement[1] * 1.0e3),
            "uz_at_max_mm": float(maximum_displacement[2] * 1.0e3),
            "point_x_m": float(maximum_point[0]),
            "point_y_m": float(maximum_point[1]),
            "point_z_m": float(maximum_point[2]),
            "point_index": maximum_index,
        })

    return float(magnitudes[maximum_index] * 1.0e3), maximum_index


def solve_springback_and_save(
    problem_temperature,
    problem_mechanics,
    temperature,
    displacement_solution,
    mechanics_params,
    active_cell_truth_table,
    current_time,
    output_paths,
    mechanics_config,
    points,
):
    constrained_displacement = np.array(displacement_solution[0])

    release_springback_constraints(problem_mechanics, mechanics_config)
    problem_mechanics.set_params(mechanics_params)

    mechanics_options = create_solver_options(
        mechanics_config,
        displacement_solution,
        mechanical=True,
    )
    released_displacement_solution = solver(problem_mechanics, solver_options=mechanics_options)

    updated_params, plastic_information = problem_mechanics.update_stress_strain(
        released_displacement_solution[0],
        mechanics_params,
    )
    springback_displacement = released_displacement_solution[0] - constrained_displacement
    vtk_path = output_paths["vtk_dir"] / "springback_final.vtu"

    save_step(
        problem_temperature=problem_temperature,
        temperature=temperature,
        displacement_solution=released_displacement_solution,
        plastic_information=plastic_information,
        mechanics_params=updated_params,
        active_cell_truth_table=active_cell_truth_table,
        current_time=current_time,
        vtk_path=vtk_path,
        additional_point_infos=[("u_springback", springback_displacement * 1.0e3)],
    )

    summary_path = output_paths["output_dir"] / "springback_summary.csv"
    maximum_springback_mm, maximum_index = write_springback_summary(
        summary_path,
        constrained_displacement,
        released_displacement_solution[0],
        points,
    )
    print(
        "Springback solved: "
        f"maximum displacement = {maximum_springback_mm:.6g} mm at point {maximum_index}."
    )
    print(f"Springback VTK: {vtk_path}")
    print(f"Springback summary: {summary_path}")

    return released_displacement_solution, updated_params


def run_simulation(config, project_root):
    validate_config(config)

    print("default_backend: ", jax.default_backend())
    print("devices:", jax.devices())
    print("local_device_count: ", jax.local_device_count())

    output_paths = prepare_output_directories(config, project_root)

    mesh = create_mesh(config, project_root, output_paths["mesh_output_path"])

    geometry_config = config["geometry"]
    thermal_config = config["thermal"]

    initial_temperature = float(thermal_config["initial_temperature_c"])

    interface_z = float(geometry_config["material_interface_z_m"])

    materials = MaterialDatabase(config=config["materials"], interface_z=interface_z)

    heat_source = GoldakHeatSource(config["heat_source"])

    if config.get("mode") == "thermal_only":
        run_thermal_only_case(config, mesh, materials, heat_source, output_paths)
        return

    mechanics_config = config["mechanics"]
    base_plate_height = float(geometry_config.get("base_plate_height_m", 0.0))

    toolpath = load_toolpath(config["toolpath"], get_paths_base(config, project_root))

    print(f"Toolpath loaded: {toolpath.shape[0] - 1} segments, final time = {toolpath[-1, 0]:.6g} s")

    temperature_bc_info = create_temperature_boundary_conditions(initial_temperature)

    displacement_bc_info = (
        create_displacement_boundary_conditions(
            fixed_plane_z=float(mechanics_config["fixed_plane_z_m"]),
            tolerance=float(mechanics_config.get("fixed_plane_tolerance_m", 1.0e-4)),
            boundary_config=mechanics_config.get("boundary_conditions"),
        )
    )

    problem_temperature = Thermal(
        mesh,
        vec=1,
        dim=3,
        dirichlet_bc_info=temperature_bc_info,
        location_fns=[all_faces_fn],
        materials=materials,
        heat_source=heat_source,
        thermal_config=thermal_config,
        geometry_config=geometry_config,
    )

    problem_mechanics = Plasticity(
        mesh,
        vec=3,
        dim=3,
        dirichlet_bc_info=displacement_bc_info,
        materials=materials,
        mechanics_config=mechanics_config,
        phase_config=config.get("phase", {}),
    )

    cells = onp.asarray(problem_temperature.fe.cells)

    points = onp.asarray(problem_temperature.fe.points)

    number_of_points = len(mesh.points)
    number_of_cells = len(cells)

    centroids = onp.mean(points[cells], axis=1)

    active_cell_truth_table = onp.zeros(number_of_cells, dtype=bool)

    active_cell_truth_table[centroids[:, 2] <= base_plate_height] = True

    old_active_cell_truth_table = active_cell_truth_table.copy()

    (
        external_faces,
        cell_faces,
        face_hash_map,
        inner_faces,
        all_faces,
    ) = initialize_hash_map(problem_temperature, active_cell_truth_table)

    active_face_truth_table = (
        get_active_face_truth_table(problem_temperature, external_faces, laser_center_z=base_plate_height)
    )

    # Keep the quiet mask length equal to the total node count and update only its boolean values to avoid JIT recompilation.
    quiet_temperature_mask = get_quiet_point_mask(problem_temperature, active_cell_truth_table)

    temperature = initial_temperature * np.ones((number_of_points, 1))

    old_temperature_for_mechanics = np.array(temperature)

    displacement_solution = [np.zeros((problem_mechanics.fes[0].num_total_nodes, problem_mechanics.fes[0].vec))]

    number_of_mechanical_quads = problem_mechanics.fe.num_quads

    u_grad_old = np.zeros((number_of_cells, number_of_mechanical_quads, 3, 3))

    sigmas_old = np.zeros_like(u_grad_old)

    plastic_strain_old = np.zeros((number_of_cells, number_of_mechanical_quads))

    back_stress_old = np.zeros_like(u_grad_old)

    delta_temperature = np.zeros((number_of_cells, number_of_mechanical_quads, 1))

    temperature_quad = initial_temperature * np.ones_like(delta_temperature)

    powder_id = int(config.get("phase", {}).get("powder", 0))

    phase = np.ones_like(delta_temperature, dtype=np.int32) * powder_id

    active_cell_quad = np.repeat(active_cell_truth_table[:, None], number_of_mechanical_quads, axis=1)

    # Use a fixed-length mask for the mechanical problem to keep Dirichlet location function input shapes stable.
    quiet_mechanics_mask = get_quiet_point_mask(problem_mechanics, active_cell_truth_table)

    mechanics_params = [
        u_grad_old,
        sigmas_old,
        plastic_strain_old,
        back_stress_old,
        delta_temperature,
        phase,
        quiet_mechanics_mask,
        active_cell_quad,
        temperature_quad,
        problem_mechanics.x_quad,
    ]

    time_config = config["time_stepping"]

    laser_on_dt_s = time_config.get("laser_on_dt_s")
    if laser_on_dt_s is not None:
        laser_on_dt_s = float(laser_on_dt_s)
        if laser_on_dt_s <= 0.0:
            raise ValueError("time_stepping.laser_on_dt_s 必须大于 0")
        number_of_laser_on_steps = None
    else:
        number_of_laser_on_steps = int(time_config["laser_on_steps_per_segment"])
        if number_of_laser_on_steps <= 0:
            raise ValueError("time_stepping.laser_on_steps_per_segment 必须大于 0")

    number_of_laser_off_steps = int(time_config["laser_off_steps_per_segment"])
    if number_of_laser_off_steps <= 0:
        raise ValueError("time_stepping.laser_off_steps_per_segment 必须大于 0")

    mechanics_interval = int(time_config.get("mechanics_interval", 1))

    thermal_solver_config = config["solver"]["thermal"]

    thermal_predictor_config = thermal_solver_config.get(
        "initial_guess_predictor", {}
    )

    mechanics_solver_config = config["solver"]["mechanics"]

    active_radius_factor = float(config["activation"].get("active_radius_factor", 2.0))

    active_radius_x = active_radius_factor * heat_source.front_radius_x

    active_radius_y = active_radius_factor * heat_source.radius_y

    laser_center = np.array([0.0, 0.0, base_plate_height])

    output_step = 0
    global_thermal_step = 0

    for segment_index in range(1, toolpath.shape[0]):
        segment = toolpath[segment_index]

        previous_time = toolpath[segment_index - 1, 0]

        segment_end_time = segment[0]
        laser_switch = int(segment[6])

        if laser_switch:
            segment_duration = float(segment_end_time - previous_time)
            if laser_on_dt_s is not None:
                # Keep the physical increment case-specific.  The final
                # substep may be shorter than the target to land exactly on
                # the toolpath segment end time.
                number_of_steps = max(1, int(onp.ceil(segment_duration / laser_on_dt_s)))
                print(
                    f"laser_on_dt_target ={laser_on_dt_s:.6e} s, "
                    f"num_dt ={number_of_steps}"
                )
            else:
                number_of_steps = number_of_laser_on_steps
                print(f"num_dt ={number_of_steps}")
        else:
            number_of_steps = number_of_laser_off_steps

        time_values = onp.linspace(previous_time, segment_end_time, number_of_steps + 1)

        x_values = onp.linspace(segment[1], segment[2], number_of_steps + 1)

        y_values = onp.linspace(segment[3], segment[4], number_of_steps + 1)

        z_position = segment[5] + base_plate_height

        is_x_scan = not onp.isclose(segment[1], segment[2])

        is_y_scan = not onp.isclose(segment[3], segment[4])

        for local_step in range(number_of_steps):
            dt = time_values[local_step + 1] - time_values[local_step]

            current_time = time_values[local_step + 1]

            # Use the heat source position at the end of the time step.
            laser_center = np.array([x_values[local_step + 1], y_values[local_step + 1], z_position])

            print("\n" "############################################################")

            state_name = "on" if laser_switch else "off"

            print(
                f"Laser {state_name}: "
                f"segment = {segment_index}/"
                f"{toolpath.shape[0] - 1}, "
                f"step = {local_step}/"
                f"{number_of_steps - 1}"
            )

            print(f"laser center =" f"{laser_center}, dt ={dt}")

            if laser_switch:
                below_laser = centroids[:, 2] < laser_center[2]

                if is_x_scan:
                    inside_transverse_radius = (centroids[:, 1] - laser_center[1])**2 <= active_radius_y**2

                    inside_scan_radius = (centroids[:, 0] - laser_center[0])**2 <= active_radius_x**2

                elif is_y_scan:
                    inside_transverse_radius = (centroids[:, 0] - laser_center[0])**2 <= active_radius_x**2

                    inside_scan_radius = (centroids[:, 1] - laser_center[1])**2 <= active_radius_y**2

                else:
                    inside_transverse_radius = onp.zeros_like(below_laser, dtype=bool)

                    inside_scan_radius = onp.zeros_like(below_laser, dtype=bool)

                active_cell_truth_table = active_cell_truth_table | (below_laser & inside_transverse_radius & inside_scan_radius)

                if onp.all(active_cell_truth_table == old_active_cell_truth_table):
                    print("No element born")
                else:
                    external_faces = update_hash_map(
                        old_active_cell_truth_table,
                        active_cell_truth_table,
                        cell_faces,
                        face_hash_map,
                        inner_faces,
                        all_faces,
                    )

                    active_face_truth_table = (
                        get_active_face_truth_table(problem_temperature, external_faces, laser_center[2])
                    )

                    # Update only the fixed-length mask contents; do not generate variable-length quiet-index arrays.
                    quiet_temperature_mask = get_quiet_point_mask(problem_temperature, active_cell_truth_table)

            else:
                print("Maximum temperature at " "start of cooling: ", float(np.max(temperature)))

            initial_guess = make_thermal_initial_guess(
                temperature=temperature,
                dt=dt,
                laser_switch=laser_switch,
                initial_temperature=initial_temperature,
                heat_source=heat_source,
                materials=materials,
                z_reference=z_position,
                predictor_config=thermal_predictor_config,
            )

            problem_temperature.set_params([
                temperature,
                dt,
                laser_center,
                laser_switch,
                quiet_temperature_mask,
                active_cell_truth_table,
                active_face_truth_table,
            ])

            thermal_options = create_solver_options(thermal_solver_config, initial_guess, mechanical=False)

            temperature = solver(problem_temperature, solver_options=thermal_options)[0]

            print("Temperature solved")

            if not laser_switch:
                print("Maximum temperature at " "end of cooling: ", float(np.max(temperature)))

            global_thermal_step += 1

            if (global_thermal_step % mechanics_interval == 0):
                (
                    displacement_solution,
                    mechanics_params,
                    old_temperature_for_mechanics,
                    output_step,
                ) = solve_mechanics_and_save(
                    problem_temperature=(problem_temperature),
                    problem_mechanics=(problem_mechanics),
                    temperature=temperature,
                    old_temperature_for_mechanics=(old_temperature_for_mechanics),
                    displacement_solution=(displacement_solution),
                    mechanics_params=(mechanics_params),
                    active_cell_truth_table=(active_cell_truth_table),
                    current_time=current_time,
                    output_step=output_step,
                    vtk_directory=(output_paths["vtk_dir"]),
                    mechanics_solver_config=(mechanics_solver_config),
                )

            old_active_cell_truth_table = active_cell_truth_table.copy()

    springback_config = mechanics_config.get("springback", {})

    if springback_config.get("enabled", False):
        # A springback solve must start from the final constrained mechanical
        # state. Add a final mechanical update when the output cadence skipped
        # the last thermal increment.
        if global_thermal_step % mechanics_interval != 0:
            (
                displacement_solution,
                mechanics_params,
                old_temperature_for_mechanics,
                output_step,
            ) = solve_mechanics_and_save(
                problem_temperature=problem_temperature,
                problem_mechanics=problem_mechanics,
                temperature=temperature,
                old_temperature_for_mechanics=old_temperature_for_mechanics,
                displacement_solution=displacement_solution,
                mechanics_params=mechanics_params,
                active_cell_truth_table=active_cell_truth_table,
                current_time=current_time,
                output_step=output_step,
                vtk_directory=output_paths["vtk_dir"],
                mechanics_solver_config=mechanics_solver_config,
            )

        (
            displacement_solution,
            mechanics_params,
        ) = solve_springback_and_save(
            problem_temperature=problem_temperature,
            problem_mechanics=problem_mechanics,
            temperature=temperature,
            displacement_solution=displacement_solution,
            mechanics_params=mechanics_params,
            active_cell_truth_table=active_cell_truth_table,
            current_time=current_time,
            output_paths=output_paths,
            mechanics_config=mechanics_config,
            points=points,
        )

    print("\n" "############################################################")
    print("Simulation finished successfully.")
