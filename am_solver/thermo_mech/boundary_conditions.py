import jax.numpy as np


def create_temperature_boundary_conditions(initial_temperature, analytical_boundary=None):
    def dirichlet_value(point):
        return initial_temperature

    # Standard AM cases have no fixed temperature boundary; quiet element
    # nodes are added during activation. A benchmark may add a case-specific
    # fixed boundary without changing the thermal problem formulation.
    if analytical_boundary is None:
        location_functions = [lambda point: False]
        value_functions = [dirichlet_value]
    else:
        location_functions, value_functions = analytical_boundary

    dof_indices = [0]

    return[location_functions, dof_indices, value_functions]


def create_point_heat_analytical_boundary(geometry, source, material, initial_temperature_c):
    """Build analytical side/bottom Dirichlet data for the point-heat case."""
    import numpy as onp
    from scipy.special import j0
    import jax.numpy as np

    lx = float(geometry["length_x_m"])
    ly = float(geometry["length_y_m"])
    lz = float(geometry["length_z_m"])
    radius = float(source["radius_m"])
    conductivity = float(material["conductivity_w_mk"])
    power = float(source["power_w"])
    absorption = float(source["absorption"])
    x_center, y_center = lx / 2.0, ly / 2.0

    radial_max = onp.hypot(lx / 2.0, ly / 2.0) / radius
    depth_max = lz / radius
    radial_grid = onp.linspace(0.0, radial_max, int(onp.ceil(radial_max / 0.01)) + 1)
    depth_grid = onp.linspace(0.0, depth_max, int(onp.ceil(depth_max / 0.01)) + 1)
    nodes, weights = onp.polynomial.legendre.leggauss(96)
    wave_number = (nodes + 1.0) * 10.0
    integration_weights = weights * 10.0
    bessel_values = j0(radial_grid[:, None] * wave_number[None, :])
    weighted_bessel = (
        integration_weights[None, :]
        * onp.exp(-wave_number[None, :] ** 2 / 8.0)
    ) * bessel_values
    integral = onp.exp(-depth_grid[:, None] * wave_number[None, :]) @ weighted_bessel.T
    temperature_table = (
        initial_temperature_c
        + absorption * power / (2.0 * onp.pi * conductivity * radius) * integral
    )

    radial_grid_jax = np.asarray(radial_grid)
    depth_grid_jax = np.asarray(depth_grid)
    temperature_table_jax = np.asarray(temperature_table)
    tolerance = min(lx, ly, lz) * 1.0e-8

    def on_side_or_bottom(point):
        return (
            np.isclose(point[0], 0.0, atol=tolerance)
            | np.isclose(point[0], lx, atol=tolerance)
            | np.isclose(point[1], 0.0, atol=tolerance)
            | np.isclose(point[1], ly, atol=tolerance)
            | np.isclose(point[2], 0.0, atol=tolerance)
        )

    def analytical_temperature(point):
        radial = np.sqrt((point[0] - x_center) ** 2 + (point[1] - y_center) ** 2) / radius
        depth = (lz - point[2]) / radius
        radial_index = np.clip(np.searchsorted(radial_grid_jax, radial, side="right") - 1, 0, len(radial_grid) - 2)
        depth_index = np.clip(np.searchsorted(depth_grid_jax, depth, side="right") - 1, 0, len(depth_grid) - 2)
        radial_fraction = (radial - radial_grid_jax[radial_index]) / (radial_grid_jax[radial_index + 1] - radial_grid_jax[radial_index])
        depth_fraction = (depth - depth_grid_jax[depth_index]) / (depth_grid_jax[depth_index + 1] - depth_grid_jax[depth_index])
        lower = temperature_table_jax[depth_index, radial_index] * (1.0 - radial_fraction) + temperature_table_jax[depth_index, radial_index + 1] * radial_fraction
        upper = temperature_table_jax[depth_index + 1, radial_index] * (1.0 - radial_fraction) + temperature_table_jax[depth_index + 1, radial_index + 1] * radial_fraction
        return lower * (1.0 - depth_fraction) + upper * depth_fraction

    def location(point):
        return on_side_or_bottom(point)

    return [location], [analytical_temperature]


def create_displacement_boundary_conditions(fixed_plane_z, tolerance, boundary_config=None):
    boundary_config = boundary_config or {}

    # The thin-wall model uses the same minimal constraints as the legacy
    # TMQE script: symmetry on y=0 and three anchor regions on the base.
    if boundary_config.get("type") == "thin_wall":
        symmetry_y = float(boundary_config.get("symmetry_y_m", 0.0))
        symmetry_tolerance = float(boundary_config.get("symmetry_tolerance_m", tolerance))

        def sym_plane_y(point):
            return np.isclose(point[1], symmetry_y, atol=symmetry_tolerance)

        def anchor_1(point):
            return (
                np.isclose(point[0], float(boundary_config.get("anchor_x_min_m", -0.25)), atol=tolerance)
                & np.isclose(point[1], float(boundary_config.get("anchor_y_m", 0.03)), atol=tolerance)
                & np.isclose(point[2], float(boundary_config.get("anchor_z_m", 0.0)), atol=tolerance)
            )

        def anchor_2(point):
            return (
                np.isclose(point[0], float(boundary_config.get("anchor_x_center_m", 0.0)), atol=tolerance)
                & (point[1] >= float(boundary_config.get("anchor_y_min_m", 0.02)))
                & (
                    np.isclose(point[2], float(boundary_config.get("anchor_z_m", 0.0)), atol=tolerance)
                    | np.isclose(point[2], float(boundary_config.get("anchor_z_bottom_m", -0.012)), atol=tolerance)
                )
            )

        def anchor_3(point):
            return (
                np.isclose(point[0], float(boundary_config.get("anchor_x_max_m", 0.25)), atol=tolerance)
                & np.isclose(point[1], float(boundary_config.get("anchor_y_m", 0.03)), atol=tolerance)
                & np.isclose(point[2], float(boundary_config.get("anchor_z_m", 0.0)), atol=tolerance)
            )

        location_functions = [lambda point: False] * 3 + [sym_plane_y]
        location_functions += [anchor_1] * 3 + [anchor_2] * 3 + [anchor_3] * 3
        dof_indices = [0, 1, 2] + [1] + [0, 1, 2] * 3
        value_functions = [lambda point: 0.0] * len(location_functions)
        return [location_functions, dof_indices, value_functions]

    def fixed_plane(point):
        return np.isclose(point[2], fixed_plane_z, atol=tolerance)

    def zero_displacement(point):
        return 0.0

    # The first three entries constrain quiet nodes.
    # The last three fix the bottom plane.
    location_functions = [lambda point: False] * 3 + [fixed_plane] * 3

    dof_indices = [0, 1, 2, 0, 1, 2]

    value_functions = [zero_displacement] * len(location_functions)

    return[location_functions, dof_indices, value_functions]
