import jax.numpy as np
import numpy as onp


def zero_displacement(point):
    return 0.0


def create_fixed_plane_fn(
    fixed_plane_z,
    tolerance,
):
    fixed_plane_z = float(fixed_plane_z)
    tolerance = float(tolerance)

    def fixed_plane(point):
        return np.isclose(
            point[2],
            fixed_plane_z,
            atol=tolerance,
        )

    return fixed_plane


def get_quiet_point_indices(
    cells,
    points,
    active_cell_truth_table,
):
    cells_np = onp.asarray(cells)
    points_np = onp.asarray(points)

    active_mask_np = onp.asarray(
        active_cell_truth_table,
        dtype=bool,
    )

    active_point_indices = set(
        cells_np[
            active_mask_np
        ].reshape(-1).tolist()
    )

    all_point_indices = set(
        onp.arange(
            len(points_np)
        ).tolist()
    )

    quiet_point_indices = onp.array(
        sorted(
            all_point_indices
            - active_point_indices
        ),
        dtype=onp.int32,
    )

    return quiet_point_indices


def get_dirichlet_location_fn(
    quiet_point_indices,
):
    def dirichlet_location_fn(
        point,
        index,
    ):
        return np.isin(
            index,
            quiet_point_indices,
        )

    return dirichlet_location_fn


def create_displacement_bc(
    quiet_point_indices,
    fixed_plane_z,
    fixed_plane_tolerance,
):
    quiet_location_fn = (
        get_dirichlet_location_fn(
            quiet_point_indices
        )
    )

    fixed_plane_fn = create_fixed_plane_fn(
        fixed_plane_z,
        fixed_plane_tolerance,
    )

    # The first three boundary functions constrain quiet nodes.
    # The last three functions fix the bottom plane.
    location_functions = (
        [quiet_location_fn] * 3
        + [fixed_plane_fn] * 3
    )

    dof_indices = [
        0, 1, 2,
        0, 1, 2,
    ]

    value_functions = [
        zero_displacement
    ] * 6

    return [
        location_functions,
        dof_indices,
        value_functions,
    ]


def update_quiet_boundary_conditions(
    problem,
    quiet_point_indices,
):
    quiet_location_fn = (
        get_dirichlet_location_fn(
            quiet_point_indices
        )
    )

    problem.fe.dirichlet_bc_info[0][0] = (
        quiet_location_fn
    )

    problem.fe.dirichlet_bc_info[0][1] = (
        quiet_location_fn
    )

    problem.fe.dirichlet_bc_info[0][2] = (
        quiet_location_fn
    )

    problem.fe.update_Dirichlet_boundary_conditions(
        problem.fe.dirichlet_bc_info
    )