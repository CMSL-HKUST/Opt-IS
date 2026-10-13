import numpy as np

from inherent_solver.boundary import (
    get_quiet_point_indices,
)


def precompute_activation_schedule(
    mesh,
    toolpath,
    base_plate_height,
    z_tolerance,
):
    cells = np.asarray(
        mesh.cells
    )

    points = np.asarray(
        mesh.points
    )

    centroids = np.mean(
        points[cells],
        axis=1,
    )

    active_mask = np.zeros(
        len(cells),
        dtype=bool,
    )

    # Activate the initial substrate.
    active_mask[
        centroids[:, 2]
        <= base_plate_height
    ] = True

    old_active_mask = (
        active_mask.copy()
    )

    schedule = []

    for step_index in range(
        1,
        toolpath.shape[0],
    ):
        current_z = (
            toolpath[step_index, 5]
            + base_plate_height
        )

        layer_mask = (
            centroids[:, 2]
            <= current_z + z_tolerance
        )

        active_mask = (
            active_mask
            | layer_mask
        )

        new_cell_indices = np.argwhere(
            active_mask
            & ~old_active_mask
        ).reshape(-1)

        quiet_point_indices = (
            get_quiet_point_indices(
                cells,
                points,
                active_mask,
            )
        )

        schedule.append({
            "step_index": step_index,
            "current_z": current_z,
            "active_mask": active_mask.copy(),
            "old_active_mask": (
                old_active_mask.copy()
            ),
            "new_cell_indices": (
                new_cell_indices
            ),
            "quiet_point_indices": (
                quiet_point_indices
            ),
        })

        old_active_mask = (
            active_mask.copy()
        )

    return schedule