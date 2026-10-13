import numpy as np


def hash_map_for_faces(active_cell_truth_table, cell_faces, face_hash_map, inner_faces, all_faces, cell_indices):
    for cell_id in cell_indices:
        if active_cell_truth_table[cell_id]:
            for face_id in range(len(cell_faces[cell_id])):
                key = tuple(cell_faces[cell_id, face_id].tolist())

                if key in face_hash_map:
                    inner_faces.append(face_hash_map[key])

                    inner_faces.append((cell_id, face_id))
                else:
                    face_hash_map[key] = cell_id, face_id

                all_faces.append((cell_id, face_id))

    external_faces = np.array(list(set(all_faces) - set(inner_faces)), dtype=np.int32)

    if external_faces.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    return external_faces.reshape(-1, 2)


def initialize_hash_map(problem, active_cell_truth_table):
    print("Initializing hash map for external faces...")

    face_indices = np.asarray(problem.fe.face_inds)

    cells = np.asarray(problem.fe.cells)

    cell_faces = np.sort(cells[:, face_indices], axis=- 1)

    face_hash_map = {}
    inner_faces = []
    all_faces = []

    cell_indices = np.arange(len(cell_faces))

    external_faces = hash_map_for_faces(
        active_cell_truth_table,
        cell_faces,
        face_hash_map,
        inner_faces,
        all_faces,
        cell_indices,
    )

    return external_faces, cell_faces, face_hash_map, inner_faces, all_faces


def update_hash_map(
    old_active_cell_truth_table,
    new_active_cell_truth_table,
    cell_faces,
    face_hash_map,
    inner_faces,
    all_faces,
):
    print("Updating hash map for external faces...")

    assert (
        np.sum(new_active_cell_truth_table)
 >= np.sum(old_active_cell_truth_table)
    ), ("Number of new born cells must be " "non - negative.")

    new_cell_indices = np.argwhere(old_active_cell_truth_table != new_active_cell_truth_table).reshape(-1)

    return hash_map_for_faces(
        new_active_cell_truth_table,
        cell_faces,
        face_hash_map,
        inner_faces,
        all_faces,
        new_cell_indices,
    )


def get_active_face_truth_table(problem, external_faces, laser_center_z, top_tolerance=1.0e-5):
    points = np.asarray(problem.fe.points)

    cells = np.asarray(problem.fe.cells)

    face_indices = np.asarray(problem.fe.face_inds)

    cell_points = np.take(points, cells, axis=0)

    cell_face_points = np.take(cell_points, face_indices, axis=1)

    all_boundary_faces = np.asarray(problem.boundary_inds_list[0])

    boundary_face_dictionary = {}

    for index, face in enumerate(all_boundary_faces):
        face_key = tuple(map(int, np.asarray(face).reshape(-1)))

        boundary_face_dictionary[face_key] = index

    active_external_faces = np.zeros(len(all_boundary_faces), dtype=bool)

    active_top_faces = np.zeros(len(all_boundary_faces), dtype=bool)

    for face in external_faces:
        face_tuple = tuple(map(int, np.asarray(face).reshape(-1)))

        if face_tuple not in boundary_face_dictionary:
            raise ValueError("External face is not present in " "the complete boundary face list.")

        boundary_index = boundary_face_dictionary[face_tuple]

        active_external_faces[boundary_index] = True

        cell_id, local_face_id = face_tuple

        face_points = cell_face_points[cell_id, local_face_id]

        if np.all(np.isclose(face_points[:, 2], float(laser_center_z), atol=top_tolerance)):
            active_top_faces[boundary_index] = True

    return np.stack((active_external_faces, active_top_faces), axis=1)


def get_quiet_point_mask(problem, active_cell_truth_table):
    cells = np.asarray(problem.fe.cells)

    points = np.asarray(problem.fe.points)

    active_mask = np.asarray(active_cell_truth_table, dtype=bool)

    quiet_point_mask = np.ones(len(points), dtype=bool)

    # Use a fixed-length quiet-node mask to avoid repeated JIT compilation caused by changing quiet-index array lengths.
    quiet_point_mask[cells[active_mask].reshape(-1)] = False

    return quiet_point_mask
