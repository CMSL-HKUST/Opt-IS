import numpy as onp
import jax.numpy as np

from jax_fem.utils import save_sol


def save_step(
    problem_temperature,
    temperature,
    displacement_solution,
    plastic_information,
    mechanics_params,
    active_cell_truth_table,
    current_time,
    vtk_path,
    additional_point_infos=None,
):
    point_infos = [("u", displacement_solution[0] * 1.0e3)]

    if additional_point_infos:
        point_infos.extend(additional_point_infos)

    # The legacy fields retain their exact names/values. Add only the three
    # normal components accumulated from the J2 flow, when enabled.
    flow_cell_infos = []
    if len(plastic_information) == 20:
        flow_cell_infos = [
            ("strain_pl_flow_" + axis, np.mean(plastic_information[17 + index], axis=1).flatten())
            for index, axis in enumerate(("xx", "yy", "zz"))
        ]

    save_sol(
        problem_temperature.fes[0],
        temperature,
        str(vtk_path),
        point_infos=point_infos,
        cell_infos=[
            ("time", onp.full(len(active_cell_truth_table), current_time)),
            ("f_plus", np.mean(plastic_information[0] / 1.0e6, axis=1).flatten()),
            ("stress_xx", np.mean(plastic_information[1] / 1.0e6, axis=1).flatten()),
            ("stress_yy", np.mean(plastic_information[2] / 1.0e6, axis=1).flatten()),
            ("stress_zz", np.mean(plastic_information[3] / 1.0e6, axis=1).flatten()),
            ("stress_mises", np.mean(plastic_information[4] / 1.0e6, axis=1).flatten()),
            ("strain_tot_xx", np.mean(plastic_information[5], axis=1).flatten()),
            ("strain_tot_yy", np.mean(plastic_information[6], axis=1).flatten()),
            ("strain_tot_zz", np.mean(plastic_information[7], axis=1).flatten()),
            ("strain_el_xx", np.mean(plastic_information[8], axis=1).flatten()),
            ("strain_el_yy", np.mean(plastic_information[9], axis=1).flatten()),
            ("strain_el_zz", np.mean(plastic_information[10], axis=1).flatten()),
            ("strain_pl_xx", np.mean(plastic_information[11], axis=1).flatten()),
            ("strain_pl_yy", np.mean(plastic_information[12], axis=1).flatten()),
            ("strain_pl_zz", np.mean(plastic_information[13], axis=1).flatten()),
            ("strain_mech_xx", np.mean(plastic_information[14], axis=1).flatten()),
            ("strain_mech_yy", np.mean(plastic_information[15], axis=1).flatten()),
            ("strain_mech_zz", np.mean(plastic_information[16], axis=1).flatten()),
            ("phase", np.max(mechanics_params[5][:, :, 0], axis=1).flatten()),
            ("p_strain", np.mean(mechanics_params[2], axis=1).flatten()),
            ("active", onp.asarray(active_cell_truth_table, dtype=onp.int32).flatten()),
        ] + flow_cell_infos,
    )
