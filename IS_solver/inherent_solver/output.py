from pathlib import Path

import jax.numpy as np
import numpy as onp

from jax_fem.utils import save_sol


def calculate_cell_results(
    stress,
    plastic_strain,
    elastic_strain,
):
    return {
        "stress_xx": (
            onp.asarray(
                np.mean(
                    stress[:, :, 0, 0],
                    axis=1,
                )
            )
            / 1.0e6
        ),
        "stress_yy": (
            onp.asarray(
                np.mean(
                    stress[:, :, 1, 1],
                    axis=1,
                )
            )
            / 1.0e6
        ),
        "stress_zz": (
            onp.asarray(
                np.mean(
                    stress[:, :, 2, 2],
                    axis=1,
                )
            )
            / 1.0e6
        ),
        "p_strain": onp.asarray(
            np.mean(
                plastic_strain,
                axis=1,
            )
        ),
        "eps_e_xx": onp.asarray(
            np.mean(
                elastic_strain[:, :, 0, 0],
                axis=1,
            )
        ),
        "eps_e_yy": onp.asarray(
            np.mean(
                elastic_strain[:, :, 1, 1],
                axis=1,
            )
        ),
        "eps_e_zz": onp.asarray(
            np.mean(
                elastic_strain[:, :, 2, 2],
                axis=1,
            )
        ),
    }


def save_forward_result(
    problem,
    solution,
    stress,
    plastic_strain,
    elastic_strain,
    active_mask,
    vtk_path,
    displacement_scale=1000.0,
):
    vtk_path = Path(vtk_path)

    vtk_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = calculate_cell_results(
        stress,
        plastic_strain,
        elastic_strain,
    )

    save_sol(
        problem.fe,
        solution,
        str(vtk_path),
        point_infos=[
            (
                "u",
                solution
                * displacement_scale,
            ),
        ],
        cell_infos=[
            (
                "stress_xx",
                results["stress_xx"],
            ),
            (
                "stress_yy",
                results["stress_yy"],
            ),
            (
                "stress_zz",
                results["stress_zz"],
            ),
            (
                "p_strain",
                results["p_strain"],
            ),
            (
                "eps_e_xx",
                results["eps_e_xx"],
            ),
            (
                "eps_e_yy",
                results["eps_e_yy"],
            ),
            (
                "eps_e_zz",
                results["eps_e_zz"],
            ),
            (
                "active_mask",
                onp.asarray(
                    active_mask,
                    dtype=onp.int32,
                ),
            ),
        ],
    )


def save_cutting_result(
    problem,
    solution,
    stress,
    plastic_strain,
    elastic_strain,
    vtk_path,
    active_mask=None,
    displacement_scale=1000.0,
):
    vtk_path = Path(vtk_path)

    vtk_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = calculate_cell_results(
        stress,
        plastic_strain,
        elastic_strain,
    )

    cell_infos = [
        (
            "stress_xx",
            results["stress_xx"],
        ),
        (
            "stress_yy",
            results["stress_yy"],
        ),
        (
            "stress_zz",
            results["stress_zz"],
        ),
        (
            "p_strain",
            results["p_strain"],
        ),
        (
            "eps_e_xx",
            results["eps_e_xx"],
        ),
        (
            "eps_e_yy",
            results["eps_e_yy"],
        ),
        (
            "eps_e_zz",
            results["eps_e_zz"],
        ),
    ]

    if active_mask is not None:
        cell_infos.append(
            (
                "active_mask",
                onp.asarray(
                    active_mask,
                    dtype=onp.int32,
                ),
            )
        )

    save_sol(
        problem.fe,
        solution,
        str(vtk_path),
        point_infos=[
            (
                "u",
                solution
                * displacement_scale,
            ),
        ],
        cell_infos=cell_infos,
    )
