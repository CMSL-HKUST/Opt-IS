import os
from pathlib import Path

import numpy as np
import pyvista as pv


STRAIN_COMPONENT_NAMES = {
    "xx": "eps_e_xx",
    "yy": "eps_e_yy",
    "zz": "eps_e_zz",
    "eps_e_xx": "eps_e_xx",
    "eps_e_yy": "eps_e_yy",
    "eps_e_zz": "eps_e_zz",
}

COMPONENT_INDICES = {
    "x": 0,
    "xx": 0,
    "0": 0,
    0: 0,
    "y": 1,
    "yy": 1,
    "1": 1,
    1: 1,
    "z": 2,
    "zz": 2,
    "2": 2,
    2: 2,
}

COMPONENT_LABELS = {
    0: "x",
    1: "y",
    2: "z",
}


def _resolve_path(project_root, path_value):
    path = Path(path_value)

    if not path.is_absolute():
        path = Path(project_root) / path

    return path.resolve()


def _read_point(config, key):
    value = config[key]

    if len(value) != 3:
        raise ValueError(
            f"postprocess.line.{key} 必须包含 3 个坐标值"
        )

    return [
        float(value[0]),
        float(value[1]),
        float(value[2]),
    ]


def _as_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _normalise_variable_specs(config):
    specs = []

    for component in config.get(
        "strain_components",
        [],
    ):
        component_name = str(component).lower()

        if component_name not in STRAIN_COMPONENT_NAMES:
            raise ValueError(
                "不支持的应变分量："
                f"{component}。可选："
                f"{list(STRAIN_COMPONENT_NAMES.keys())}"
            )

        variable_name = STRAIN_COMPONENT_NAMES[
            component_name
        ]
        specs.append({
            "name": variable_name,
            "location": "cell",
            "components": [],
            "scale": 1.0,
            "label": variable_name,
        })

    for variable in config.get(
        "variables",
        [],
    ):
        if isinstance(variable, str):
            specs.append({
                "name": variable,
                "location": "auto",
                "components": [],
                "scale": 1.0,
                "label": variable,
            })
            continue

        variable_name = str(variable["name"])
        specs.append({
            "name": variable_name,
            "location": str(
                variable.get(
                    "location",
                    "auto",
                )
            ).lower(),
            "components": _as_list(
                variable.get(
                    "components",
                    variable.get(
                        "component",
                    ),
                )
            ),
            "scale": float(
                variable.get(
                    "scale",
                    1.0,
                )
            ),
            "label": str(
                variable.get(
                    "label",
                    variable_name,
                )
            ),
        })

    if not specs:
        raise ValueError(
            "postprocess 至少需要配置 "
            "variables 或 strain_components"
        )

    return specs


def _resolve_vtk_path(
    config,
    paths_base,
    vtk_paths,
    default_source,
):
    if config.get("vtk_path"):
        return _resolve_path(
            paths_base,
            config["vtk_path"],
        )

    source = config.get(
        "source",
        "auto",
    )

    if source == "auto":
        source = default_source

    if source not in vtk_paths:
        raise ValueError(
            f"不支持的 postprocess.source：{source}。"
            f"可选：{list(vtk_paths.keys())} 或 auto"
        )

    return vtk_paths[source]


def _resolve_location(mesh, variable_name, location):
    if location == "auto":
        if variable_name in mesh.point_data:
            return "point"

        if variable_name in mesh.cell_data:
            return "cell"

    if location == "point":
        if variable_name in mesh.point_data:
            return "point"

    if location == "cell":
        if variable_name in mesh.cell_data:
            return "cell"

    raise ValueError(
        f"VTU 中找不到 {location} 变量：{variable_name}。"
        f"可用 point_data：{list(mesh.point_data.keys())}；"
        f"cell_data：{list(mesh.cell_data.keys())}"
    )


def _component_index(component):
    if component not in COMPONENT_INDICES:
        raise ValueError(
            f"不支持的分量：{component}。"
            "可选 x/y/z、0/1/2 或 magnitude"
        )

    return COMPONENT_INDICES[component]


def _column_name(spec, component=None):
    label = spec["label"]

    if component is None:
        return label

    if str(component).lower() in {
        "magnitude",
        "mag",
        "norm",
    }:
        return f"{label}_magnitude"

    return (
        f"{label}_"
        f"{COMPONENT_LABELS[_component_index(component)]}"
    )


def _extract_columns(sampled_data, spec):
    variable_name = spec["name"]

    if variable_name not in sampled_data.point_data:
        raise ValueError(
            f"采样结果中找不到变量：{variable_name}"
        )

    values = np.asarray(
        sampled_data.point_data[variable_name]
    )

    if values.ndim == 1:
        if spec["components"]:
            raise ValueError(
                f"变量 {variable_name} 是标量，不能指定分量"
            )

        return [(
            _column_name(spec),
            values * spec["scale"],
        )]

    components = spec["components"]

    if not components:
        components = list(
            range(values.shape[1])
        )

    columns = []

    for component in components:
        component_name = str(component).lower()

        if component_name in {
            "magnitude",
            "mag",
            "norm",
        }:
            data = np.linalg.norm(
                values,
                axis=1,
            )
        else:
            index = _component_index(component)

            if index >= values.shape[1]:
                raise ValueError(
                    f"变量 {variable_name} 没有 "
                    f"{component} 分量，当前形状为 {values.shape}"
                )

            data = values[:, index]

        columns.append((
            _column_name(spec, component),
            data * spec["scale"],
        ))

    return columns


def _write_plot(
    output_directory,
    output_stem,
    distances,
    column_name,
    values,
):
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(output_directory / ".matplotlib"),
    )

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_path = (
        output_directory
        / f"{output_stem}_{column_name}.png"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        distances,
        values,
        "o-",
        linewidth=2,
        markersize=4,
        alpha=0.8,
    )
    plt.xlabel("Distance from Start Point (m)")
    plt.ylabel(column_name)
    plt.title(
        f"{column_name} Distribution Along Line"
    )
    plt.grid(
        True,
        linestyle="--",
        alpha=0.6,
    )
    plt.tight_layout()
    plt.savefig(
        plot_path,
        dpi=300,
    )
    plt.close()

    return plot_path


def _run_single_line_postprocess(
    job_config,
    paths_base,
    vtk_paths,
    default_source,
):
    variable_specs = _normalise_variable_specs(
        job_config
    )

    line_config = job_config["line"]
    line_start = _read_point(
        line_config,
        "start_m",
    )
    line_end = _read_point(
        line_config,
        "end_m",
    )
    number_of_points = int(
        line_config.get(
            "num_points",
            100,
        )
    )

    if number_of_points < 2:
        raise ValueError(
            "postprocess.line.num_points 必须大于等于 2"
        )

    vtk_path = _resolve_vtk_path(
        job_config,
        paths_base,
        vtk_paths,
        default_source,
    )

    if not vtk_path.exists():
        raise FileNotFoundError(
            f"找不到后处理 VTU 文件：{vtk_path}"
        )

    output_directory = _resolve_path(
        paths_base,
        job_config.get(
            "output_directory",
            "output/postprocess",
        ),
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_stem = job_config.get(
        "output_stem",
        "line_sample",
    )

    mesh = pv.read(str(vtk_path))

    resolved_specs = []

    for spec in variable_specs:
        resolved_spec = dict(spec)
        resolved_spec["location"] = _resolve_location(
            mesh,
            spec["name"],
            spec["location"],
        )
        resolved_specs.append(resolved_spec)

    needs_cell_to_point = any(
        spec["location"] == "cell"
        for spec in resolved_specs
    )

    mesh_for_sampling = (
        mesh.cell_data_to_point_data()
        if needs_cell_to_point
        else mesh
    )

    print("\n" + "=" * 60)
    print("Postprocess line sampling:")
    if job_config.get("name"):
        print(f"  Job: {job_config['name']}")
    print(f"  Source VTU: {vtk_path}")
    print(
        "  Variables: "
        + ", ".join(
            f"{spec['name']}({spec['location']})"
            for spec in resolved_specs
        )
    )
    print(
        f"  Line: {line_start} -> {line_end}, "
        f"points={number_of_points}"
    )
    print("=" * 60)

    sampled_data = (
        mesh_for_sampling.sample_over_line(
            line_start,
            line_end,
            resolution=number_of_points - 1,
        )
    )

    distances = np.asarray(
        sampled_data["Distance"]
    )
    points = np.asarray(
        sampled_data.points
    )

    data_columns = [
        distances,
        points[:, 0],
        points[:, 1],
        points[:, 2],
    ]

    header_names = [
        "Distance(m)",
        "X(m)",
        "Y(m)",
        "Z(m)",
    ]

    plotted_columns = []

    for spec in resolved_specs:
        for column_name, values in _extract_columns(
            sampled_data,
            spec,
        ):
            data_columns.append(values)
            header_names.append(column_name)
            plotted_columns.append((
                column_name,
                values,
            ))

    csv_path = (
        output_directory
        / f"{output_stem}.csv"
    )

    np.savetxt(
        csv_path,
        np.column_stack(data_columns),
        delimiter=",",
        header=",".join(header_names),
        comments="",
        fmt="%.8e",
    )

    print(f"  Saved CSV: {csv_path}")

    plot_paths = []

    if job_config.get(
        "save_plot",
        True,
    ):
        for column_name, values in plotted_columns:
            plot_path = _write_plot(
                output_directory,
                output_stem,
                distances,
                column_name,
                values,
            )
            plot_paths.append(plot_path)
            print(f"  Saved plot: {plot_path}")

    return {
        "csv_path": csv_path,
        "plot_paths": plot_paths,
    }


def _merge_job_config(config, job_config):
    merged = {
        key: value
        for key, value in config.items()
        if key != "jobs"
    }

    merged.update(job_config)

    if "output_stem" not in merged:
        merged["output_stem"] = str(
            job_config.get(
                "name",
                "line_sample",
            )
        )

    return merged


def run_line_postprocess(
    config,
    paths_base,
    vtk_paths,
    default_source,
):
    if not config.get(
        "enabled",
        False,
    ):
        return None

    jobs = config.get("jobs")

    if jobs:
        results = []

        for job_config in jobs:
            results.append(
                _run_single_line_postprocess(
                    _merge_job_config(
                        config,
                        job_config,
                    ),
                    paths_base,
                    vtk_paths,
                    default_source,
                )
            )

        return results

    return _run_single_line_postprocess(
        config,
        paths_base,
        vtk_paths,
        default_source,
    )
