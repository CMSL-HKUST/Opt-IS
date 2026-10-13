from pathlib import Path

import numpy as np


def generate_toolpath(config):
    number_of_layers = int(
        config["number_of_layers"]
    )

    layer_duration = float(
        config["layer_duration_s"]
    )

    layer_height = float(
        config["layer_height_m"]
    )

    scan_config = config["scan"]

    scan_length = float(
        scan_config["length_m"]
    )

    scan_width = float(
        scan_config.get(
            "width_m",
            0.0,
        )
    )

    y_position = float(
        scan_config.get(
            "y_m",
            0.0,
        )
    )

    rows = []
    current_time = 0.0

    rows.append([
        current_time,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
    ])

    for layer_index in range(
        1,
        number_of_layers + 1,
    ):
        current_time += layer_duration

        z_position = (
            layer_index
            * layer_height
        )

        rows.append([
            current_time,
            -scan_length / 2.0,
            scan_length / 2.0,
            y_position,
            y_position,
            z_position,
            1,
        ])

    return np.asarray(
        rows,
        dtype=float,
    )


def load_toolpath(config, paths_base=None):
    mode = config.get(
        "mode",
        "generated",
    )

    if mode == "generated":
        return generate_toolpath(config)

    if mode == "file":
        file_path = Path(config["file"])

        if (
            paths_base is not None
            and not file_path.is_absolute()
        ):
            file_path = Path(paths_base) / file_path

        delimiter = config.get(
            "delimiter",
            ",",
        )

        skip_rows = int(
            config.get(
                "skip_rows",
                0,
            )
        )

        toolpath = np.loadtxt(
            file_path,
            delimiter=delimiter,
            skiprows=skip_rows,
            dtype=float,
        )

        if toolpath.ndim == 1:
            toolpath = toolpath[None, :]

        if toolpath.shape[1] != 7:
            raise ValueError(
                "打印路径文件必须包含 7 列："
                "time, x1, x2, y1, y2, z, flag"
            )

        return toolpath

    raise ValueError(
        f"不支持的 toolpath.mode：{mode}"
    )


def create_layerwise_inherent_strain(
    inherent_strain_config,
    number_of_layers,
):
    default_strain = inherent_strain_config[
        "default"
    ]

    strain_vector = np.array([
        float(default_strain["xx"]),
        float(default_strain["yy"]),
        float(default_strain["zz"]),
    ])

    return np.tile(
        strain_vector[None, :],
        (number_of_layers, 1),
    )
