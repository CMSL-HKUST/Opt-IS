import glob
import os
from pathlib import Path


def resolve_path(project_root, path_value):
    path = Path(path_value)

    if not path.is_absolute():
        path = Path(project_root) / path

    return path.resolve()


def get_paths_base(config, project_root):
    relative_to = config.get(
        "paths",
        {},
    ).get(
        "relative_to",
        "project",
    )

    if relative_to == "config":
        config_dir = config.get("_config_dir")

        if config_dir is None:
            raise KeyError(
                "paths.relative_to: config 需要由 run.py 提供 _config_dir"
            )

        return Path(config_dir).resolve()

    if relative_to == "project":
        return Path(project_root).resolve()

    raise ValueError(
        f"不支持的 paths.relative_to：{relative_to}"
    )


def validate_config(config):
    required_sections = [
        "paths",
        "mesh",
        "geometry",
        "activation",
        "toolpath",
        "material",
        "inherent_strain",
        "solver",
        "output",
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(
                f"配置文件缺少配置段：{section}"
            )

    number_of_layers = int(
        config["toolpath"]["number_of_layers"]
    )

    if number_of_layers < 1:
        raise ValueError(
            "toolpath.number_of_layers 必须大于零"
        )

    number_of_increments = int(
        config["inherent_strain"][
            "increments_per_layer"
        ]
    )

    if number_of_increments < 1:
        raise ValueError(
            "increments_per_layer 必须大于或等于 1"
        )

    poisson_ratio = float(
        config["material"]["poisson_ratio"]
    )

    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError(
            "泊松比必须位于 (-1, 0.5) 内"
        )


def prepare_output_directories(
    config,
    project_root,
):
    paths_base = get_paths_base(
        config,
        project_root,
    )

    output_directory = resolve_path(
        paths_base,
        config["paths"]["output_directory"],
    )

    vtk_directory = resolve_path(
        paths_base,
        config["paths"]["vtk_directory"],
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vtk_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    clear_vtk = bool(
        config.get(
            "output",
            {},
        ).get(
            "clear_vtk_directory",
            False,
        )
    )

    if clear_vtk:
        for file_path in glob.glob(
            str(vtk_directory / "*")
        ):
            if os.path.isfile(file_path):
                os.remove(file_path)

    return {
        "paths_base": paths_base,
        "output_directory": output_directory,
        "vtk_directory": vtk_directory,
    }
