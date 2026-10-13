from pathlib import Path
import glob
import os


def resolve_path(project_root, path_value):
    path = Path(path_value)

    if not path.is_absolute():
        path = Path(project_root) / path

    return path.resolve()


def get_paths_base(config, project_root):
    relative_to = config.get("paths", {}).get("relative_to", "project")

    if relative_to == "config":
        config_dir = config.get("_config_dir")

        if config_dir is None:
            raise KeyError("paths.relative_to: config 需要由 run.py 提供 _config_dir")

        return Path(config_dir).resolve()

    if relative_to == "project":
        return Path(project_root).resolve()

    raise ValueError(f"不支持的 paths.relative_to：{relative_to}")


def prepare_output_directories(config, project_root):
    path_config = config["paths"]
    paths_base = get_paths_base(config, project_root)

    output_dir = resolve_path(paths_base, path_config["output_directory"])

    vtk_dir = resolve_path(paths_base, path_config["vtk_directory"])

    mesh_output_path = resolve_path(paths_base, path_config["mesh_output"])

    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_dir.mkdir(parents=True, exist_ok=True)
    mesh_output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.get("output", {}).get("clear_vtk_directory", False):
        for file_path in glob.glob(str(vtk_dir / "*")):
            if os.path.isfile(file_path):
                os.remove(file_path)

    return{"output_dir": output_dir, "vtk_dir": vtk_dir, "mesh_output_path": mesh_output_path}


def validate_config(config):
    required_sections = [
        "paths",
        "mesh",
        "geometry",
        "thermal",
        "heat_source",
        "solver",
        "materials",
    ]

    if config.get("mode") != "thermal_only":
        required_sections.extend(
            ["mechanics", "activation", "toolpath", "time_stepping"]
        )
    elif "thermal_benchmark" not in config:
        raise KeyError("thermal_only case 缺少 thermal_benchmark 配置")

    for section in required_sections:
        if section not in config:
            raise KeyError(f"YAML 缺少配置段：{section}")

    if "upper_region" not in config["materials"]:
        raise KeyError("materials 中缺少 upper_region")

    if "lower_region" not in config["materials"]:
        raise KeyError("materials 中缺少 lower_region")
