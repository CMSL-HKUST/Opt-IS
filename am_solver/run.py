import copy
import os
import sys
from pathlib import Path

import yaml


CASE_NAME = "thin_wall"


def load_raw_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"配置文件为空：{config_path}")

    return config


def load_shared_solver_config(project_root):
    path = project_root.parent / "solver_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"公共求解器配置不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        shared_config = yaml.safe_load(file)

    solver_config = shared_config.get("am") if isinstance(shared_config, dict) else None
    if not isinstance(solver_config, dict):
        raise ValueError(f"公共求解器配置缺少 am 段：{path}")
    for key in ("thermal", "mechanics"):
        if not isinstance(solver_config.get(key), dict):
            raise ValueError(f"公共求解器配置缺少 am.{key}：{path}")

    return copy.deepcopy(solver_config)


def configure_runtime(config):
    runtime = config.get("runtime", {})

    os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime.get("cuda_visible_devices", "0"))

    preallocate = runtime.get("xla_preallocate", False)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"

    memory_fraction = runtime.get("xla_memory_fraction")

    if memory_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)


def resolve_config_path(project_root, argument):
    if argument is None:
        return project_root / "cases" / CASE_NAME / f"{CASE_NAME}.yaml"

    argument_path = Path(argument).expanduser()

    if argument_path.exists():
        if argument_path.is_dir():
            candidates = [
                argument_path / f"{argument_path.name}.yaml",
                argument_path / "config.yaml",
            ]

            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()

            raise FileNotFoundError(f"案例目录中找不到配置文件：{argument_path}")

        return argument_path.resolve()

    case_dir = project_root / "cases" / argument
    candidates = [
        case_dir / f"{argument}.yaml",
        case_dir / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return argument_path.resolve()


def main():
    project_root = Path(__file__).resolve().parent

    config_path = resolve_config_path(
        project_root,
        sys.argv[1] if len(sys.argv) > 1 else None,
    )

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    raw_config = load_raw_config(config_path)
    raw_config["_config_dir"] = str(config_path.parent)
    # The thermal-only validation case intentionally keeps its case-local
    # solver choice so it can isolate the AM thermal formulation. Production
    # AM cases continue to use the repository-level shared solver policy.
    if raw_config.get("mode") == "thermal_only":
        print(f"Using case thermal solver: {raw_config['solver']['thermal']}")
    else:
        raw_config["solver"] = load_shared_solver_config(project_root)

    print(f"Using config: {config_path}")

    # Execute before importing JAX-related modules.
    configure_runtime(raw_config)

    from thermo_mech.simulation import run_simulation

    run_simulation(config=raw_config, project_root=project_root)


if __name__ == "__main__":
    main()
