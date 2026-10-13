import copy
import os
import sys
from pathlib import Path

import yaml


CASE_NAME = "dc"
#ASE_NAME = "bg"

def load_yaml(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"配置文件为空：{config_path}")

    return config


def load_shared_solver_config(project_root, section):
    path = project_root.parent / "solver_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"公共求解器配置不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        shared_config = yaml.safe_load(file)

    solver_config = shared_config.get(section) if isinstance(shared_config, dict) else None
    if not isinstance(solver_config, dict):
        raise ValueError(f"公共求解器配置缺少 {section} 段：{path}")

    required_keys = {
        "forward": ("forward", "cutting"),
    }.get(section, ())
    for key in required_keys:
        if not isinstance(solver_config.get(key), dict):
            raise ValueError(f"公共求解器配置缺少 {section}.{key}：{path}")

    return copy.deepcopy(solver_config)


def configure_runtime(config):
    runtime_config = config.get("runtime", {})

    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        runtime_config.get("cuda_visible_devices", "0")
    )

    if not runtime_config.get("xla_preallocate", True):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    memory_fraction = runtime_config.get(
        "xla_memory_fraction"
    )

    if memory_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            memory_fraction
        )


def resolve_config_path(project_root, argument):
    if argument is None:
        return (
            project_root
            / "cases"
            / CASE_NAME
            / f"{CASE_NAME}.yaml"
        )

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

            raise FileNotFoundError(
                f"案例目录中找不到配置文件：{argument_path}"
            )

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
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}"
        )

    config = load_yaml(config_path)
    config["_config_dir"] = str(config_path.parent)
    # Keep forward and cutting settings independent of the selected case.
    config["solver"] = load_shared_solver_config(project_root, "forward")

    # Execute before importing jax.
    configure_runtime(config)

    import jax

    jax.config.update(
        "jax_enable_x64",
        bool(
            config.get(
                "runtime",
                {},
            ).get("enable_x64", True)
        ),
    )

    # Place after the GPU and JAX precision settings.
    from inherent_solver.simulation import run_simulation

    run_simulation(
        config=config,
        project_root=project_root,
    )

if __name__ == "__main__":
    main()
