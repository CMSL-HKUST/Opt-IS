import os
import sys
from pathlib import Path

import yaml


def load_raw_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"配置文件为空：{config_path}")

    return config


def configure_runtime(config):
    runtime = config.get("runtime", {})

    os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime.get("cuda_visible_devices", "0"))

    preallocate = runtime.get("xla_preallocate", False)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"

    memory_fraction = runtime.get("xla_memory_fraction")

    if memory_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)


def main():
    project_root = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).resolve()
    else:
        config_path = project_root / "configs" / "track1.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    raw_config = load_raw_config(config_path)

    # Execute before importing JAX-related modules.
    configure_runtime(raw_config)

    from thermo_mech.simulation import run_simulation

    run_simulation(config=raw_config, project_root=project_root)


if __name__ == "__main__":
    main()