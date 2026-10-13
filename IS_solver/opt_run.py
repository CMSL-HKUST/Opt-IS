import logging
import sys
from pathlib import Path

from run import configure_runtime, load_shared_solver_config, load_yaml


def resolve_config_path(project_root, argument):
    if argument is None:
        return (
            project_root
            / "opt_case"
            / "5_300"
            / "v3.yaml"
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

    case_dir = project_root / "opt_case" / argument
    candidates = [
        case_dir / f"{argument}.yaml",
        case_dir / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return argument_path.resolve()


def main():
    # Also keep direct invocations responsive when stdout is a log pipe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    project_root = Path(__file__).resolve().parent

    config_path = resolve_config_path(
        project_root,
        sys.argv[1] if len(sys.argv) > 1 else None,
    )

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    config = load_yaml(config_path)
    config["_config_dir"] = str(config_path.parent)
    shared_solver_path = project_root.parent / "solver_config.yaml"
    optimization_solver = load_shared_solver_config(
        project_root,
        "optimization",
    )
    config.setdefault("optimization", {})["solver"] = optimization_solver

    print(
        "Using shared optimization solver: "
        f"config={shared_solver_path}, "
        f"backend={optimization_solver.get('backend', 'spsolve')}"
    )

    configure_runtime(config)

    import jax

    jax.config.update(
        "jax_enable_x64",
        bool(
            config.get("runtime", {}).get(
                "enable_x64",
                True,
            )
        ),
    )

    from optimization.inherent_strain_optimization import run_optimization

    # JAX-FEM sets its logger to DEBUG during import.  INFO keeps Newton
    # residuals/timings visible during expensive objective evaluations while
    # avoiding per-kernel DEBUG chatter.  This changes reporting only.
    log_level = str(config.get("runtime", {}).get("optimization_log_level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Invalid runtime.optimization_log_level: {log_level}")
    logging.getLogger("jax_fem").setLevel(getattr(logging, log_level))

    run_optimization(
        config=config,
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
