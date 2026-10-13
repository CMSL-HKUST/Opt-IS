"""Build solver options understood by the upstream JAX-FEM package."""


def native_linear_solver_options(solver_config=None):
    """Return an upstream JAX-FEM linear-solver option mapping.

    The portable default is SciPy's sparse direct solver.  Optional upstream
    JAX-FEM backends remain available through their native option names, but
    the project never imports a patched JAX-FEM module.
    """

    config = solver_config or {}
    backend = str(config.get("backend", "spsolve")).strip().lower()

    if backend in {"spsolve", "scipy", "scipy_spsolve", "spsolve_solver"}:
        return {"spsolve_solver": {}}

    if backend in {"jax", "jax_solver"}:
        return {
            "jax_solver": {
                "precond": bool(config.get("precond", True)),
            }
        }

    if backend in {"petsc", "petsc_solver"}:
        options = {}
        for key in ("ksp_type", "pc_type", "factor_solver_type"):
            if key in config:
                options[key] = config[key]
        return {"petsc_solver": options}

    if backend in {"amgx", "amgx_solver"}:
        options = {}
        if config.get("cfg_path") is not None:
            options["cfg_path"] = config["cfg_path"]
        return {"amgx_solver": options}

    if backend in {"mpi_hypre", "petsc_cuda"}:
        raise ValueError(
            f"backend={backend!r} is not part of upstream JAX-FEM; "
            "use backend: spsolve, jax, petsc, or amgx"
        )

    raise ValueError(
        f"Unsupported solver backend {backend!r}; "
        "use backend: spsolve, jax, petsc, or amgx"
    )
