"""Build solver options understood by the upstream JAX-FEM package.

The project used to import a local JAX-FEM extension to translate the
case-level ``backend`` setting. Upstream JAX-FEM already provides the linear
solver registry, so keep the translation at the project boundary and pass
only upstream option names to ``jax_fem.solver``.
"""


def native_linear_solver_options(solver_config=None):
    """Return an upstream JAX-FEM linear-solver option mapping.

    ``spsolve`` is the portable default because it is provided by SciPy, which
    is already required by JAX-FEM.  The other mappings are retained for
    users who have the corresponding optional upstream dependency installed.
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
