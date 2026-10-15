# OPT-IS

Code for additive manufacturing (AM) thermomechanical simulation, inherent strain optimization, and macroscale forward analysis.

Personal website: [Residual Stress Hub](https://dongdaoyang.github.io/).

## Directory Structure

- `am_solver/`: AM thermomechanical solver.
- `IS_solver/`: Inherent strain forward analysis and optimization modules.
- `opt_solver/`: Workflow entry point connecting AM simulation, inherent strain optimization, and macroscale forward analysis.
- `opt_solver/cases/cantilever/`: Cantilever case configuration and input meshes.
- `solver_config.yaml`: Shared solver configuration for all stages.

See [opt_solver/README.md](opt_solver/README.md) for workflow instructions. With the current directory structure, replace `dc` with `cantilever` in the example case commands. Refer to `solver_config.yaml` in the repository root for the actual solver parameters.

## Version Control

This repository tracks source code, case configurations, and input meshes. Python caches, local virtual environments, environment variable files, and `opt_output/` directories within the repository are excluded by `.gitignore`.

After making and checking your changes, run the following commands from the repository root:

```bash
git status
git add .
git commit -m "Describe the changes made in this commit"
git push
```

Before continuing work on another computer, run `git pull --ff-only` to fetch and apply the latest commits from the remote repository.
