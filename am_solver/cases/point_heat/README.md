# Point-heat thermal example

Run from the repository root:

```bash
python am_solver/run.py point_heat
```

This thermal-only case uses a structured HEX8 box, a stationary Gaussian surface
heat source, and analytical half-space temperatures on the side and bottom
boundaries. The default mesh has 80 × 80 × 60 cells, and the configured final time
is 0.08 ms. Reduce `mesh.cells_xyz` in `point_heat.yaml` for a smaller calculation.

The AM entry point preserves this case's thermal solver settings. Output is
written beside the repository under `opt_output/am_solver/point_heat/`.
