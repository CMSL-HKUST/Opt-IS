# AM -> 固有应变优化 -> 宏观正向求解

`opt_solver` 只调用 `am_solver` 和 `IS_solver` 中的求解代码。案例特有的配置、
材料参数、工具路径和网格放在 `cases/<case>/` 内；所有 case 共用的求解器策略
放在项目根目录的 `solver_config.yaml`，不读取两个求解器原有的案例目录。

一个案例包含：

- `config.yaml`：流程、网格文件名、求解器位置和单位换算。
- `am.yaml`：AM 热-力参数及完整的温度相关材料参数。
- `optimization.yaml`：小网格固有应变反演参数。
- `forward.yaml`：大网格正向、切割和后处理参数。
- `mesh/small.inp`：AM 和优化共用的小网格。
- `mesh/large.inp`：宏观正向求解使用的大网格。

所有 case 共用的线性求解器后端和选项统一放在项目根目录的
`solver_config.yaml`，不再写入 `cases/<case>/` 下的任何 YAML。编排器会把这份
公共配置注入每个阶段生成的配置；独立运行 `am_solver/run.py` 或
`IS_solver/run.py` 时也会读取同一份配置。`optimization` 下的选项会应用到优化
过程中的正问题和伴随求解。

当前公共策略是：AM 热-力扫描、固有应变优化（包括伴随求解）和宏观正向阶段均
使用 PETSc `tfqmr + lu`；切割阶段保留 SciPy 稀疏直接求解器（`backend: spsolve`）。
AM thermal 的温升初值预测也在公共配置中统一开启。需要 `petsc4py` 和带 LU
因子分解支持的 PETSc。

以后切换求解器时只修改根目录的 `solver_config.yaml`；case 内的阶段 YAML 不应再
添加或覆盖 `solver`，否则 `opt_solver/run.py` 会以公共配置为准。

```yaml
am:
  thermal:
    backend: petsc
    ksp_type: tfqmr
    pc_type: lu
    initial_guess_predictor:
      enabled: true
      reference_heat_capacity_j_kgk: 500.0
      scale: 1.0
  mechanics:
    backend: petsc
    ksp_type: tfqmr
    pc_type: lu
    line_search: true
    tolerance: 1.0e-6
    relative_tolerance: 1.0e-8
optimization:
  backend: petsc
  ksp_type: tfqmr
  pc_type: lu
  line_search: true
  tolerance: 1.0e-6
  relative_tolerance: 1.0e-8
forward:
  forward:
    backend: petsc
    ksp_type: tfqmr
    pc_type: lu
    line_search: true
    tolerance: 1.0e-6
    relative_tolerance: 1.0e-8
  cutting:
    backend: spsolve
```

在 `config.yaml` 开头设置是否运行宏观模型：

```yaml
run_macro_model: false
```

设为 `false` 时，流程只运行 AM 和固有应变优化，不校验或生成大网格正向
求解配置，此时不能使用 `--from-stage forward`。未设置该选项时默认值为
`true`。

## 运行

先进入同时安装了 `PyYAML`、JAX/JAX-FEM、MeshIO 和 SciPy 的求解环境；本机
现有环境可使用：

```bash
conda activate jax-fem-env
```

先检查路径并生成已启用阶段的实际配置，不启动耗时计算：

```bash
python opt_solver/run.py dc --dry-run
```

执行完整流程：

```bash
python opt_solver/run.py dc
```

前序结果已经存在时，可以断点继续：

```bash
python opt_solver/run.py dc --from-stage optimization
python opt_solver/run.py dc --from-stage forward
```

## 数据流

1. `am_solver/run.py` 使用小网格执行扫描。
2. 自动选择 AM 输出中编号最大的 `step_*.vtu` 作为优化参考。
3. `IS_solver/opt_run.py` 在同一小网格上反演 `xx/yy/zz`，写入
   `epsilon_result.json`。
4. 启用宏观模型时，编排器把该向量写进大网格配置的
   `inherent_strain.default`，再调用 `IS_solver/run.py`。

输出默认位于项目目录旁边的 `opt_output/<case>/`，与代码和输入网格分开。
其中 `generated_configs/` 保存本次已启用阶段的完整冻结配置，
`workflow_result.json` 是全部结果路径的索引。该路径按项目与 `opt_output`
同级的目录结构配置，移动项目和输出目录时保持两者相对位置即可。

AM 参考 VTU 和优化必须具有相同的节点、单元数量及顺序。示例 AM 输出的
位移单位是 mm，因此 `reference.displacement_scale_to_m` 设为 `0.001`；如果
参考文件中的位移已经是 m，应改为 `1.0`。

新建 case 时直接复制整个案例目录，然后只修改副本内的文件。入口会检查
已启用阶段的 YAML、所需 INP 和显式参考 VTU 均位于当前 case 内。
