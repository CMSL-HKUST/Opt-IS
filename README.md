# ICIS_solver

增材制造（AM）热-力模拟、固有应变优化与宏观正向求解代码。

## 目录结构

- `am_solver/`：AM 热-力求解器。
- `IS_solver/`：固有应变正向求解与优化模块。
- `opt_solver/`：串联 AM、固有应变优化和宏观正向求解的流程入口。
- `opt_solver/cases/cantilever/`：悬臂梁案例配置与输入网格。
- `solver_config.yaml`：各阶段共用的求解器配置。

流程说明见 [opt_solver/README.md](opt_solver/README.md)。其中案例命令中的 `dc` 在当前目录结构下应替换为 `cantilever`；实际求解器参数以根目录的 `solver_config.yaml` 为准。

## 版本管理

本仓库保存源代码、案例配置与输入网格。Python 缓存、本地虚拟环境、环境变量文件和仓库内的 `opt_output/` 输出目录由 `.gitignore` 排除。

修改并检查后，在仓库根目录执行：

```bash
git status
git add .
git commit -m "说明这次修改的内容"
git push
```

在另一台电脑继续工作前，先执行 `git pull --ff-only`，获取远程最新提交。
