#!/usr/bin/env python3
"""Launch the AM -> inherent-strain optimization -> macro forward workflow."""

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


STAGES = ("am", "optimization", "forward")
NUMERIC_STEP = re.compile(r"^step_(\d+)\.vtu$")
SHARED_SOLVER_CONFIG = "solver_config.yaml"


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是映射：{path}")
    return data


def load_solver_config(project_root):
    """Load the one solver policy shared by every case and stage."""
    path = project_root / SHARED_SOLVER_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"公共求解器配置不存在：{path}")

    config = load_yaml(path)
    required_sections = ("am", "optimization", "forward")
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(
                f"公共求解器配置缺少映射段：{section}（文件：{path}）"
            )

    for section, required_keys in {
        "am": ("thermal", "mechanics"),
        "optimization": (),
        "forward": ("forward", "cutting"),
    }.items():
        for key in required_keys:
            if not isinstance(config[section].get(key), dict):
                raise ValueError(
                    f"公共求解器配置缺少映射：{section}.{key}（文件：{path}）"
                )

    return config


def describe_solver_policy(solver_config, path):
    """Print the effective policy so stale generated configs are diagnosable."""
    optimization = solver_config["optimization"]
    print(
        "Shared solver policy: "
        f"config={path}, "
        f"optimization.backend={optimization.get('backend', 'spsolve')}"
    )


def deep_merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_from(base_dir, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def require_inside_case(path, case_dir, label):
    try:
        path.relative_to(case_dir)
    except ValueError as error:
        raise ValueError(
            f"{label} 必须放在当前 case 目录内：{path}"
        ) from error


def resolve_case_path(project_root, argument):
    if argument is None:
        return project_root / "cases" / "dc" / "config.yaml"
    direct = Path(argument).expanduser()
    if direct.exists():
        if direct.is_dir():
            for name in ("config.yaml", f"{direct.name}.yaml"):
                candidate = direct / name
                if candidate.exists():
                    return candidate.resolve()
            raise FileNotFoundError(f"案例目录中找不到 YAML：{direct}")
        return direct.resolve()
    case_dir = project_root / "cases" / argument
    for name in ("config.yaml", f"{argument}.yaml"):
        candidate = case_dir / name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到工作流配置或案例：{argument}")


def source_paths_base(source_config, source_path, solver_root):
    relative_to = source_config.get("paths", {}).get("relative_to", "project")
    if relative_to == "config":
        return source_path.parent
    if relative_to == "project":
        return solver_root
    raise ValueError(f"不支持的 paths.relative_to：{relative_to}")


def normalize_input_paths(config, original_base):
    paths = config.setdefault("paths", {})
    if "mesh" in paths:
        paths["mesh"] = str(resolve_from(original_base, paths["mesh"]))
    toolpath = config.get("toolpath", {})
    if toolpath.get("mode") == "file" and "file" in toolpath:
        toolpath["file"] = str(resolve_from(original_base, toolpath["file"]))
    paths["relative_to"] = "config"


def load_stage_config(workflow, stage, case_dir, solver_root, solver_config):
    stage_spec = workflow["stages"][stage]
    source_path = resolve_from(case_dir, stage_spec["config"])
    require_inside_case(source_path, case_dir, f"stages.{stage}.config")
    if not source_path.is_file():
        raise FileNotFoundError(f"{stage} 基准配置不存在：{source_path}")
    source_config = load_yaml(source_path)
    original_base = source_paths_base(source_config, source_path, solver_root)
    config = deep_merge(source_config, stage_spec.get("overrides", {}))

    # Solver settings are deliberately assigned after case overrides so a
    # case cannot silently diverge from the project-wide solver policy.
    if stage == "optimization":
        optimization = config.setdefault("optimization", {})
        optimization["solver"] = copy.deepcopy(solver_config["optimization"])
    elif stage == "am":
        config["solver"] = copy.deepcopy(solver_config["am"])
    elif stage == "forward":
        config["solver"] = copy.deepcopy(solver_config["forward"])
    else:
        raise ValueError(f"不支持的阶段：{stage}")

    normalize_input_paths(config, original_base)
    return config


def validate_local_toolpath(config, case_dir, label):
    toolpath = config.get("toolpath", {})
    if toolpath.get("mode") == "file":
        toolpath_path = Path(toolpath["file"])
        require_inside_case(toolpath_path, case_dir, f"{label} toolpath")
        if not toolpath_path.is_file():
            raise FileNotFoundError(f"{label} toolpath 不存在：{toolpath_path}")


def save_generated_config(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)


def run_stage(label, python, script, config_path):
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}", flush=True)
    # The outer parameter runner pipes stdout.  Its own -u does not carry
    # over to this child, so force live stage output instead of buffering a
    # whole optimization until process exit.
    subprocess.run(
        [python, "-u", str(script), str(config_path)],
        cwd=str(script.parent),
        check=True,
    )


def find_latest_am_step(vtk_dir):
    candidates = []
    for path in vtk_dir.glob("step_*.vtu"):
        match = NUMERIC_STEP.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"AM 输出中没有数值 step_*.vtu：{vtk_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def read_inherent_strain(result_path, require_success):
    if not result_path.is_file():
        raise FileNotFoundError(f"优化结果不存在：{result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    optimization = result.get("optimization", {})
    if require_success and not optimization.get("success", False):
        raise RuntimeError(
            "优化器未成功收敛：" + str(optimization.get("message", "unknown"))
        )
    strain = result["inherent_strain"]
    return {axis: float(strain[axis]) for axis in ("xx", "yy", "zz")}


def macro_model_enabled(workflow):
    enabled = workflow.get("run_macro_model", True)
    if not isinstance(enabled, bool):
        raise ValueError("run_macro_model 必须是 YAML 布尔值 true 或 false")
    return enabled


def validate_workflow(workflow):
    run_macro_model = macro_model_enabled(workflow)
    if "solver_options" in workflow:
        raise ValueError(
            "求解器设置必须放在项目根目录的 solver_config.yaml，"
            "不要在 case config.yaml 中设置 solver_options"
        )
    for section in ("paths", "solvers", "stages"):
        if section not in workflow:
            raise KeyError(f"工作流 YAML 缺少配置段：{section}")
    required_paths = ["small_mesh", "output_directory"]
    if run_macro_model:
        required_paths.append("large_mesh")
    for path_name in required_paths:
        if path_name not in workflow["paths"]:
            raise KeyError(f"paths 缺少：{path_name}")
    required_stages = ["am", "optimization"]
    if run_macro_model:
        required_stages.append("forward")
    for stage in required_stages:
        if stage not in workflow["stages"]:
            raise KeyError(f"stages 缺少：{stage}")
        if "config" not in workflow["stages"][stage]:
            raise KeyError(f"stages.{stage} 缺少 config")
    reference = workflow.get("reference") or {}
    displacement_scale = float(reference.get("displacement_scale_to_m", 1.0))
    if displacement_scale <= 0.0:
        raise ValueError("reference.displacement_scale_to_m 必须大于零")


def write_manifest(
    path,
    config_path,
    artifacts,
    completed,
    dry_run,
    inherent_strain,
    run_macro_model,
):
    payload = {
        "workflow_config": str(config_path),
        "dry_run": dry_run,
        "run_macro_model": run_macro_model,
        "completed_stages": completed,
        "inherent_strain": inherent_strain,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="依次执行小网格 AM、固有应变优化和大网格正向求解。"
    )
    parser.add_argument("case", nargs="?", help="案例名、案例目录或 YAML 路径")
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default="am",
        help="从指定阶段继续，默认从 am 开始",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查路径并生成启用阶段的配置，不启动求解",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = resolve_case_path(project_root, args.case)
    case_dir = config_path.parent
    workflow = load_yaml(config_path)
    validate_workflow(workflow)
    run_macro_model = macro_model_enabled(workflow)
    if not run_macro_model and args.from_stage == "forward":
        raise ValueError(
            "run_macro_model 为 false 时不能使用 --from-stage forward"
        )

    repo_root = project_root.parent
    shared_solver_config = load_solver_config(repo_root)
    describe_solver_policy(
        shared_solver_config,
        repo_root / SHARED_SOLVER_CONFIG,
    )
    am_root = resolve_from(case_dir, workflow["solvers"].get("am_root", repo_root / "am_solver"))
    is_root = resolve_from(case_dir, workflow["solvers"].get("is_root", repo_root / "IS_solver"))
    am_script = am_root / "run.py"
    opt_script = is_root / "opt_run.py"
    forward_script = is_root / "run.py"
    required_scripts = [am_script, opt_script]
    if run_macro_model:
        required_scripts.append(forward_script)
    for script in required_scripts:
        if not script.is_file():
            raise FileNotFoundError(f"求解器入口不存在：{script}")

    small_mesh = resolve_from(case_dir, workflow["paths"]["small_mesh"])
    required_meshes = [("小网格", small_mesh)]
    if run_macro_model:
        large_mesh = resolve_from(case_dir, workflow["paths"]["large_mesh"])
        required_meshes.append(("大网格", large_mesh))
    for label, mesh_path in required_meshes:
        require_inside_case(mesh_path, case_dir, label)
        if not mesh_path.is_file():
            raise FileNotFoundError(f"{label}不存在：{mesh_path}")

    output_root = resolve_from(case_dir, workflow["paths"]["output_directory"])
    generated_dir = output_root / "generated_configs"
    am_dir = output_root / "am"
    opt_dir = output_root / "optimization"
    forward_dir = output_root / "forward"
    output_root.mkdir(parents=True, exist_ok=True)

    execution = workflow.get("execution") or {}
    python = str(execution.get("python", sys.executable))
    start_index = STAGES.index(args.from_stage)
    stage_total = 3 if run_macro_model else 2
    completed = []
    artifacts = {}

    am_config = load_stage_config(
        workflow,
        "am",
        case_dir,
        am_root,
        shared_solver_config,
    )
    validate_local_toolpath(am_config, case_dir, "AM")
    am_config["paths"].update({
        "mesh": str(small_mesh),
        "output_directory": str(am_dir),
        "vtk_directory": str(am_dir / "vtk"),
        "mesh_output": str(am_dir / "mesh" / "small_mesh.vtu"),
    })
    am_generated = generated_dir / "am.yaml"
    save_generated_config(am_config, am_generated)
    artifacts["am_config"] = am_generated

    reference_config = workflow.get("reference") or {}
    explicit_reference = reference_config.get("vtu")
    if start_index == 0 and not args.dry_run:
        run_stage(
            f"1/{stage_total} 小网格热-力 AM 扫描",
            python,
            am_script,
            am_generated,
        )
        completed.append("am")
    if explicit_reference:
        reference_vtu = resolve_from(case_dir, explicit_reference)
        require_inside_case(reference_vtu, case_dir, "reference.vtu")
    elif args.dry_run or start_index == 2:
        reference_vtu = am_dir / "vtk" / "__latest_numeric_step__.vtu"
    else:
        reference_vtu = find_latest_am_step(am_dir / "vtk")
    artifacts["am_reference_vtu"] = reference_vtu

    opt_config = load_stage_config(
        workflow,
        "optimization",
        case_dir,
        is_root,
        shared_solver_config,
    )
    validate_local_toolpath(opt_config, case_dir, "optimization")
    opt_config["paths"].update({
        "mesh": str(small_mesh),
        "reference_vtu": str(reference_vtu),
        "output_directory": str(opt_dir),
        "vtk_directory": str(opt_dir / "vtk"),
    })
    opt_config.setdefault("optimization", {})[
        "reference_displacement_scale_to_m"
    ] = float(reference_config.get("displacement_scale_to_m", 1.0))
    epsilon_result = opt_dir / "epsilon_result.json"
    opt_config.setdefault("output", {})["epsilon_result_json"] = str(epsilon_result)
    opt_generated = generated_dir / "optimization.yaml"
    save_generated_config(opt_config, opt_generated)
    artifacts["optimization_config"] = opt_generated
    artifacts["epsilon_result_json"] = epsilon_result
    artifacts["optimization_forward_error_vtu"] = opt_dir / "vtk" / "optimized_forward_error.vtu"

    if start_index <= 1 and not args.dry_run:
        if not reference_vtu.is_file():
            raise FileNotFoundError(f"优化参考 VTU 不存在：{reference_vtu}")
        run_stage(
            f"2/{stage_total} 小网格固有应变优化",
            python,
            opt_script,
            opt_generated,
        )
        completed.append("optimization")

    if args.dry_run:
        values = opt_config["optimization"]["initial_params"]
        if bool(opt_config["optimization"].get("tie_xy", False)):
            if len(values) != 2:
                raise ValueError(
                    "optimization.tie_xy=true 时 initial_params 必须是 [eps_xx_yy, eps_zz]"
                )
            inherent_strain = {
                "xx": float(values[0]),
                "yy": float(values[0]),
                "zz": float(values[1]),
            }
        else:
            inherent_strain = dict(zip(("xx", "yy", "zz"), map(float, values)))
    else:
        require_success = bool(
            workflow["stages"]["optimization"].get("require_success", False)
        )
        inherent_strain = read_inherent_strain(epsilon_result, require_success)

    if run_macro_model:
        forward_config = load_stage_config(
            workflow,
            "forward",
            case_dir,
            is_root,
            shared_solver_config,
        )
        validate_local_toolpath(forward_config, case_dir, "forward")
        forward_config["paths"].update({
            "mesh": str(large_mesh),
            "output_directory": str(forward_dir),
            "vtk_directory": str(forward_dir / "vtk"),
        })
        postprocess_config = forward_config.get("postprocess", {})
        if postprocess_config.get("enabled", False):
            postprocess_config["output_directory"] = str(
                forward_dir / "postprocess"
            )
        forward_config.setdefault("inherent_strain", {})[
            "default"
        ] = inherent_strain
        forward_generated = generated_dir / "forward.yaml"
        save_generated_config(forward_config, forward_generated)
        artifacts["forward_config"] = forward_generated
        artifacts["forward_vtk_directory"] = forward_dir / "vtk"

        if start_index <= 2 and not args.dry_run:
            run_stage(
                "3/3 大网格固有应变正向求解",
                python,
                forward_script,
                forward_generated,
            )
            completed.append("forward")

    manifest_path = output_root / "workflow_result.json"
    write_manifest(
        manifest_path,
        config_path,
        artifacts,
        completed,
        args.dry_run,
        inherent_strain,
        run_macro_model,
    )
    action = "配置检查完成" if args.dry_run else "完整流程完成"
    print(f"\n{action}。结果索引：{manifest_path}")


if __name__ == "__main__":
    main()
