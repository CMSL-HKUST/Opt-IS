"""Check that shipped cases remain portable through the public workflow CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "workflow" / "run.py"
CASES = ROOT / "workflow" / "cases"


class WorkflowCaseTests(unittest.TestCase):
    def prepare_case(self, destination, original, macro=None):
        case = destination / original.parent.name
        case.mkdir()
        for source in original.parent.rglob("*"):
            if source.suffix not in (".yaml", ".inp"):
                continue
            target = case / source.relative_to(original.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".inp":
                os.link(source, target)
            else:
                target.write_text(source.read_text())
        config_path = case / original.name
        config = yaml.safe_load(config_path.read_text())
        config["paths"]["output_directory"] = str(destination / "results")
        config["solvers"] = {
            "am_root": str(ROOT / "am_solver"),
            "is_root": str(ROOT / "IS_solver"),
        }
        if macro is not None:
            config["run_macro_model"] = macro
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        return config_path, config

    def test_all_supplied_workflows_generate_consistent_stage_inputs(self):
        configs = sorted(CASES.glob("*/config*.yaml"))
        self.assertEqual({p.parent.name for p in configs},
                         {"cantilever", "dc", "bridge", "dc2"})
        for original in configs:
            for macro in (False, True):
                with self.subTest(case=str(original.relative_to(CASES)), macro=macro):
                    with tempfile.TemporaryDirectory() as directory:
                        config_path, config = self.prepare_case(
                            Path(directory), original, macro
                        )
                        subprocess.run(
                            [sys.executable, str(LAUNCHER), str(config_path), "--dry-run"],
                            cwd=directory, check=True, capture_output=True, text=True,
                        )
                        results = Path(config["paths"]["output_directory"])
                        manifest = json.loads((results / "workflow_result.json").read_text())
                        self.assertTrue(manifest["dry_run"])
                        self.assertEqual(manifest["completed_stages"], [])
                        generated = results / "generated_configs"
                        am = yaml.safe_load((generated / "am.yaml").read_text())
                        opt = yaml.safe_load((generated / "optimization.yaml").read_text())
                        expected_mesh = str((config_path.parent /
                                             config["paths"]["small_mesh"]).resolve())
                        self.assertEqual(am["paths"]["mesh"], expected_mesh)
                        self.assertEqual(opt["paths"]["mesh"], expected_mesh)
                        self.assertEqual((generated / "forward.yaml").exists(), macro)
                        if macro:
                            forward = yaml.safe_load((generated / "forward.yaml").read_text())
                            self.assertEqual(forward["inherent_strain"]["default"],
                                             manifest["inherent_strain"])
                            self.assertTrue(Path(forward["paths"]["mesh"]).is_file())

    def test_forward_resume_is_rejected_for_local_only_cases(self):
        for name in ("dc", "dc2"):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                path, _ = self.prepare_case(Path(directory), CASES / name / "config.yaml")
                result = subprocess.run(
                    [sys.executable, str(LAUNCHER), str(path),
                     "--from-stage", "forward", "--dry-run"],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("run_macro_model", result.stderr)

    def test_standalone_case_input_files_are_present(self):
        for component in ("am_solver", "IS_solver"):
            for path in sorted((ROOT / component / "cases").glob("*/*.yaml")):
                with self.subTest(config=str(path.relative_to(ROOT))):
                    config = yaml.safe_load(path.read_text())
                    self.assertEqual(config["paths"]["relative_to"], "config")
                    mesh = config["paths"].get("mesh")
                    if mesh:
                        self.assertTrue((path.parent / mesh).is_file())
                    else:
                        self.assertEqual(config["mesh"]["mode"], "structured_box")


if __name__ == "__main__":
    unittest.main()
