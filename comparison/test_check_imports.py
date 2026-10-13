"""Test import-check process isolation without training dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "comparison_check_imports", Path(__file__).with_name("check_imports.py")
)
check_imports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_imports)


class TestCheckImports(unittest.TestCase):
    """Verify failure aggregation and isolation needed for paired imports."""

    def test_failed_variant_does_not_skip_other_source(self):
        """A missing Ajai dependency must not hide the Carlos check."""
        with patch.object(
            check_imports.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(returncode=1),
                SimpleNamespace(returncode=0),
            ],
        ) as run:
            result = check_imports.run_checks(
                ["ajai", "carlos"], "original-source", "new-source"
            )
        self.assertEqual(result, 1)
        self.assertEqual(run.call_count, 2)
        for call, variant, root in zip(
            run.call_args_list,
            ["ajai", "carlos"],
            ["new-source", "original-source"],
        ):
            command = call.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[2:4], ["--worker", variant])
            self.assertEqual(command[-1], str(Path(root).resolve()))
            self.assertEqual(call.kwargs["env"]["PYTHONNOUSERSITE"], "1")
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_all_pass_returns_success(self):
        """Both source checks must succeed to report overall success."""
        with patch.object(
            check_imports.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ):
            self.assertEqual(
                check_imports.run_checks(["carlos", "ajai"], "old", "new"), 0
            )


if __name__ == "__main__":
    unittest.main()
