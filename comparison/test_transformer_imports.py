"""Test optional transformer imports without requiring training dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class TestTransformerImports(unittest.TestCase):
    """Load the actual package initializer independently of graphnet/torch."""

    def setUp(self):
        """Give the initializer an isolated package name for relative imports."""
        path = (
            Path(__file__).resolve().parents[1]
            / "src/graphnet/models/transformer/__init__.py"
        )
        self.name = "_graphnet_transformer_import_test"
        spec = importlib.util.spec_from_file_location(self.name, path)
        self.package = importlib.util.module_from_spec(spec)
        self.modules = patch.dict(sys.modules, {self.name: self.package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.spec = spec

    def test_package_import_does_not_load_iseecube(self):
        """Importing sibling models must not require ISeeCube dependencies."""
        with patch.dict(sys.modules, {self.name + ".iseecube": None}):
            self.spec.loader.exec_module(self.package)
        self.assertNotIn("ISeeCube", vars(self.package))
        self.assertIn("ISeeCube", dir(self.package))
        with self.assertRaises(AttributeError):
            getattr(self.package, "UnknownModel")

    def test_public_import_resolves_and_caches_iseecube(self):
        """The existing package-level import still returns the model class."""
        self.spec.loader.exec_module(self.package)
        module = ModuleType(self.name + ".iseecube")
        module.ISeeCube = type("ISeeCube", (), {})
        with patch.dict(sys.modules, {module.__name__: module}):
            namespace = {}
            exec(f"from {self.name} import ISeeCube", namespace)
        self.assertIs(namespace["ISeeCube"], module.ISeeCube)
        self.assertIs(self.package.ISeeCube, module.ISeeCube)

    def test_missing_dependency_error_is_preserved(self):
        """Requesting ISeeCube still reports its missing optional dependency."""
        self.spec.loader.exec_module(self.package)
        error = ModuleNotFoundError("No module named 'torchscale'")
        with patch.object(self.package, "import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as caught:
                self.package.ISeeCube
        self.assertIs(caught.exception, error)
        self.assertNotIn("ISeeCube", vars(self.package))


if __name__ == "__main__":
    unittest.main()

