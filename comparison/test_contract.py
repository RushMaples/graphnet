"""CPU checks: python -m unittest discover -s comparison -p 'test_*.py'."""

import csv
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from contract import (
    DATABASES,
    FEATURES,
    SPLITS,
    TRUTH,
    audit_databases,
    event_seed,
    load_splits,
    sha256,
    write_json,
)
from evaluate import check_probes


class FrozenInputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract = {"selections": {}}
        for database in DATABASES:
            for i, split in enumerate(SPLITS):
                name = f"{Path(database).stem}_{split}.csv"
                path = self.root / name
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerows(
                        [["event_no"], [i * 10 + 1], [i * 10 + 2]]
                    )
                self.contract["selections"][name] = {
                    "count": 2,
                    "sha256": sha256(path),
                }

    def test_composite_keys_allow_same_id_in_different_databases(self):
        result = load_splits(self.root, self.contract)
        self.assertEqual(result["train"], [[1, 2], [1, 2], [1, 2]])

    def test_modified_selection_rejected(self):
        path = next(self.root.glob("*.csv"))
        path.write_text("event_no\n999\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_splits(self.root, self.contract)

    def test_leakage_rejected_even_with_updated_hash(self):
        name = f"{Path(DATABASES[0]).stem}_validation.csv"
        path = self.root / name
        path.write_text("event_no\n1\n12\n")
        self.contract["selections"][name]["sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_splits(self.root, self.contract)

    def test_repeated_ids_rejected(self):
        name = f"{Path(DATABASES[0]).stem}_validation.csv"
        path = self.root / name
        path.write_text("event_no\n12\n12\n")
        self.contract["selections"][name]["sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_splits(self.root, self.contract)

    def test_seed_is_independent_of_staging_path_and_distinguishes_database(
        self,
    ):
        self.assertEqual(
            event_seed("/tmp/a/nue.db", 42), event_seed("/other/nue.db", 42)
        )
        self.assertNotEqual(
            event_seed("nue.db", 42), event_seed("numu.db", 42)
        )
        self.assertNotEqual(event_seed("nue.db", 42), event_seed("nue.db", 43))


class ProbeTest(unittest.TestCase):
    def test_input_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [
                Path(temporary) / variant for variant in ("carlos", "ajai")
            ]
            for i, root in enumerate(roots):
                root.mkdir()
                write_json(
                    root / "resolved_settings.json",
                    {
                        "epochs": 10,
                        "variant": root.name,
                        "group_factor": i * 16,
                    },
                )
                write_json(
                    root / "database_manifest.json",
                    [{"database": "nue.db", "sha256": "same"}],
                )
                write_json(
                    root / "provenance.json",
                    {
                        key: "same"
                        for key in (
                            "python",
                            "torch",
                            "cuda_runtime",
                            "lightning",
                            "gpu",
                            "pip_freeze",
                        )
                    },
                )
                write_json(
                    root / "input_probe.json",
                    [{"event_no": 1, "input_sha256": str(i)}],
                )
            with self.assertRaisesRegex(ValueError, "Input probes differ"):
                check_probes(*roots)

    def test_coverage_uniform_limit(self):
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            self.skipTest("NumPy/pandas unavailable")
        from evaluate import event_metrics

        frame = pd.DataFrame(
            {
                "dir_x_pred": [1.0],
                "dir_y_pred": [0.0],
                "dir_z_pred": [0.0],
                "azimuth": [0.0],
                "zenith": [0.0],
                "direction_kappa": [0.0],
                **{f"position_{axis}": [0.0] for axis in "xyz"},
                **{f"position_{axis}_pred": [0.0] for axis in "xyz"},
            }
        )
        metrics = event_metrics(frame)
        self.assertTrue(np.isfinite(metrics["cone_cdf"]).all())
        self.assertAlmostEqual(metrics["cone_cdf"][0], 0.5)
        self.assertAlmostEqual(metrics["angle_deg"][0], 90.0)


class DatabaseAuditTest(unittest.TestCase):
    def test_missing_selected_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = {
                split: [[i + 1] for _ in DATABASES]
                for i, split in enumerate(SPLITS)
            }
            for database in DATABASES:
                with closing(sqlite3.connect(root / database)) as connection:
                    columns = ", ".join(f'"{name}" REAL' for name in TRUTH)
                    connection.execute(
                        f"CREATE TABLE truth (event_no INTEGER, {columns})"
                    )
                    columns = ", ".join(f'"{name}" REAL' for name in FEATURES)
                    connection.execute(
                        f"CREATE TABLE SRTInIcePulses (event_no INTEGER, {columns})"
                    )
                    connection.executemany(
                        "INSERT INTO truth (event_no) VALUES (?)", [(1,), (2,)]
                    )
                    connection.commit()
            with self.assertRaisesRegex(ValueError, "selected events missing"):
                audit_databases(root, splits)


if __name__ == "__main__":
    unittest.main()
