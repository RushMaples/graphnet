"""CPU artifact checks for augmentation selection; no model or GPU imports."""

import contextlib
import csv
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import augmentation
from contract import TRUTH, load_contract, sha256, write_json


class AugmentationConfigTest(unittest.TestCase):
    def test_carlos_rates_are_conditional(self):
        enabled = augmentation.augmentation_settings("both")
        self.assertEqual(enabled["token_drop"], 0.05)
        self.assertEqual(enabled["drop_chance"], 0.05)
        self.assertTrue(enabled["rotation"])
        self.assertAlmostEqual(
            enabled["token_drop"] * enabled["drop_chance"], 0.0025
        )
        disabled = augmentation.augmentation_settings("baseline")
        self.assertEqual(disabled["token_drop"], 0.0)
        self.assertFalse(disabled["rotation"])

    def test_invalid_rates_and_unknown_arm_fail(self):
        for rate in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                augmentation.augmentation_settings("both", token_drop=rate)
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                augmentation.augmentation_settings("both", drop_chance=rate)
        with self.assertRaises(ValueError):
            augmentation.augmentation_settings("typo")


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas unavailable")
class ScreenArtifactTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        counts = patch.object(
            augmentation, "COUNTS", {"train": 2, "validation": 2, "test": 2}
        )
        counts.start()
        self.addCleanup(counts.stop)
        for arm, score in zip(augmentation.ARMS, (4.0, 3.0, 2.0, 1.0)):
            self.make_arm(arm, score)

    def make_arm(self, arm, score):
        root = self.root / arm
        validation = root / "validation"
        validation.mkdir(parents=True)
        cfg = load_contract()
        settings = augmentation.augmentation_settings(arm)
        cfg.update(
            stage="screen",
            arm=arm,
            epochs=2,
            augmentation=settings,
            rotation_augmentation=settings["rotation"],
        )
        write_json(root / "resolved_settings.json", cfg)
        write_json(
            root / "complete.json",
            {"epochs": 2, "validation_rows": 2, "test_exported": False},
        )
        checkpoint_path = root / "best.ckpt"
        checkpoint_path.write_bytes(b"synthetic checkpoint")
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "epoch": 1,
            "monitor": "val_loss",
            "score": score,
        }
        write_json(root / "selected_checkpoint.json", checkpoint)
        write_json(
            root / "provenance.json",
            {
                **{
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
                "source_files": [{"path": "src/model.py", "sha256": "same"}],
                "harness_files": [
                    {"path": "comparison/augmentation.py", "sha256": "same"}
                ],
            },
        )
        write_json(
            root / "database_manifest.json",
            [{"database": "nue.db", "sha256": "same"}],
        )
        write_json(
            root / "input_probe.json",
            [{"event_no": 1, "input_sha256": "same"}],
        )
        with (validation / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_database",
                    "event_no",
                    "input_sha256",
                    *TRUTH,
                ],
            )
            writer.writeheader()
            for event in (1, 2):
                writer.writerow(
                    {
                        "source_database": "nue.db",
                        "event_no": event,
                        "input_sha256": f"input-{event}",
                        **{key: event for key in TRUTH},
                    }
                )
        write_json(
            validation / "prediction_manifest.json",
            {
                "split": "validation",
                "rows": 2,
                "checkpoint": checkpoint,
                "predictions_sha256": sha256(validation / "predictions.csv"),
            },
        )
        with (validation / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "topology",
                    "energy_min_GeV",
                    "angle_deg_median",
                    "vertex_m_median",
                    "coverage_0.68",
                ],
            )
            writer.writeheader()
            # Deliberately oppose the loss ranking: diagnostics cannot select.
            writer.writerow(
                {
                    "topology": "all",
                    "energy_min_GeV": "",
                    "angle_deg_median": 10 - score,
                    "vertex_m_median": 20 - score,
                    "coverage_0.68": 0.68,
                }
            )

    def mutate_json(self, relative, update):
        path = self.root / relative
        value = augmentation.read_json(path)
        update(value)
        write_json(path, value)

    def summarize(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return augmentation.summarize(self.root)

    def test_selects_validation_loss_without_any_test_artifacts(self):
        receipt = self.summarize()
        self.assertEqual(receipt["selected_arm"], "both")
        self.assertFalse(receipt["test_used_for_selection"])
        self.assertTrue((self.root / "summary.csv").is_file())
        self.assertFalse(any(self.root.glob("*/test*")))

    def test_rejects_unequal_training_settings(self):
        self.mutate_json(
            "rotation/resolved_settings.json",
            lambda value: value.update(lr=0.001),
        )
        with self.assertRaises(ValueError):
            self.summarize()

    def test_rejects_augmentation_mislabeled_as_baseline(self):
        self.mutate_json(
            "baseline/resolved_settings.json",
            lambda value: value["augmentation"].update(rotation=True),
        )
        with self.assertRaisesRegex(ValueError, "describe arm"):
            self.summarize()

    def test_rejects_different_dropout_rates_between_arms(self):
        self.mutate_json(
            "both/resolved_settings.json",
            lambda value: value["augmentation"].update(token_drop=0.1),
        )
        with self.assertRaisesRegex(ValueError, "rates differ"):
            self.summarize()

    def test_rejects_changed_harness(self):
        self.mutate_json(
            "both/provenance.json",
            lambda value: value.update(harness_files=[{"sha256": "changed"}]),
        )
        with self.assertRaisesRegex(ValueError, "differ beyond"):
            self.summarize()

    def test_rejects_test_export_during_screen(self):
        self.mutate_json(
            "rotation/complete.json",
            lambda value: value.update(test_exported=True),
        )
        with self.assertRaises(ValueError):
            self.summarize()

    def test_rejects_checkpoint_tampering(self):
        (self.root / "both" / "best.ckpt").write_bytes(b"changed checkpoint")
        with self.assertRaises(ValueError):
            self.summarize()

    def test_rejects_different_validation_pulses_even_with_updated_file_hash(
        self,
    ):
        path = self.root / "rotation" / "validation" / "predictions.csv"
        path.write_text(
            path.read_text().replace("input-1", "different-pulses")
        )
        self.mutate_json(
            "rotation/validation/prediction_manifest.json",
            lambda value: value.update(predictions_sha256=sha256(path)),
        )
        with self.assertRaisesRegex(ValueError, "retained pulses"):
            self.summarize()

    def test_rejects_same_pulse_keys_with_different_truth(self):
        path = self.root / "rotation" / "validation" / "predictions.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["position_x"] = "999"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.mutate_json(
            "rotation/validation/prediction_manifest.json",
            lambda value: value.update(predictions_sha256=sha256(path)),
        )
        with self.assertRaisesRegex(ValueError, "truth"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
