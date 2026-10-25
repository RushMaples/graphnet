"""CPU continuation contracts; no Torch, Lightning, GraphNeT or GPU needed."""

import contextlib
import copy
import csv
import importlib.util
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continue_augmentation as continuation
from augmentation import augmentation_settings
from contract import TRUTH, load_contract, sha256, write_json


def settings(arm="baseline"):
    cfg = load_contract()
    aug = augmentation_settings(arm)
    cfg.update(
        arm=arm,
        variant="ajai",
        stage="screen",
        epochs=10,
        augmentation=aug,
        rotation_augmentation=aug["rotation"],
        group_factor=16,
        drop_last=False,
        heldout_policy="p14-eval-v1",
    )
    return cfg


def complete(epochs=10):
    return {
        "epochs": epochs,
        "train_rows": continuation.COUNTS["train"],
        "validation_rows": continuation.COUNTS["validation"],
        "test_exported": False,
        "test_rows": 0,
    }


def checkpoint(epochs=10):
    return {
        "epoch": epochs - 1,
        "global_step": 4517 * epochs,
        "state_dict": {"model.weight": [1]},
        "loops": {"fit_loop": {"progress": 10}},
        "optimizer_states": [
            {"state": {0: {"step": 45170}}, "param_groups": [{"lr": 6.25e-5}]}
        ],
        "lr_schedulers": [
            {"best": 1.66, "num_bad_epochs": 1, "last_epoch": epochs}
        ],
        "MixedPrecision": {"scale": 65536, "_growth_tracker": 10},
    }


class SettingsTests(unittest.TestCase):
    def test_accepts_original_screen_and_preserves_parent_settings(self):
        for arm in continuation.PAIRED_ARMS:
            cfg = settings(arm)
            before = copy.deepcopy(cfg)
            self.assertEqual(
                continuation.validate_settings(cfg, complete(), 30), 10
            )
            continued = continuation.continuation_settings(cfg, "/parent", 30)
            self.assertEqual(cfg, before)
            self.assertEqual(continued["starting_epochs"], 10)
            self.assertEqual(continued["epochs"], 30)
            self.assertEqual(continued["augmentation"], cfg["augmentation"])
            self.assertEqual(
                continuation.validate_settings(continued, complete(30), 80), 30
            )

    def test_rejects_changed_frozen_hyperparameters_and_selection(self):
        changes = {
            "seed": 43,
            "batch_size": 64,
            "max_pulses": 512,
            "group_factor": 8,
            "precision": "32-true",
            "drop_last": True,
            "lr": 0.001,
            "scheduler_patience": 3,
            "accumulate_grad_batches": 2,
        }
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_settings(
                    dict(settings(), **{key: value}), complete(), 30
                )
        cfg = settings()
        next(iter(cfg["selections"].values()))["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "frozen"):
            continuation.validate_settings(cfg, complete(), 30)

    def test_rejects_changed_augmentation_rates_and_pivot(self):
        for key, value in (
            ("token_drop", 0.1),
            ("drop_chance", 0.1),
            ("rotation_pivot_m", [0, 0]),
            ("rotation", False),
        ):
            cfg = settings("both")
            cfg["augmentation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_settings(cfg, complete(), 30)

    def test_rejects_nonpaired_or_wrong_stage_and_screen_horizon(self):
        for update in (
            {"arm": "rotation"},
            {"stage": "full"},
            {"epochs": 9},
            {"epochs": 11},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                continuation.validate_settings(
                    dict(settings(), **update), complete(), 30
                )

    def test_rejects_incomplete_or_test_using_parent(self):
        for key, value in (
            ("epochs", 9),
            ("train_rows", 1),
            ("validation_rows", 1),
            ("test_exported", True),
            ("test_rows", 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_settings(
                    settings(), dict(complete(), **{key: value}), 30
                )

    def test_requires_strictly_larger_total_horizon(self):
        for target in (0, 9, 10, 10.5, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                continuation.validate_settings(settings(), complete(), target)

    def test_continuation_requires_valid_ancestry(self):
        cfg = continuation.continuation_settings(settings(), "/parent", 30)
        for update in (
            {"starting_epochs": 30},
            {"starting_epochs": 0},
            {"parent_run": ""},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                continuation.validate_settings(
                    dict(cfg, **update), complete(30), 80
                )


class CheckpointTests(unittest.TestCase):
    def test_accepts_full_lightning_24_checkpoint_and_legacy_amp(self):
        continuation.validate_checkpoint(checkpoint(), 10)
        state = checkpoint()
        state["native_amp_scaling_state"] = state.pop("MixedPrecision")
        continuation.validate_checkpoint(state, 10)

    def test_rejects_partial_or_wrong_epoch_and_global_step(self):
        for key, value in (
            ("epoch", 8),
            ("global_step", 45169),
            ("state_dict", {}),
            ("loops", {}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_checkpoint(
                    dict(checkpoint(), **{key: value}), 10
                )

    def test_rejects_weights_only_or_missing_optimizer_scheduler_amp(self):
        for key in ("optimizer_states", "lr_schedulers", "MixedPrecision"):
            state = checkpoint()
            del state[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_checkpoint(state, 10)
        state = checkpoint()
        state["optimizer_states"][0]["state"] = {}
        with self.assertRaisesRegex(ValueError, "optimizer"):
            continuation.validate_checkpoint(state, 10)

    def test_parent_best_survives_worse_new_scores_and_ties(self):
        parent = {"monitor": "val_loss", "score": 1.66, "epoch": 8}
        for score in (1.7, 1.66):
            self.assertIs(
                continuation.choose_checkpoint(
                    parent, dict(parent, score=score, epoch=29)
                ),
                parent,
            )
        new = dict(parent, score=1.6, epoch=20)
        self.assertIs(continuation.choose_checkpoint(parent, new), new)
        with self.assertRaises(ValueError):
            continuation.choose_checkpoint(
                parent, dict(new, score=float("nan"))
            )


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_original_files_immutable_but_new_files_allowed(self):
        model = self.root / "src" / "model.py"
        model.parent.mkdir()
        model.write_bytes(b"original\n")
        harness = self.root / "comparison" / "augmentation.py"
        harness.parent.mkdir()
        harness.write_bytes(b"original harness\n")
        provenance = {
            "source_files": [
                {"path": "src/model.py", "sha256": sha256(model)}
            ],
            "harness_files": [
                {
                    "path": "comparison/augmentation.py",
                    "sha256": sha256(harness),
                }
            ],
        }
        (harness.parent / "continue_augmentation.py").write_text("new code")
        continuation.verify_original_files(provenance, self.root, self.root)
        harness.write_bytes(b"changed harness\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            continuation.verify_original_files(
                provenance, self.root, self.root
            )
        harness.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            continuation.verify_original_files(
                provenance, self.root, self.root
            )

    def test_relocated_selected_checkpoint_requires_matching_hash(self):
        folder = self.root / "checkpoints"
        folder.mkdir()
        local = folder / "best.ckpt"
        local.write_bytes(b"checkpoint")
        record = {
            "path": "/old/run/checkpoints/best.ckpt",
            "sha256": sha256(local),
        }
        self.assertEqual(
            continuation.checkpoint_path(self.root, record), local.resolve()
        )
        local.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            continuation.checkpoint_path(self.root, record)

    def timing(self, cfg):
        fields = [
            "epoch",
            "train_events",
            "validation_events",
            "optimizer_steps",
            "val_loss",
        ]
        with (self.root / "timing.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(cfg.get("starting_epochs", 0), cfg["epochs"]):
                writer.writerow(
                    dict(
                        epoch=epoch,
                        train_events=continuation.COUNTS["train"],
                        validation_events=continuation.COUNTS["validation"],
                        optimizer_steps=4517,
                        val_loss=1.7,
                    )
                )

    def test_parent_exposure_and_epoch_sequence_required(self):
        cfg = settings()
        self.timing(cfg)
        continuation.validate_timing(self.root, cfg)
        path = self.root / "timing.csv"
        path.write_text(path.read_text().replace(",4517,", ",4516,", 1))
        with self.assertRaisesRegex(ValueError, "exposure"):
            continuation.validate_timing(self.root, cfg)
        continued = continuation.continuation_settings(cfg, "/parent", 30)
        self.timing(continued)
        continuation.validate_timing(self.root, continued)
        with self.assertRaisesRegex(ValueError, "horizon"):
            continuation.validate_timing(self.root, cfg)


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("numpy"),
    "pandas/numpy unavailable",
)
class PairedSummaryTests(ArtifactTests):
    def setUp(self):
        super().setUp()
        counts = patch.object(
            continuation, "COUNTS", {"train": 2, "validation": 2, "test": 2}
        )
        counts.start()
        self.addCleanup(counts.stop)
        for arm in continuation.PAIRED_ARMS:
            root = self.root / arm
            validation = root / "validation"
            validation.mkdir(parents=True)
            cfg = continuation.continuation_settings(
                settings(arm), f"/screen/{arm}", 30
            )
            write_json(root / "resolved_settings.json", cfg)
            write_json(root / "complete.json", complete(30))
            saved_root, self.root = self.root, root
            self.timing(cfg)
            self.root = saved_root
            checkpoint_path = root / "best.ckpt"
            checkpoint_path.write_bytes(b"parent best weights")
            selected = {
                "path": str(checkpoint_path),
                "sha256": sha256(checkpoint_path),
                "epoch": 8,
                "monitor": "val_loss",
                "score": 1.66,
            }
            write_json(root / "selected_checkpoint.json", selected)
            write_json(
                root / "provenance.json",
                {
                    "source_files": [
                        {"path": "src/model.py", "sha256": "same"}
                    ],
                    "harness_files": [
                        {"path": "comparison/run.py", "sha256": "same"}
                    ],
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
            with (validation / "predictions.csv").open(
                "w", newline=""
            ) as handle:
                fields = [
                    "source_database",
                    "event_no",
                    "input_sha256",
                    *TRUTH,
                    *[f"position_{axis}_pred" for axis in "xyz"],
                    *[f"dir_{axis}_pred" for axis in "xyz"],
                    "direction_kappa",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for event in (1, 2):
                    writer.writerow(
                        {
                            **{name: 0 for name in fields},
                            "source_database": "nue.db",
                            "event_no": event,
                            "input_sha256": f"input-{event}",
                            "energy": 20,
                            "pid": 14 if event == 1 else 12,
                            "interaction_type": 1,
                            "dir_z_pred": 1,
                            "direction_kappa": 0,
                        }
                    )
            write_json(
                validation / "prediction_manifest.json",
                {
                    "split": "validation",
                    "rows": 2,
                    "precision": "32-true",
                    "evaluation_seed": 42,
                    "checkpoint": selected,
                    "predictions_sha256": sha256(
                        validation / "predictions.csv"
                    ),
                },
            )

    def summary(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return continuation.summarize(self.root)

    def mutate(self, relative, update):
        path = self.root / relative
        value = continuation.read_json(path)
        update(value)
        write_json(path, value)

    def test_paired_summary_validation_only_and_fp32_loss_diagnostics(self):
        receipt = self.summary()
        self.assertFalse(receipt["test_used"])
        self.assertFalse(receipt["automatic_extension"])
        self.assertEqual(receipt["total_epochs"], 30)
        self.assertAlmostEqual(
            receipt["results"][0]["fp32_prediction_joint_loss"],
            math.log(4 * math.pi),
        )
        self.assertIn("angle_deg_p99", receipt["results"][0])
        self.assertTrue((self.root / "validation_diagnostics.csv").is_file())

    def test_exported_loss_matches_vmf_closed_form_and_position_units(self):
        import pandas as pd

        frame = pd.read_csv(self.root / "baseline/validation/predictions.csv")
        frame["position_x_pred"] = 100.0
        # Direction is exactly correct; the vertex is 100 m away. The vertex
        # term must therefore be 2.6 after the model's /100 unit conversion.
        for kappa in (1e-8, 1.0, 699.0, 700.0, 10000.0):
            frame["direction_kappa"] = kappa
            rows = continuation.prediction_diagnostics(frame, 2.6)
            overall = next(
                row
                for row in rows
                if row["topology"] == "all" and row["energy_min_GeV"] is None
            )
            if kappa < 1e-4:
                expected_direction = (
                    math.log(4 * math.pi) - kappa + kappa**2 / 6
                )
            else:
                expected_direction = (
                    math.log(2 * math.pi)
                    - math.log(kappa)
                    + math.log(-math.expm1(-2 * kappa))
                )
            if kappa == 10000.0:
                # Frozen tight approximation anchored at kappa=700; this
                # differs slightly from the exact high-kappa density.
                expected_direction = -7.373788829807381
            with self.subTest(kappa=kappa):
                self.assertAlmostEqual(overall["weighted_position_loss"], 2.6)
                self.assertAlmostEqual(
                    overall["direction_loss"], expected_direction, delta=1e-8
                )

    def test_rejects_prediction_tampering_and_changed_runtime(self):
        self.mutate(
            "both/provenance.json",
            lambda value: value.update(torch="different"),
        )
        with self.assertRaisesRegex(ValueError, "Paired arms differ"):
            self.summary()
        self.mutate(
            "both/provenance.json", lambda value: value.update(torch="same")
        )
        with (self.root / "both/validation/predictions.csv").open(
            "a"
        ) as handle:
            handle.write("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.summary()

    def test_rejects_different_validation_inputs_even_with_updated_export_hash(
        self,
    ):
        path = self.root / "both/validation/predictions.csv"
        path.write_text(path.read_text().replace("input-1", "changed"))
        self.mutate(
            "both/validation/prediction_manifest.json",
            lambda value: value.update(predictions_sha256=sha256(path)),
        )
        with self.assertRaisesRegex(ValueError, "retained inputs differ"):
            self.summary()


if __name__ == "__main__":
    unittest.main()
