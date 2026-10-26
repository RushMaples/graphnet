"""CPU protocol and recorder tests for the early-stopped full continuation."""

import ast
import contextlib
import copy
import csv
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import full_augmentation as full
from contract import sha256, write_json
from full_diagnostics import (
    cone_regions,
    diagnostics,
    preflight_diagnostics,
    validation_callback,
)
from test_continuation import checkpoint, complete, settings


class StoppingTests(unittest.TestCase):
    def test_absolute_delta_is_strict_and_small_gains_accumulate(self):
        policy = full.stopping_policy(3, 0.125)
        initial = full.replay_stopping([{"val_loss": 2}], policy)
        state = full.replay_stopping(
            [
                {"val_loss": 1.875},
                {"val_loss": 1.9},
                {"val_loss": 1.874},
            ],
            policy,
            initial,
        )
        self.assertEqual(initial["best_score"], 2)
        self.assertEqual(state["best_score"], 1.874)
        self.assertEqual(state["wait_count"], 0)

    def test_carried_patience_stops_after_eleven_additional_bad_epochs(self):
        policy = full.stopping_policy()
        initial = full.replay_stopping(
            [
                {"val_loss": 1.6},
                {"val_loss": 1.7},
            ],
            policy,
        )
        final = full.replay_stopping([{"val_loss": 1.8}] * 11, policy, initial)
        final["stopped_epoch"] = 40
        self.assertEqual(final["wait_count"], 12)
        self.assertEqual(
            full.completion_reason(30, 80, 41, final, policy), "early_stopping"
        )
        final["stopped_epoch"] = 39
        with self.assertRaises(RuntimeError):
            full.completion_reason(30, 80, 41, final, policy)

    def test_maximum_and_interrupted_or_empty_runs(self):
        policy = full.stopping_policy()
        final = {"best_score": 1.6, "wait_count": 1, "stopped_epoch": 0}
        self.assertEqual(
            full.completion_reason(30, 80, 80, final, policy), "max_epochs"
        )
        for actual in (30, 31, 79, 81):
            with self.subTest(actual=actual), self.assertRaises(RuntimeError):
                full.completion_reason(30, 80, actual, final, policy)
        for value in (float("inf"), float("nan")):
            with self.assertRaises(FloatingPointError):
                full.replay_stopping([{"val_loss": value}], policy)

    def test_invalid_policy(self):
        for patience, delta in (
            (0, 0.1),
            (True, 0.1),
            (12, -1),
            (12, float("nan")),
        ):
            with self.assertRaises(ValueError):
                full.stopping_policy(patience, delta)

    def test_real_callback_protocol_uses_seed_and_persists_native_state(self):
        class EarlyStopping:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.stopped_epoch = 0

        callbacks = types.SimpleNamespace(EarlyStopping=EarlyStopping)
        torch = types.SimpleNamespace(
            tensor=lambda value, **kw: value, float64="float64"
        )
        with patch.dict(
            sys.modules,
            {"torch": torch, "pytorch_lightning.callbacks": callbacks},
        ):
            callback = full.early_stopping_callback(
                {"best_score": 2, "wait_count": 11}, full.stopping_policy()
            )
            stop, _ = callback._evaluate_stopping_criteria(2)
        self.assertIsInstance(callback, EarlyStopping)
        self.assertTrue(stop)
        self.assertEqual(callback.wait_count, 12)
        self.assertTrue(callback.check_on_train_epoch_end)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def timing(self, root, start, end):
        root.mkdir(parents=True, exist_ok=True)
        with (root / "timing.csv").open("w", newline="") as handle:
            fields = [
                "epoch",
                "train_events",
                "validation_events",
                "optimizer_steps",
                "val_loss",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(start, end):
                writer.writerow(
                    dict(
                        epoch=epoch,
                        train_events=578121,
                        validation_events=72264,
                        optimizer_steps=4517,
                        val_loss=2 - epoch / 100,
                    )
                )

    def history(self):
        screen = self.root / "screen-1" / "both"
        parent = self.root / "continue-2" / "both"
        self.timing(screen, 0, 10)
        self.timing(parent, 10, 30)
        cfg = settings("both")
        write_json(screen / "resolved_settings.json", cfg)
        write_json(screen / "complete.json", complete())
        hashes = {
            name: sha256(screen / name)
            for name in (
                "resolved_settings.json",
                "complete.json",
                "timing.csv",
            )
        }
        cfg = full.continuation_settings(cfg, "/missing/screen-1/both", 30)
        write_json(parent / "resolved_settings.json", cfg)
        write_json(parent / "complete.json", complete(30))
        write_json(
            parent / "checkpoint_source.json",
            {
                "parent_run": cfg["parent_run"],
                "starting_epochs": 10,
                "parent_artifact_hashes": hashes,
            },
        )
        return screen, parent

    def test_history_relocation_hashes_and_complete_sequence(self):
        screen, parent = self.history()
        rows, ancestry = full.parent_history(parent)
        self.assertEqual([int(row["epoch"]) for row in rows], list(range(30)))
        self.assertEqual(len(ancestry), 2)
        with (screen / "timing.csv").open("a") as handle:
            handle.write("tamper")
        with self.assertRaisesRegex(ValueError, "Ancestor artifact changed"):
            full.parent_history(parent)

    def test_history_requires_ancestry_and_correct_exposure(self):
        screen, parent = self.history()
        (screen / "timing.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            full.parent_history(parent)
        self.timing(parent, 11, 30)
        with self.assertRaisesRegex(ValueError, "horizon"):
            full.parent_history(parent)

    def test_no_overwrite_or_nested_outputs(self):
        parent = self.root / "parent"
        parent.mkdir()
        for output in (parent, parent / "nested", self.root):
            with self.assertRaises(ValueError):
                full.validate_output(output, parent)
        output = self.root / "output"
        output.mkdir()
        (output / "receipt").write_text("keep")
        with self.assertRaises(FileExistsError):
            full.validate_output(output, parent)

    def test_inherited_checkpoint_and_source_guards_are_used_directly(self):
        import continue_augmentation as continuation

        self.assertIs(
            full.validate_checkpoint, continuation.validate_checkpoint
        )
        self.assertIs(
            full.verify_original_files, continuation.verify_original_files
        )
        self.assertIs(
            full.verify_data_identity, continuation.verify_data_identity
        )
        full.validate_checkpoint(checkpoint(30), 30)
        parent = {"monitor": "val_loss", "score": 1.6}
        self.assertIs(
            full.choose_checkpoint(parent, dict(parent, score=1.7)), parent
        )
        with self.assertRaises(ValueError):
            full.validate_checkpoint(checkpoint(30), 31)

    def test_resume_checkpoint_requires_matching_completed_parent_hash(self):
        parent = self.root / "parent"
        destination = parent / "checkpoints" / "last.ckpt"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(
            b"the completed both arm full-state checkpoint"
        )
        cfg = full.continuation_settings(settings("both"), "/screen/both", 30)
        receipt = dict(
            complete(30),
            last_checkpoint={
                "path": "/original/both/checkpoints/last.ckpt",
                "sha256": sha256(destination),
            },
        )
        self.assertEqual(
            full.resume_checkpoint_path(parent, cfg, receipt),
            destination.resolve(),
        )
        destination.write_bytes(b"different arm, same epoch and step counters")
        with self.assertRaisesRegex(ValueError, "changed from its receipt"):
            full.resume_checkpoint_path(parent, cfg, receipt)
        with self.assertRaisesRegex(ValueError, "receipt is required"):
            full.resume_checkpoint_path(parent, cfg, complete(30))
        # Original screen artifacts have no last-file receipt.
        self.assertEqual(
            full.resume_checkpoint_path(parent, settings("both"), complete()),
            destination.resolve(),
        )

    def test_early_completion_replays_curve_and_rejects_partial_receipts(self):
        _, parent = self.history()
        history, _ = full.parent_history(parent)
        output = self.root / "full" / "both"
        self.timing(output, 30, 31)
        cfg = full.continuation_settings(settings("both"), parent, 80)
        cfg.update(
            stage="full-continuation",
            starting_epochs=30,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0001,
        )
        policy = full.stopping_policy(1)
        initial = full.replay_stopping(history, policy)
        # The newly completed epoch is worse than the parent's final loss.
        path = output / "timing.csv"
        path.write_text(path.read_text().replace("1.7", "1.8"))
        current = full.timing_rows(output, 30, 31)
        final = full.replay_stopping(current, policy, initial)
        final["stopped_epoch"] = 30
        full.write_curve(output, history, current)
        write_json(
            output / "early_stopping.json",
            {
                "policy": policy,
                "initial": initial,
                "final": final,
                "stop_reason": "early_stopping",
            },
        )
        receipt = dict(
            complete(31),
            complete=True,
            actual_completed_epochs=31,
            max_total_epochs=80,
            starting_epochs=30,
            additional_epochs=1,
            parent_run=cfg["parent_run"],
            stop_reason="early_stopping",
            early_stopping_initial=initial,
            early_stopping_final=final,
            artifact_hashes={},
            training_curve_sha256=sha256(output / "training_curve.csv"),
        )
        self.assertEqual(
            full.validate_completion(output, cfg, receipt),
            (31, "early_stopping"),
        )
        for update in (
            {"complete": False},
            {"epochs": 80},
            {"additional_epochs": 2},
            {"max_total_epochs": 100},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                full.validate_completion(output, cfg, dict(receipt, **update))
        receipt = copy.deepcopy(receipt)
        receipt["early_stopping_final"]["wait_count"] = 2
        with self.assertRaisesRegex(ValueError, "Stopping state"):
            full.validate_completion(output, cfg, receipt)

    def test_train_ast_excludes_test_and_explicitly_reloads_selected_weights(
        self,
    ):
        tree = ast.parse(Path(full.__file__).read_text())
        train = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "train"
        )
        subscripts = [
            n for n in ast.walk(train) if isinstance(n, ast.Subscript)
        ]
        for node in subscripts:
            if isinstance(node.value, ast.Name) and node.value.id == "loaders":
                self.assertNotIn("test", ast.dump(node.slice))
        source = ast.get_source_segment(Path(full.__file__).read_text(), train)
        self.assertIn('if key != "test"', source)
        self.assertIn(
            'model.load_state_dict(state["state_dict"], strict=True)', source
        )
        self.assertIn("ckpt_path=str(last_parent_path)", source)
        self.assertIn('"test_exported": False', source)


class DiagnosticsTests(unittest.TestCase):
    def overall(self, frame):
        return next(
            row
            for row in diagnostics(frame, 2.6)
            if row["topology"] == "all" and row["energy_min_GeV"] is None
        )

    def test_fp16_epsilon_direction_normalization_is_valid_but_zero_is_not(
        self,
    ):
        import numpy as np

        frame = self.frame()
        raw = np.float16(0.1)
        norm = raw / (raw + np.finfo(np.float16).eps)
        frame[[f"dir_{axis}_pred" for axis in "xyz"]] *= float(norm)
        overall = self.overall(frame)
        self.assertLess(overall["direction_norm_min"], 0.995)
        self.assertTrue(math.isfinite(overall["fp32_prediction_joint_loss"]))
        frame.loc[0, [f"dir_{axis}_pred" for axis in "xyz"]] = 0
        with self.assertRaisesRegex(ValueError, "finite nonzero norm"):
            diagnostics(frame, 2.6)

    def test_effective_concentration_preserves_raw_natural_vector_loss(self):
        frame = self.frame()
        frame.direction_kappa = 10
        frame[[f"dir_{axis}_pred" for axis in "xyz"]] *= 0.5
        original = frame.copy()
        overall = self.overall(frame)
        reference = self.frame()
        reference.direction_kappa = 5
        expected = self.overall(reference)
        self.assertTrue(frame.equals(original))
        self.assertAlmostEqual(overall["kappa_median"], 10)
        self.assertAlmostEqual(overall["effective_kappa_median"], 5)
        self.assertAlmostEqual(overall["direction_norm_median"], 0.5)
        self.assertAlmostEqual(
            overall["direction_loss"], expected["direction_loss"]
        )
        self.assertAlmostEqual(
            overall["fp32_prediction_joint_loss"],
            expected["fp32_prediction_joint_loss"],
        )
        for probability in (0.68, 0.9):
            self.assertEqual(
                overall[f"effective_coverage_{probability}"],
                expected[f"coverage_{probability}"],
            )
            self.assertAlmostEqual(
                overall[f"effective_cone_radius_{probability}_deg_median"],
                expected[f"cone_radius_{probability}_deg_median"],
            )
            self.assertGreater(
                overall[f"effective_cone_radius_{probability}_deg_median"],
                overall[f"cone_radius_{probability}_deg_median"],
            )
        reference[[f"dir_{axis}_pred" for axis in "xyz"]] *= 1 - 1e-6
        nearly_unit = self.overall(reference)
        self.assertAlmostEqual(
            nearly_unit["effective_cone_radius_0.68_deg_median"],
            nearly_unit["cone_radius_0.68_deg_median"],
            places=4,
        )

    def test_diagnostic_preflight_uses_one_extra_eval_forward_on_probe(self):
        import numpy as np
        from unittest.mock import Mock

        batch = Mock(num_graphs=4)
        batch.to.return_value = batch
        probe = Mock()
        probe.shared_step.return_value = 1.6
        callback = Mock()
        torch = types.SimpleNamespace(
            float16="float16",
            no_grad=contextlib.nullcontext,
            autocast=lambda **kwargs: contextlib.nullcontext(),
            isfinite=np.isfinite,
        )
        with patch.dict(sys.modules, {"torch": torch}), patch(
            "full_diagnostics.validation_callback", return_value=callback
        ) as factory, tempfile.TemporaryDirectory() as output:
            receipt = preflight_diagnostics(
                probe, [batch], output, {"precision": "16-mixed"}
            )
        probe.eval.assert_called_once()
        probe.shared_step.assert_called_once_with([batch], 0)
        self.assertEqual(factory.call_args.args[2], 4)
        callback.on_validation_batch_end.assert_called_once()
        callback.on_validation_epoch_end.assert_called_once()
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["extra_forward_passes"], 1)
        self.assertTrue(receipt["separate_model"])

    def test_exact_vmf_cone_regions_uniform_and_large_concentration(self):
        import numpy as np

        radius68, solid68 = cone_regions([0, 1e-8, 1, 100, 1e6, 1e100], 0.68)
        radius90, solid90 = cone_regions([0, 1e-8, 1, 100, 1e6, 1e100], 0.9)
        self.assertAlmostEqual(
            radius68[0], math.degrees(math.acos(1 - 2 * 0.68))
        )
        self.assertAlmostEqual(solid68[0], 4 * math.pi * 0.68)
        self.assertEqual(radius68[0], radius68[1])
        self.assertTrue(np.isfinite(radius68).all())
        self.assertTrue(np.isfinite(solid90).all())
        self.assertTrue((radius68 > 0).all())
        self.assertTrue((np.diff(radius68) <= 0).all())
        self.assertTrue((radius90 > radius68).all())
        self.assertTrue((solid90 > solid68).all())
        # Invert the original vMF CDF at a moderate concentration.
        cosine = math.cos(math.radians(radius68[3]))
        recovered = math.expm1(100 * (cosine - 1)) / math.expm1(-200)
        self.assertAlmostEqual(recovered, 0.68)
        for kappa in ([-1], [float("inf")], [float("nan")]):
            with self.assertRaises(ValueError):
                cone_regions(kappa, 0.68)

    def frame(self):
        import numpy as np
        import pandas as pd

        truth = np.radians([359, 1, 90, 180])
        prediction = np.radians([1, 359, 90, 180])
        rows = {
            "azimuth": truth,
            "zenith": np.repeat(math.pi / 2, 4),
            "energy": [20] * 4,
            "pid": [14, 14, 12, 12],
            "interaction_type": [1] * 4,
            "direction_kappa": [0] * 4,
            "dir_x_pred": np.cos(prediction),
            "dir_y_pred": np.sin(prediction),
            "dir_z_pred": [0] * 4,
        }
        for axis in "xyz":
            rows[f"position_{axis}"] = [0] * 4
            rows[f"position_{axis}_pred"] = (
                [100, 0, 0, 0] if axis == "x" else [0] * 4
            )
        return pd.DataFrame(rows)

    def test_azimuth_wrap_tails_topology_coverage_and_loss_units(self):
        rows = diagnostics(self.frame(), 2.6, azimuth_bins=True)
        all_events = next(
            r
            for r in rows
            if r["topology"] == "all"
            and r["energy_min_GeV"] is None
            and r["azimuth_min_deg"] is None
        )
        self.assertAlmostEqual(all_events["azimuth_abs_deg_p99"], 2)
        self.assertAlmostEqual(all_events["weighted_position_loss"], 0.65)
        self.assertEqual(all_events["coverage_0.99"], 1)
        self.assertAlmostEqual(all_events["coverage_error_0.99"], 0.01)
        self.assertAlmostEqual(
            all_events["cone_radius_0.68_deg_median"],
            math.degrees(math.acos(1 - 2 * 0.68)),
        )
        self.assertAlmostEqual(
            all_events["cone_solid_angle_0.9_sr_median"], 4 * math.pi * 0.9
        )
        self.assertAlmostEqual(all_events["vertex_m_p99"], 97)
        binned = [
            r
            for r in rows
            if r["azimuth_min_deg"] is not None and r["topology"] == "all"
        ]
        self.assertEqual(sum(r["events"] for r in binned), 4)
        self.assertEqual(
            {r["topology"] for r in rows}, {"all", "track", "cascade"}
        )

    def test_recorder_callback_no_extra_forward_and_exact_validation_count(
        self,
    ):
        import numpy as np

        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            def float(self):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.array

            def reshape(self, *shape):
                return Tensor(self.array.reshape(*shape))

        frame = self.frame()
        p = frame[[f"position_{axis}_pred" for axis in "xyz"]].to_numpy() / 100
        d = frame[[f"dir_{axis}_pred" for axis in "xyz"]].to_numpy()
        d = np.column_stack((d, frame.direction_kappa))
        recorders = {
            "position": types.SimpleNamespace(last_prediction=Tensor(p)),
            "direction": types.SimpleNamespace(last_prediction=Tensor(d)),
        }
        physics = types.SimpleNamespace(_recorders=lambda model: recorders)
        with patch.dict(
            sys.modules,
            {
                "pytorch_lightning": types.SimpleNamespace(Callback=object),
                "graphnet.training.callbacks": types.SimpleNamespace(
                    PhysicsMetricsCallback=physics
                ),
            },
        ), tempfile.TemporaryDirectory() as folder:
            callback = validation_callback(
                folder, {"vertex_weight": 2.6, "precision": "16-mixed"}, 4
            )
            trainer = types.SimpleNamespace(current_epoch=30)
            batch = types.SimpleNamespace(
                **{k: Tensor(v) for k, v in frame.items()}
            )
            callback.on_validation_epoch_start(trainer, object())
            callback.on_validation_batch_end(trainer, object(), None, batch, 0)
            callback.on_validation_epoch_end(trainer, object())
            with (Path(folder) / "epoch_diagnostics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({r["epoch"] for r in rows}, {"30"})
            self.assertNotIn("fp32_prediction_joint_loss", rows[0])
            self.assertIn("recorder_prediction_joint_loss", rows[0])
            callback.on_validation_epoch_start(trainer, object())
            callback.on_validation_batch_end(trainer, object(), None, batch, 0)
            callback.on_validation_batch_end(trainer, object(), None, batch, 1)
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                callback.on_validation_epoch_end(trainer, object())


if __name__ == "__main__":
    unittest.main()
