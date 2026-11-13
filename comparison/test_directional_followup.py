"""Focused task/count, stopping, continuation and comparison regression tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directional_followup import (TASK_NAMES, advance_stopping, make_callbacks,
                                 seal_last_checkpoint, settings, stopping_state)
from directional_run import make_model, settings as original_settings


class ProtocolTests(unittest.TestCase):
    def test_agreed_matrix_and_preserved_frozen_settings(self):
        self.assertEqual(len(TASK_NAMES), 5)
        for name in TASK_NAMES:
            cfg = settings(name)
            self.assertEqual(cfg["epochs"], 50)
            self.assertEqual(cfg["seed"], 42)
            self.assertFalse(cfg["king_parent_count"])
            self.assertFalse(cfg["test_used"])
            self.assertFalse(cfg["additional_training_seeds"])
            self.assertFalse(cfg["sampling_convergence_study"])
            self.assertEqual(cfg["flow_layers"], 4)
            self.assertEqual(cfg["posterior_samples"], 128)
            self.assertEqual(cfg["selections"], original_settings(cfg["arm"])["selections"])
            self.assertEqual(cfg["starting_epochs"], 10 if name in ("joint_vmf", "joint_flow") else 0)

    def test_strict_delta_patience_and_nonfinite(self):
        self.assertEqual(advance_stopping(1.0, None, 0), (1., 0))
        self.assertEqual(advance_stopping(.9999, 1., 2), (1., 3))
        self.assertEqual(advance_stopping(.9998, 1., 2), (.9998, 0))
        self.assertEqual(stopping_state([{"x": 1.}, {"x": 1.1}], "x"), {"best_score": 1., "wait_count": 1})
        with self.assertRaises(FloatingPointError):
            advance_stopping(float("nan"), 1., 0)


def fake_original(cfg):
    import torch
    from directional_heads import unit_vector, vmf_log_prob

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping = torch.nn.Linear(4, 384)

        def forward(self, batch):
            value = self.mapping(batch.x)
            return value, value

    class Position(torch.nn.Linear):
        def __init__(self):
            super().__init__(384, 3)

        def compute_loss(self, prediction, truth):
            target = torch.stack([truth["position_"+a] for a in "xyz"], -1)/100
            return 2.6*torch.linalg.vector_norm(prediction-target, dim=-1).mean()

    class Direction(torch.nn.Linear):
        def __init__(self):
            super().__init__(384, 4)

        def forward(self, x):
            raw = super().forward(x)
            return torch.cat((unit_vector(raw[:, :3]), torch.nn.functional.softplus(raw[:, 3:4])), -1)

        def compute_loss(self, prediction, truth):
            return -vmf_log_prob(prediction[:, :3], prediction[:, 3], truth["direction"]).mean()

    return types.SimpleNamespace(backbone=Backbone(), _tasks=[Position(), Direction()]), {}


def batch():
    import torch

    class Batch(dict):
        __getattr__ = dict.__getitem__
    result = Batch(x=torch.randn(2, 4), num_graphs=2, n_pulses=torch.tensor([16, 512]),
                   direction=torch.tensor([[1., 0., 0.], [0., 0., 1.]]))
    for name in ("position_x", "position_y", "position_z", "azimuth", "zenith", "energy", "pid", "interaction_type", "oneweight"):
        result[name] = torch.ones(2)
    return result


class ModelTests(unittest.TestCase):
    def build(self, cfg):
        fake = types.ModuleType("augmentation")
        fake.model_and_data = fake_original
        with patch.dict(sys.modules, {"augmentation": fake}):
            return make_model(cfg)

    def test_count_ablation_changes_only_conditioning(self):
        import torch
        old, _, old_initial = self.build(original_settings("king"))
        new, _, new_initial = self.build(settings("joint_king_no_count"))
        self.assertEqual(old_initial, new_initial)
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        b = batch()
        original_counts = b.n_pulses.clone()
        _, first = new(b)
        b.n_pulses = torch.tensor([4096, 8])
        _, second = new(b)
        for key in ("direction", "alpha", "beta"):
            torch.testing.assert_close(getattr(first, key), getattr(second, key), rtol=0, atol=0)
        _, enabled = old(b)
        self.assertFalse(torch.allclose(enabled.alpha, first.alpha))
        # The ablation must not overwrite the actual event count in its input.
        torch.testing.assert_close(b.n_pulses, torch.tensor([4096, 8]))

    def test_direction_only_removes_vertex_gradient_and_preserves_initialization(self):
        import torch
        joint, _, original = self.build(original_settings("vmf"))
        direction, _, changed = self.build(settings("direction_vmf"))
        self.assertEqual(original, changed)
        b = batch()
        j = joint.components(b)
        d = direction.components(b)
        torch.testing.assert_close(d[0], j[2])
        self.assertEqual(float(d[1]), 0.)
        self.assertIsNone(d[4])
        self.assertIsNone(direction(b)[0])
        d[0].backward()
        self.assertTrue(any(p.grad is not None for p in direction.backbone.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in direction.position_task.parameters()))
        self.assertTrue(all(p.requires_grad for p in joint.position_task.parameters()))


class ContinuationTests(unittest.TestCase):
    def test_lightning_restores_optimizer_scheduler_loops_and_keeps_parent_best(self):
        import torch
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import Callback, ModelCheckpoint
        from torch.utils.data import DataLoader, TensorDataset
        from contract import sha256

        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def training_step(self, b, i):
                return self.weight.square()

            def validation_step(self, b, i):
                score = 10.-self.current_epoch if self.current_epoch < 10 else 2.
                self.log("val_loss", score, batch_size=1)
                self.log("val_direction_nll", score, batch_size=1)

            def configure_optimizers(self):
                opt = torch.optim.AdamW(self.parameters(), lr=.01)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6)
                return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

        class Monitor(Callback):
            def __init__(self, *args):
                self.completed_epochs = 0
            def on_train_epoch_end(self, trainer, model):
                self.completed_epochs += 1

        class Augment(Callback):
            def __init__(self, *args):
                pass

        runtime = types.ModuleType("runtime")
        runtime.RunMonitor = Monitor
        augmentation = types.ModuleType("augmentation_callbacks")
        augmentation.AugmentationCallback = Augment
        loader = DataLoader(TensorDataset(torch.ones(1)), batch_size=1)
        opts = dict(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                    enable_model_summary=False, num_sanity_val_steps=0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = ModelCheckpoint(dirpath=str(root / "parent"), monitor="val_loss", save_last=True,
                                         filename="best-{epoch:02d}-{val_loss:.4f}")
            first = pl.Trainer(max_epochs=10, callbacks=[checkpoint], **opts)
            first.fit(Tiny(), loader, loader)
            old_hash = sha256(checkpoint.best_model_path)
            history = [{"epoch": e, "val_loss": 10.-e, "val_direction_nll": 10.-e} for e in range(10)]
            parent = {"history": history, "best_path": checkpoint.best_model_path,
                      "best_direction_score": 1., "selected": {"epoch": 9, "score": 1.}}
            with patch.dict(sys.modules, {"runtime": runtime, "augmentation_callbacks": augmentation}):
                callbacks, monitor, snapshots, kept, stopping = make_callbacks(settings("joint_vmf"), root / "child", parent)
            second = pl.Trainer(max_epochs=12, callbacks=callbacks, **opts)
            second.fit(Tiny(), loader, loader, ckpt_path=str(root / "parent" / "last.ckpt"))
            self.assertEqual(monitor.completed_epochs, 2)
            self.assertEqual(second.global_step, 12)
            self.assertEqual(stopping.wait_count, 2)
            self.assertEqual(sha256(checkpoint.best_model_path), old_hash)
            self.assertEqual(float(kept["objective"].best_model_score), 1.)
            self.assertIn("parent-epoch=09", kept["objective"].best_model_path)
            snapshot = json.loads((root / "child/snapshots/epoch10/selection.json").read_text())
            self.assertEqual(snapshot["budget_epochs"], 10)
            self.assertEqual(snapshot["checkpoints"]["direction"]["epoch"], 9)
            seal_last_checkpoint(second, root / "child/checkpoints/last.ckpt", 12)
            final = torch.load(root / "child/checkpoints/last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(final["epoch"], 11)
            self.assertEqual(final["lr_schedulers"][0]["last_epoch"], 12)
            self.assertEqual(float(next(iter(final["optimizer_states"][0]["state"].values()))["step"]), 12.)
            third = pl.Trainer(max_epochs=13, enable_checkpointing=False, **opts)
            third.fit(Tiny(), loader, loader, ckpt_path=str(root / "child/checkpoints/last.ckpt"))
            self.assertEqual(third.global_step, 13)
            self.assertEqual(third.lr_scheduler_configs[0].scheduler.last_epoch, 13)


class ComparisonTests(unittest.TestCase):
    def test_complete_five_run_comparison_and_incomplete_run_rejected(self):
        import numpy as np
        import pandas as pd
        from contract import sha256, write_json
        from directional_followup import COUNTS
        from compare_directional_followup import compare
        env = {k: "fixture" for k in ("python", "torch", "cuda_runtime", "lightning", "gpu", "pip_freeze")}
        initial, databases = {"backbone": "same", "vertex": "same"}, [{"database": "P14", "sha256": "same"}]
        frame = pd.DataFrame({"source_database": ["P14"]*3, "event_no": [1, 2, 3],
            "input_sha256": ["a", "b", "c"], "n_pulses": [10, 20, 30],
            "position_x": [0.]*3, "position_y": [0.]*3, "position_z": [0.]*3,
            "azimuth": [0.]*3, "zenith": [np.pi/2]*3, "energy": [10., 20., 30.],
            "pid": [14, 12, 16], "interaction_type": [1]*3, "oneweight": [1.]*3,
            "direction_x": [1.]*3, "direction_y": [0.]*3, "direction_z": [0.]*3,
            "dir_x_pred": [1.]*3, "dir_y_pred": [0.]*3, "dir_z_pred": [0.]*3,
            "position_x_pred": [1.]*3, "position_y_pred": [1.]*3, "position_z_pred": [1.]*3,
            "direction_nll": [1., 2., 3.], "angle_deg": [0.]*3, "vertex_error_m": [3**.5]*3,
            "hpd_mass_at_truth": [.4, .6, .8]})
        with tempfile.TemporaryDirectory() as directory, patch.dict(COUNTS, validation=3):
            root = Path(directory)
            parent = root / "parent"
            plan = {"protocol": "directional-followup-v1", "test_used": False, "source_root": str(root),
                    "source_files": [], "parents": {"king": {"folder": str(parent), "receipts": {}}},
                    "runs": {n: settings(n) for n in TASK_NAMES}, "parent_environment": env,
                    "common_initialization": initial, "parent_databases": {"P14": "same"},
                    "parent_input_probe": [], "continuation_rng_limit": "fixture limit",
                    "historical_epochs": 43, "historical_selected_epoch": 31}
            def exported(folder, data, cfg, provenance):
                (folder / "validation").mkdir(parents=True)
                write_json(folder / "resolved_settings.json", cfg)
                write_json(folder / "provenance.json", provenance)
                write_json(folder / "database_manifest.json", databases)
                selected = {"path": "fixture.ckpt", "sha256": "weights", "epoch": 9, "monitor": "val_loss", "score": 1.}
                write_json(folder / "selected_checkpoint.json", selected)
                data.to_csv(folder / "validation/predictions.csv", index=False)
                manifest = {"complete": True, "rows": 3, "split": "validation", "test_used": False,
                    "density_measure": "per steradian", "sha256": sha256(folder / "validation/predictions.csv"),
                    "selected_checkpoint": selected}
                for kind, name in (("settings", "resolved_settings.json"), ("provenance", "provenance.json"), ("database_manifest", "database_manifest.json")):
                    manifest[kind+"_sha256"] = sha256(folder / name)
                write_json(folder / "validation/prediction_manifest.json", manifest)
                return {"folder": str(folder), "manifest_sha256": sha256(folder / "validation/prediction_manifest.json"),
                        "prediction_sha256": manifest["sha256"], "selected_checkpoint": selected,
                        "criterion": "objective", "budget_epochs": 10}
            old_entry = exported(parent, frame, {}, env)
            plan["parents"]["king"]["selected"] = old_entry["selected_checkpoint"]
            historical = root / "historical"
            (historical / "validation").mkdir(parents=True)
            old = frame.copy()
            old["direction_kappa"] = 1.
            old.to_csv(historical / "validation/predictions.csv", index=False)
            plan["historical_root"] = str(historical)
            plan["historical_predictions_sha256"] = sha256(historical / "validation/predictions.csv")
            plan_path = root / "plan.json"
            write_json(plan_path, plan)
            for name in TASK_NAMES:
                cfg = settings(name)
                folder = root / "runs" / name / "model"
                folder.mkdir(parents=True)
                provenance = dict(env, source_files=[])
                write_json(folder / "resolved_settings.json", cfg)
                write_json(folder / "provenance.json", provenance)
                write_json(folder / "initial_shared_state.json", initial)
                write_json(folder / "input_probe.json", [])
                write_json(folder / "database_manifest.json", databases)
                write_json(folder / "validation_history.json", [])
                (folder / "timing.csv").write_text("epoch\n9\n")
                data = frame.copy()
                if cfg["task"] == "direction":
                    data = data.drop(columns=["vertex_error_m", "position_x_pred", "position_y_pred", "position_z_pred"])
                entry = exported(folder / "exports/final", data, cfg, provenance)
                tags = ("epoch10/objective", "epoch10/direction", "final/objective", "final/direction") if name in ("joint_vmf", "joint_flow") else (
                    ("epoch10/objective", "final/objective") if cfg["task"] == "joint" else ("epoch10/direction", "final/direction"))
                complete = {"run_name": name, "protocol": plan["protocol"], "plan_sha256": sha256(plan_path),
                    "settings_sha256": sha256(folder / "resolved_settings.json"),
                    "provenance_sha256": sha256(folder / "provenance.json"), "epochs": 10,
                    "test_exported": False, "test_rows": 0, "validation_exported": True,
                    "exports": {tag: entry for tag in tags}}
                write_json(folder / "complete.json", complete)
            compare(plan_path, root)
            self.assertTrue((root / "summary/results.html").is_file())
            self.assertTrue((root / "directional-followup-results.tar.gz").is_file())
            result = pd.read_csv(root / "summary/headline.csv")
            self.assertEqual(len(result[result.question == "count_epoch10"]), 2)
            self.assertEqual(len(result[result.question == "joint_heads_final"]), 3)
            self.assertEqual(len(result[result.question == "vertex_final"]), 4)
            # A missing arm must not be turned into a successful partial table.
            (root / "runs/direction_flow/model/complete.json").unlink()
            with self.assertRaises(FileNotFoundError):
                compare(plan_path, root / "incomplete")

    def test_mismatched_actual_inputs_rejected(self):
        import pandas as pd
        from compare_directional_followup import require_same_inputs
        fields = ["input_sha256", "n_pulses", "position_x", "position_y", "position_z",
                  "azimuth", "zenith", "energy", "pid", "interaction_type", "oneweight"]
        a = pd.DataFrame({k: ["abc" if k == "input_sha256" else 1] for k in fields})
        b = a.copy()
        b.loc[0, "input_sha256"] = "changed"
        with self.assertRaises(ValueError):
            require_same_inputs(a, b)

    def test_direction_only_summary_has_no_vertex_result(self):
        import pandas as pd
        from compare_directional_followup import summarize
        f = pd.DataFrame({"pid": [14, 12], "interaction_type": [1, 1],
                          "direction_nll": [1., 2.], "angle_deg": [10., 20.], "hpd_mass_at_truth": [.5, .8]})
        rows = summarize(f, model="direction")
        self.assertTrue(all("median_vertex_m" not in row for row in rows))


if __name__ == "__main__":
    unittest.main()
