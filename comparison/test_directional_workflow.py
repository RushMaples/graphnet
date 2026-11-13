"""Protocol and original-vMF parity checks (full parity requires GraphNeT)."""

import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directional_run import COUNTS, make_model, settings


class ProtocolTest(unittest.TestCase):
    def test_frozen_contract_survives_model_family_change(self):
        variants = [settings(arm) for arm in ("vmf", "king", "flow")]
        original = copy.deepcopy(variants[0])
        original.pop("arm")
        for cfg in variants[1:]:
            cfg = copy.deepcopy(cfg)
            cfg.pop("arm")
            self.assertEqual(cfg, original)
        self.assertEqual(sum(x["count"] for k,x in original["selections"].items()
                             if k.endswith("_train.csv")), COUNTS["train"])
        self.assertEqual(original["epochs"], 10)
        self.assertFalse(original["drop_last"])
        self.assertFalse(original["test_used"])

    def test_invalid_model_or_depth_rejected(self):
        for arm,depth in (("unknown",4),("flow",3)):
            with self.assertRaises(ValueError):
                settings(arm, depth)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires Torch")
class AMPUpdateTest(unittest.TestCase):
    def test_optimizer_hook_counts_updates_but_allows_scaler_recovery(self):
        import torch
        if not hasattr(torch.amp, "GradScaler"):
            self.skipTest("This Torch lacks the CPU GradScaler test API")
        try:
            scaler = torch.amp.GradScaler("cpu", init_scale=4)
        except (TypeError, AttributeError):
            self.skipTest("This Torch lacks the CPU GradScaler test API")
        fake_pl = types.ModuleType("pytorch_lightning")
        fake_pl.LightningModule = torch.nn.Module
        fake_pl.seed_everything = lambda seed, **kw: torch.manual_seed(seed)
        fake_augmentation = types.ModuleType("augmentation")
        def original(cfg):
            return types.SimpleNamespace(backbone=torch.nn.Linear(4, 4),
                _tasks=[torch.nn.Linear(4, 3), torch.nn.Linear(4, 4)]), {}
        fake_augmentation.model_and_data = original
        with patch.dict(sys.modules, {"pytorch_lightning": fake_pl, "augmentation": fake_augmentation}):
            model, _, _ = make_model(settings("vmf"))
        optimizer = model.configure_optimizers()["optimizer"]
        model.trainer = types.SimpleNamespace(precision_plugin=types.SimpleNamespace(scaler=scaler))
        before = model.backbone.weight.detach().clone()
        loss = model.backbone.weight.square().sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        model.on_before_optimizer_step(optimizer)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        self.assertEqual(model.successful_optimizer_steps, 1)
        self.assertFalse(torch.equal(before, model.backbone.weight))
        scaler.scale(model.backbone.weight.square().sum()).backward()
        model.backbone.weight.grad[0, 0] = float("inf")
        scaler.unscale_(optimizer)
        model.on_before_optimizer_step(optimizer)
        scaler.step(optimizer); scaler.update()
        self.assertEqual(model.successful_optimizer_steps, 1)
        self.assertEqual(model.overflow_batches, 1)


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None
                         for name in ("torch", "torch_geometric", "pytorch_lightning", "graphnet")),
                     "Requires the full GraphNeT training environment")
class OriginalVMFParityTest(unittest.TestCase):
    def test_wrapped_joint_loss_and_initial_shared_weights(self):
        import torch
        import pytorch_lightning as pl
        from torch_geometric.data import Data
        from augmentation import model_and_data
        from directional_run import tensor_hash
        cfg = settings("vmf")
        wrapped, _, initial = make_model(cfg)
        pl.seed_everything(42, workers=True)
        original, _ = model_and_data(cfg)
        self.assertEqual(initial["backbone"], tensor_hash(original.backbone.state_dict()))
        self.assertEqual(initial["vertex"], tensor_hash(original._tasks[0].state_dict()))
        batch = Data(x=torch.randn(8,6)*.01,
                     batch=torch.tensor([0]*4+[1]*4),
                     n_pulses=torch.tensor([32,64]),
                     direction=torch.tensor([[1.,0.,0.],[0.,0.,1.]]),
                     position_x=torch.tensor([10.,20.]), position_y=torch.tensor([30.,40.]),
                     position_z=torch.tensor([-200.,-100.]))
        for name in ("azimuth", "zenith", "energy", "pid", "interaction_type", "oneweight"):
            batch[name] = torch.ones(2)
        original.eval(); wrapped.eval()
        with torch.no_grad():
            expected = original.shared_step([batch], 0)
            actual = wrapped.components(batch)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
