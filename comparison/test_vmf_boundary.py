"""Runs in the PACE environment; skipped when local Torch is unavailable."""

import importlib.util
import math
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "requires the GraphNeT Torch environment",
)
class VmfBoundaryTest(unittest.TestCase):
    def test_uniform_limit_and_finite_gradients(self):
        import torch
        from graphnet.training.loss_functions import VonMisesFisher3DLoss

        for dtype in (torch.float16, torch.float32, torch.float64):
            kappa = torch.tensor(
                [0.0, 1e-8, 1e-5, 1e-4, 1.0, 700.0, 10000.0],
                dtype=dtype,
                requires_grad=True,
            )
            value = VonMisesFisher3DLoss.log_cmk_exact(3, kappa)
            value.sum().backward()
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.isfinite(kappa.grad).all())
            self.assertAlmostEqual(
                float(value[0]), -math.log(4 * math.pi), delta=0.002
            )
            self.assertEqual(float(kappa.grad[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
