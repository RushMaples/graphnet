"""CPU-only checks for benchmark identity, parity and split exclusion."""

import csv
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

import frozen_test as final
from contract import DATABASES, load_splits, sha256


class FinalProtocolTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            dict(source_database=DATABASES[0], event_no=1, input_sha256="abc",
                 **dict(zip(final.PREDICTIONS, [100., -50., 200., 1., 0., 0., 10.])))
        ])

    def test_parity_requires_all_seven_outputs_and_exact_actual_inputs(self):
        expected = self.frame()
        self.assertTrue(final.check_parity(expected, expected.copy())["passed"])
        for name in final.PREDICTIONS:
            actual = expected.copy()
            actual.loc[0, name] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                final.check_parity(expected, actual)
        actual = expected.copy()
        actual.loc[0, "input_sha256"] = "different"
        with self.assertRaises(ValueError):
            final.check_parity(expected, actual)
        actual = pd.concat([expected, expected])
        with self.assertRaises(ValueError):
            final.check_parity(expected, actual)

    def test_parity_rejects_wrong_event_key_and_nonfinite_predictions(self):
        expected = self.frame()
        for column, value in (("event_no", 2), ("source_database", DATABASES[1])):
            actual = expected.copy()
            actual.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                final.check_parity(expected, actual)
        for column in final.PREDICTIONS:
            for value in (float("nan"), float("inf"), -float("inf")):
                actual = expected.copy()
                actual.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    final.check_parity(expected, actual)

    def test_strict_checkpoint_load_and_fp32_eval_inference(self):
        model = Mock()
        model.float.return_value = model
        model.eval.return_value = model
        torch = types.SimpleNamespace(load=Mock(return_value={"epoch": 30, "state_dict": {"weight": 1}}))
        final.restore_model(model, "selected.ckpt", {"epoch": 30}, torch)
        model.load_state_dict.assert_called_once_with({"weight": 1}, strict=True)
        model.float.assert_called_once()
        model.eval.assert_called_once()
        model.inference.assert_called_once()
        torch.load.return_value["epoch"] = 42
        with self.assertRaises(ValueError):
            final.restore_model(model, "last.ckpt", {"epoch": 30}, torch)

    def test_selection_loader_rejects_leakage_even_with_matching_csv_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = {"selections": {}}
            for database in DATABASES:
                for index, split in enumerate(("train", "validation", "test")):
                    name = f"{Path(database).stem}_{split}.csv"
                    with (root / name).open("w", newline="") as handle:
                        csv.writer(handle).writerows([["event_no"], [index]])
                    cfg["selections"][name] = {"count": 1, "sha256": sha256(root / name)}
            splits = load_splits(root, cfg)
            self.assertEqual(len(final.expected_keys(splits["test"])), 3)
            name = f"{Path(DATABASES[0]).stem}_test.csv"
            with (root / name).open("w", newline="") as handle:
                csv.writer(handle).writerows([["event_no"], [0]])
            cfg["selections"][name]["sha256"] = sha256(root / name)
            with self.assertRaises(ValueError):
                load_splits(root, cfg)

if __name__ == "__main__":
    unittest.main()
