"""Numerical CPU checks of production augmentation with imports isolated.

Declarations are extracted from production sources. Their real Torch operations
run unchanged, without importing unrelated PyG extensions or Lightning trainers.
These checks do not exercise framework hook dispatch, model configuration, or
attention layers. The normal IceMix pytest module covers configuration with a
complete GraphNeT runtime.
"""

import ast
import copy
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest(
        "Numerical augmentation checks require Torch"
    ) from error


ROOT = Path(__file__).resolve().parents[1]


def load_declaration(relative_path, name, **namespace):
    path = ROOT / relative_path
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declaration = next(
        node
        for node in parsed.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            declaration,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {
        "Callback": object,
        "torch": torch,
        "np": SimpleNamespace(pi=math.pi),
        "os": os,
        **namespace,
    }
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


RandomRotationCallback = load_declaration(
    "src/graphnet/training/callbacks.py", "RandomRotationCallback"
)
AugmentationCallback = load_declaration(
    "comparison/augmentation_callbacks.py",
    "AugmentationCallback",
    RandomRotationCallback=RandomRotationCallback,
)


class RecordingEncoder(torch.nn.Module):
    def forward(self, x, lengths):
        self.x = x.detach().clone()
        self.lengths = lengths.detach().clone()
        return x.new_zeros((*x.shape[:2], 384))


class NoOpBlock(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, *args):
        return x


array_to_sequence = load_declaration(
    "src/graphnet/models/utils.py", "array_to_sequence"
)
IceMix = load_declaration(
    "src/graphnet/models/transformer/icemix.py",
    "IceMix",
    Model=torch.nn.Module,
    nn=torch.nn,
    _RDEEncoder=RecordingEncoder,
    SpacetimeEncoder=NoOpBlock,
    Block=NoOpBlock,
    Block_rel=NoOpBlock,
    array_to_sequence=array_to_sequence,
)


def sample_data():
    vertex = torch.tensor([[146.3, -34.9], [46.3, 15.1], [-3.7, -34.9]])
    batch = torch.tensor([0, 0, 1, 1, 2, 2])
    x = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    x[:, :2] = vertex[batch] / 500.0
    x[1::2, 0] += 0.1
    return SimpleNamespace(
        x=x,
        batch=batch,
        direction=torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [-0.6, 0, 0.8]]),
        position_x=vertex[:, 0].clone(),
        position_y=vertex[:, 1].clone(),
        position_z=torch.tensor([15.0, -20.0, 35.0]),
        azimuth=torch.tensor([0.0, math.pi / 2, math.pi]),
    )


class RotationNumericsTest(unittest.TestCase):
    def test_rotates_pulses_and_labels_consistently_about_physical_pivot(self):
        data = sample_data()
        original = copy.deepcopy(data)
        callback = RandomRotationCallback(
            seed=42,
            node_xy_origin=(46.3 / 500.0, -34.9 / 500.0),
            position_xy_origin=(46.3, -34.9),
        )
        generator = torch.Generator().manual_seed(42)
        angles = torch.rand(3, generator=generator) * 2 * math.pi
        cosine, sine = angles.cos(), angles.sin()
        callback.on_train_batch_start(None, None, data, 0)

        offset_x = original.position_x - 46.3
        offset_y = original.position_y + 34.9
        expected_x = offset_x * cosine - offset_y * sine + 46.3
        expected_y = offset_x * sine + offset_y * cosine - 34.9
        torch.testing.assert_close(data.position_x, expected_x)
        torch.testing.assert_close(data.position_y, expected_y)
        # The first pulse is at the vertex, in normalized node coordinates.
        torch.testing.assert_close(data.x[::2, 0] * 500, data.position_x)
        torch.testing.assert_close(data.x[::2, 1] * 500, data.position_y)
        # The second pulse is 50 m farther in x before the same rotation.
        displacement = (data.x[1::2, :2] - data.x[::2, :2]) * 500
        torch.testing.assert_close(displacement[:, 0], 50 * cosine)
        torch.testing.assert_close(displacement[:, 1], 50 * sine)

        direction = original.direction
        torch.testing.assert_close(
            data.direction[:, 0],
            direction[:, 0] * cosine - direction[:, 1] * sine,
        )
        torch.testing.assert_close(
            data.direction[:, 1],
            direction[:, 0] * sine + direction[:, 1] * cosine,
        )
        torch.testing.assert_close(
            data.direction.norm(dim=1), direction.norm(dim=1)
        )
        torch.testing.assert_close(data.direction[:, 2], direction[:, 2])
        torch.testing.assert_close(
            data.azimuth, (original.azimuth + angles) % (2 * math.pi)
        )
        torch.testing.assert_close(data.position_z, original.position_z)
        torch.testing.assert_close(data.x[:, 2:], original.x[:, 2:])
        self.assertFalse(torch.equal(data.x[:2, :2], data.x[2:4, :2]))

    def test_local_rotation_generator_does_not_consume_global_rng(self):
        before = torch.random.get_rng_state().clone()
        callback = RandomRotationCallback(seed=42)
        callback.on_train_batch_start(None, None, sample_data(), 0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_no_validation_or_test_rotation_hook_is_declared(self):
        for callback in (RandomRotationCallback, AugmentationCallback):
            for name in callback.__dict__:
                self.assertFalse(name.startswith("on_validation"))
                self.assertFalse(name.startswith("on_test"))


class AugmentationSeedTest(unittest.TestCase):
    def invoke(self, callback, epoch=2, batch_idx=3, rank=0):
        seeds = []
        backbone = SimpleNamespace(set_token_drop_seed=seeds.append)
        trainer = SimpleNamespace(current_epoch=epoch, global_rank=rank)
        data = sample_data()
        callback.on_train_batch_start(
            trainer, SimpleNamespace(backbone=backbone), data, batch_idx
        )
        return seeds[0], data

    def test_repeated_batch_seed_replays_rotation_without_global_rng(self):
        callback = AugmentationCallback(seed=42, rotation=True)
        before = torch.random.get_rng_state().clone()
        seed, first = self.invoke(callback)
        repeated_seed, second = self.invoke(callback)
        self.assertEqual(seed, repeated_seed)
        self.assertEqual(seed, (42 + 2 * 1_000_000_007 + 3 * 1_000_003))
        for attribute in (
            "x",
            "direction",
            "position_x",
            "position_y",
            "azimuth",
        ):
            torch.testing.assert_close(
                getattr(first, attribute),
                getattr(second, attribute),
                rtol=0,
                atol=0,
            )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_epoch_batch_and_rank_have_distinct_seed_streams(self):
        callback = AugmentationCallback(seed=42, rotation=True)
        results = [
            self.invoke(callback),
            self.invoke(callback, epoch=3),
            self.invoke(callback, batch_idx=4),
            self.invoke(callback, rank=1),
        ]
        self.assertEqual(len({seed for seed, _ in results}), 4)
        for _, data in results[1:]:
            self.assertFalse(torch.equal(results[0][1].x, data.x))

    def test_disabled_rotation_leaves_data_unchanged(self):
        callback = AugmentationCallback(seed=42, rotation=False)
        before = torch.random.get_rng_state().clone()
        seed, data = self.invoke(callback)
        original = sample_data()
        self.assertGreater(seed, 0)
        for attribute in vars(original):
            torch.testing.assert_close(
                getattr(data, attribute),
                getattr(original, attribute),
                rtol=0,
                atol=0,
            )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))


class PulseDropNumericsTest(unittest.TestCase):
    def test_rejects_invalid_probability_boundaries(self):
        for name in ("token_drop", "drop_chance"):
            for value in (-0.1, 1.1, float("nan"), float("inf")):
                with self.subTest(parameter=name, value=value):
                    with self.assertRaisesRegex(ValueError, "probabilities"):
                        IceMix(**{name: value})

    def test_disabled_training_and_eval_preserve_inputs_and_rng(self):
        for token_drop, drop_chance, training in (
            (0.0, 1.0, True),
            (1.0, 0.0, True),
            (1.0, 1.0, False),
        ):
            with self.subTest(
                token_drop=token_drop,
                drop_chance=drop_chance,
                training=training,
            ):
                model = IceMix(
                    token_drop=token_drop, drop_chance=drop_chance
                ).train(training)
                data = sample_data()
                before = copy.deepcopy(data)
                state = torch.random.get_rng_state().clone()
                model(data)
                self.assertEqual(model.fourier_ext.lengths.tolist(), [2, 2, 2])
                for attribute in vars(before):
                    torch.testing.assert_close(
                        getattr(data, attribute),
                        getattr(before, attribute),
                        rtol=0,
                        atol=0,
                    )
                self.assertTrue(
                    torch.equal(torch.random.get_rng_state(), state)
                )

    def test_event_and_pulse_sampling_update_encoder_without_mutation(self):
        from unittest.mock import patch

        model = IceMix(token_drop=0.5, drop_chance=0.5)
        data = sample_data()
        x_before, batch_before = data.x, data.batch
        values_before = copy.deepcopy(data)
        draws = [
            torch.tensor([0.1, 0.8, 0.3]),
            torch.tensor([0.1, 0.9, 0.1, 0.1, 0.9, 0.1]),
        ]
        with patch.object(torch, "rand", side_effect=draws) as random:
            model(data)
        self.assertEqual(random.call_count, 2)
        self.assertEqual(model.fourier_ext.lengths.tolist(), [1, 2, 1])
        torch.testing.assert_close(model.fourier_ext.x[0, 0], data.x[1])
        torch.testing.assert_close(model.fourier_ext.x[1, :2], data.x[2:4])
        torch.testing.assert_close(model.fourier_ext.x[2, 0], data.x[4])
        self.assertIs(data.x, x_before)
        self.assertIs(data.batch, batch_before)
        for attribute in vars(values_before):
            torch.testing.assert_close(
                getattr(data, attribute),
                getattr(values_before, attribute),
                rtol=0,
                atol=0,
            )

    def test_total_dropout_keeps_first_pulse_per_event(self):
        data = sample_data()
        model = IceMix(token_drop=1.0, drop_chance=1.0)
        model(data)
        self.assertEqual(model.fourier_ext.lengths.tolist(), [1, 1, 1])
        torch.testing.assert_close(
            model.fourier_ext.x[:, 0], data.x[[0, 2, 4]]
        )

    def test_seed_replays_exact_carlos_draws_and_preserves_global_rng(self):
        data = sample_data()
        model = IceMix(token_drop=0.7, drop_chance=0.8)
        model.set_token_drop_seed(12345)
        state = torch.random.get_rng_state().clone()
        model(data)
        first_x = model.fourier_ext.x.clone()
        first_lengths = model.fourier_ext.lengths.clone()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        model(data)
        torch.testing.assert_close(
            model.fourier_ext.x, first_x, rtol=0, atol=0
        )
        torch.testing.assert_close(model.fourier_ext.lengths, first_lengths)

        # Match the Carlos reference: event draws precede all pulse draws.
        generator = torch.Generator().manual_seed(12345)
        events = torch.rand(3, generator=generator) < 0.8
        pulses = torch.rand(6, generator=generator) < 0.7
        keep = ~(events[data.batch] & pulses)
        for event in range(3):
            indices = torch.where(data.batch == event)[0]
            if not keep[indices].any():
                keep[indices[0]] = True
        expected, _, lengths = array_to_sequence(
            data.x[keep], data.batch[keep]
        )
        torch.testing.assert_close(first_x, expected)
        torch.testing.assert_close(first_lengths, lengths)
        model.set_token_drop_seed(None)
        model(data)
        self.assertFalse(torch.equal(torch.random.get_rng_state(), state))

    def test_augmentation_preserves_checkpoint_structure_and_eval_output(self):
        data = sample_data()
        original = IceMix().eval()
        augmented = IceMix(token_drop=0.5, drop_chance=0.5).eval()
        augmented.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(
            original.state_dict().keys(), augmented.state_dict().keys()
        )
        for expected, actual in zip(original(data), augmented(data)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
