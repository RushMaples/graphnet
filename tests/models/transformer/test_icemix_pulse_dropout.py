"""CPU contracts for pulse dropout before IceMix sequence encoding."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pytorch_lightning")

from torch import nn
from torch_geometric.data import Data

from graphnet.models.transformer import icemix
from graphnet.utilities.config import ModelConfig


class _RecordingEncoder(nn.Module):
    def forward(self, x, lengths):
        self.x = x.detach().clone()
        self.lengths = lengths.detach().clone()
        return x.new_zeros((*x.shape[:2], 384))


class _NoOpBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, *args):
        return x


class _NoOpRelativePosition(_NoOpBlock):
    def forward(self, x):
        return None


@pytest.fixture
def lightweight(monkeypatch):
    """Exercise the actual input path without constructing attention layers."""
    monkeypatch.setattr(icemix, "_RDEEncoder", _RecordingEncoder)
    monkeypatch.setattr(icemix, "SpacetimeEncoder", _NoOpRelativePosition)
    monkeypatch.setattr(icemix, "Block", _NoOpBlock)
    monkeypatch.setattr(icemix, "Block_rel", _NoOpBlock)


@pytest.fixture
def data():
    lengths = torch.tensor([4, 6, 5])
    return Data(
        x=torch.arange(15 * 6, dtype=torch.float32).reshape(15, 6),
        batch=torch.repeat_interleave(torch.arange(3), lengths),
        n_pulses=lengths,
    )


@pytest.mark.parametrize("parameter", ["token_drop", "drop_chance"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities_are_rejected(parameter, value):
    with pytest.raises(ValueError, match="Pulse dropout probabilities"):
        icemix.IceMix(**{parameter: value})


@pytest.mark.parametrize(
    "token_drop,drop_chance,training",
    [(0.0, 1.0, True), (1.0, 0.0, True), (1.0, 1.0, False)],
)
def test_disabled_dropout_preserves_inputs_and_uses_no_rng(
    lightweight, data, monkeypatch, token_drop, drop_chance, training
):
    model = icemix.IceMix(token_drop=token_drop, drop_chance=drop_chance)
    model.train(training)
    original_array_to_sequence = icemix.array_to_sequence

    def record_sequence(x, batch, **kwargs):
        assert x is data.x
        assert batch is data.batch
        return original_array_to_sequence(x, batch, **kwargs)

    def unexpected_randomness(*args, **kwargs):
        pytest.fail("Disabled pulse dropout must not draw a random mask")

    monkeypatch.setattr(icemix, "array_to_sequence", record_sequence)
    monkeypatch.setattr(torch, "rand", unexpected_randomness)
    normalized, raw = model(data)
    assert normalized.shape == raw.shape == (3, 384)
    assert torch.equal(model.fourier_ext.lengths, data.n_pulses)


def test_event_then_pulse_sampling_updates_encoder_lengths(
    lightweight, data, monkeypatch
):
    model = icemix.IceMix(token_drop=0.5, drop_chance=0.5)
    x_before, batch_before = data.x, data.batch
    pulses_before = data.n_pulses.clone()
    draws = iter(
        [
            torch.tensor([0.1, 0.8, 0.3]),
            torch.tensor(
                [0.1, 0.9, 0.1, 0.9] + [0.1] * 6 + [0.1, 0.1, 0.9, 0.1, 0.9]
            ),
        ]
    )

    def controlled_randomness(size, **kwargs):
        result = next(draws)
        assert result.numel() == size
        return result.to(kwargs["device"])

    monkeypatch.setattr(torch, "rand", controlled_randomness)
    model(data)
    encoded = model.fourier_ext
    assert encoded.lengths.tolist() == [2, 6, 2]
    assert torch.equal(encoded.x[0, :2], data.x[[1, 3]])
    assert torch.equal(encoded.x[1, :6], data.x[4:10])
    assert torch.equal(encoded.x[2, :2], data.x[[12, 14]])
    assert data.x is x_before and data.batch is batch_before
    assert torch.equal(data.n_pulses, pulses_before)
    with pytest.raises(StopIteration):
        next(draws)


def test_total_dropout_keeps_first_pulse_of_every_event(lightweight, data):
    model = icemix.IceMix(token_drop=1.0, drop_chance=1.0)
    model(data)
    assert model.fourier_ext.lengths.tolist() == [1, 1, 1]
    assert torch.equal(model.fourier_ext.x[:, 0], data.x[[0, 4, 10]])


def test_seed_replays_masks_without_consuming_global_rng(lightweight, data):
    model = icemix.IceMix(token_drop=0.7, drop_chance=0.8)
    model.set_token_drop_seed(12345)
    state_before = torch.random.get_rng_state().clone()
    model(data)
    first_x = model.fourier_ext.x.clone()
    first_lengths = model.fourier_ext.lengths.clone()
    assert torch.equal(torch.random.get_rng_state(), state_before)
    model(data)
    assert torch.equal(model.fourier_ext.x, first_x)
    assert torch.equal(model.fourier_ext.lengths, first_lengths)
    assert torch.equal(torch.random.get_rng_state(), state_before)
    model.set_token_drop_seed(None)
    model(data)
    assert not torch.equal(torch.random.get_rng_state(), state_before)


def test_augmentation_preserves_checkpoints_and_eval_outputs(
    lightweight, data
):
    baseline = icemix.IceMix()
    augmented = icemix.IceMix(token_drop=0.2, drop_chance=0.5)
    augmented.load_state_dict(baseline.state_dict(), strict=True)
    assert baseline.state_dict().keys() == augmented.state_dict().keys()
    baseline.eval()
    augmented.eval()
    for original, restored in zip(baseline(data), augmented(data)):
        assert torch.equal(original, restored)


def test_reconstruction_config_roundtrip_and_old_defaults(
    lightweight, tmp_path
):
    old_path = tmp_path / "old.yml"
    ModelConfig(class_name="IceMixReconstruction", arguments={}).dump(
        str(old_path)
    )
    old_model = icemix.load_icemix_model(str(old_path))
    assert old_model.backbone.token_drop == 0.0
    assert old_model.backbone.drop_chance == 1.0

    model = icemix.IceMixReconstruction(token_drop=0.25, drop_chance=0.6)
    saved_path = tmp_path / "augmented.yml"
    model.save_config(str(saved_path))
    restored = icemix.load_icemix_model(str(saved_path))
    assert restored.config.arguments == model.config.arguments
    assert restored.config.arguments["token_drop"] == 0.25
    assert restored.config.arguments["drop_chance"] == 0.6
    assert restored.backbone.token_drop == 0.25
    assert restored.backbone.drop_chance == 0.6
    restored.load_state_dict(model.state_dict(), strict=True)
