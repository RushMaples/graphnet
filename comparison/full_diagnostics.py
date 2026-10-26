"""Validation diagnostics from saved predictions or existing loss recorders."""

import csv
from pathlib import Path

from continue_augmentation import prediction_diagnostics


def cone_regions(kappa, probability):
    """Exact vMF probability-cone radius (degrees) and solid angle (sr).

    Work with 1-cos(radius) directly, and use the half-angle identity to
    retain small, nonzero radii when concentration is extremely large.
    """
    import numpy as np

    kappa = np.asarray(kappa, dtype=np.float64)
    if (
        not np.isfinite(kappa).all()
        or (kappa < 0).any()
        or not 0 < probability < 1
    ):
        raise ValueError(
            "Require finite nonnegative kappa and 0 < probability < 1"
        )
    deficit = np.full_like(kappa, 2 * probability)
    positive = kappa > 1e-8
    deficit[positive] = (
        -np.log1p(probability * np.expm1(-2 * kappa[positive]))
        / kappa[positive]
    )
    deficit = np.clip(deficit, 0, 2)
    radius = np.degrees(2 * np.arcsin(np.sqrt(deficit / 2)))
    return radius, 2 * np.pi * deficit


def diagnostics(frame, vertex_weight, azimuth_bins=False):
    """Report topology, tails, exact vMF coverage, and wrapped azimuth error."""
    import numpy as np
    from evaluate import event_metrics

    numeric = frame.select_dtypes(include="number").to_numpy()
    if len(frame) == 0 or not np.isfinite(numeric).all():
        raise ValueError("Empty/nonfinite diagnostic predictions")
    direction = frame[[f"dir_{axis}_pred" for axis in "xyz"]].to_numpy()
    direction_norm = np.linalg.norm(direction, axis=1)
    if not np.isfinite(direction_norm).all() or (direction_norm == 0).any():
        raise ValueError("Diagnostic directions must have finite nonzero norm")
    arrays = event_metrics(frame)
    # Retain reported kappa/coverage for comparison to the original harness.
    # The loss natural vector is raw_direction * reported_kappa, so its
    # concentration is the norm of that vector. FP16 normalization uses
    # norm + eps and can leave direction norms meaningfully below one.
    effective = frame.copy()
    effective["direction_kappa"] = (
        direction_norm * frame.direction_kappa.to_numpy()
    )
    effective_arrays = event_metrics(effective)
    truth_azimuth = np.mod(frame.azimuth.to_numpy(), 2 * np.pi)
    residual = np.arctan2(direction[:, 1], direction[:, 0]) - truth_azimuth
    residual = np.degrees(np.arctan2(np.sin(residual), np.cos(residual)))
    track = (frame.pid.abs().to_numpy() == 14) & (
        frame.interaction_type.to_numpy() == 1
    )
    regions = {
        probability: cone_regions(
            frame.direction_kappa.to_numpy(), probability
        )
        for probability in (0.5, 0.68, 0.9, 0.95, 0.99)
    }
    effective_regions = {
        probability: cone_regions(
            effective.direction_kappa.to_numpy(), probability
        )
        for probability in regions
    }
    rows = prediction_diagnostics(frame, vertex_weight)
    for row in rows:
        mask = {
            "all": np.ones(len(frame), bool),
            "track": track,
            "cascade": ~track,
        }[row["topology"]].copy()
        if row["energy_min_GeV"] is not None:
            mask &= (frame.energy >= row["energy_min_GeV"]).to_numpy()
            mask &= (frame.energy < row["energy_max_GeV"]).to_numpy()
        row.update(azimuth_min_deg=None, azimuth_max_deg=None)
        for probability in (0.5, 0.68, 0.9, 0.95, 0.99):
            coverage = float(np.mean(arrays["cone_cdf"][mask] <= probability))
            row[f"coverage_{probability}"] = coverage
            row[f"coverage_error_{probability}"] = coverage - probability
            radius, solid_angle = regions[probability]
            row[f"cone_radius_{probability}_deg_median"] = float(
                np.median(radius[mask])
            )
            row[f"cone_solid_angle_{probability}_sr_median"] = float(
                np.median(solid_angle[mask])
            )
            effective_coverage = float(
                np.mean(effective_arrays["cone_cdf"][mask] <= probability)
            )
            row[f"effective_coverage_{probability}"] = effective_coverage
            row[f"effective_coverage_error_{probability}"] = (
                effective_coverage - probability
            )
            radius, solid_angle = effective_regions[probability]
            row[f"effective_cone_radius_{probability}_deg_median"] = float(
                np.median(radius[mask])
            )
            row[f"effective_cone_solid_angle_{probability}_sr_median"] = float(
                np.median(solid_angle[mask])
            )
        for probability in (0.5, 0.68, 0.9, 0.95, 0.99):
            row[f"azimuth_abs_deg_p{int(100 * probability)}"] = float(
                np.quantile(np.abs(residual[mask]), probability)
            )
        row["azimuth_signed_deg_median"] = float(np.median(residual[mask]))
        row["kappa_median"] = float(np.median(frame.direction_kappa[mask]))
        row["effective_kappa_median"] = float(
            np.median(effective.direction_kappa[mask])
        )
        row["direction_norm_min"] = float(np.min(direction_norm[mask]))
        row["direction_norm_median"] = float(np.median(direction_norm[mask]))
    if azimuth_bins:
        for low in range(0, 360, 45):
            mask = (np.degrees(truth_azimuth) >= low) & (
                np.degrees(truth_azimuth) < low + 45
            )
            if mask.any():
                for row in diagnostics(frame.loc[mask], vertex_weight):
                    if row["energy_min_GeV"] is None:
                        rows.append(
                            dict(
                                row,
                                azimuth_min_deg=low,
                                azimuth_max_deg=low + 45,
                            )
                        )
    return rows


def append_rows(path, rows):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def validation_callback(output, cfg, expected_rows):
    """Collect existing detached validation predictions; no extra forward."""
    from pytorch_lightning import Callback
    from graphnet.training.callbacks import PhysicsMetricsCallback

    class ValidationDiagnostics(Callback):
        def on_validation_epoch_start(self, trainer, pl_module):
            self.parts = []

        def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx=0,
        ):
            import pandas as pd

            recorders = PhysicsMetricsCallback._recorders(pl_module)
            position, direction = recorders["position"], recorders["direction"]
            p = position.last_prediction.float().cpu().numpy()
            d = direction.last_prediction.float().cpu().numpy()
            rows = {
                f"position_{axis}_pred": p[:, i] * 100
                for i, axis in enumerate("xyz")
            }
            rows.update(
                {f"dir_{axis}_pred": d[:, i] for i, axis in enumerate("xyz")}
            )
            rows["direction_kappa"] = d[:, 3]
            for name in (
                "position_x",
                "position_y",
                "position_z",
                "azimuth",
                "zenith",
                "energy",
                "pid",
                "interaction_type",
            ):
                rows[name] = (
                    getattr(batch, name).detach().cpu().reshape(-1).numpy()
                )
            self.parts.append(pd.DataFrame(rows))

        def on_validation_epoch_end(self, trainer, pl_module):
            import pandas as pd

            frame = pd.concat(self.parts, ignore_index=True)
            self.parts = []
            if len(frame) != expected_rows:
                raise RuntimeError(
                    "Incomplete per-epoch validation diagnostics"
                )
            rows = diagnostics(frame, cfg["vertex_weight"], azimuth_bins=True)
            for row in rows:
                row["recorder_prediction_joint_loss"] = row.pop(
                    "fp32_prediction_joint_loss"
                )
            append_rows(
                Path(output) / "epoch_diagnostics.csv",
                [
                    dict(
                        epoch=trainer.current_epoch,
                        precision=cfg["precision"],
                        prediction_source="existing validation loss recorders",
                        **row,
                    )
                    for row in rows
                ],
            )

    return ValidationDiagnostics()


def preflight_diagnostics(model, loader, output, cfg):
    """Exercise the real recorder callback once using the separate probe.

    This is one explicitly recorded extra preflight forward; epoch callbacks
    still reuse existing predictions and add no model forward passes.
    """
    import types
    import torch

    destination = Path(output) / "diagnostic_preflight"
    destination.mkdir()
    batch = next(iter(loader)).to("cuda")
    callback = validation_callback(destination, cfg, int(batch.num_graphs))
    trainer = types.SimpleNamespace(current_epoch=-1)
    callback.on_validation_epoch_start(trainer, model)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        loss = model.shared_step([batch], 0)
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite diagnostic preflight loss")
    callback.on_validation_batch_end(trainer, model, loss, batch, 0)
    callback.on_validation_epoch_end(trainer, model)
    return {
        "passed": True,
        "validation_events": int(batch.num_graphs),
        "extra_forward_passes": 1,
        "separate_model": True,
        "precision": cfg["precision"],
        "diagnostics_path": str(destination / "epoch_diagnostics.csv"),
    }
