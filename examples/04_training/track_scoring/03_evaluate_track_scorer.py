"""Evaluate saved track-score predictions without rerunning the GNN.

Separating evaluation from training makes it cheap to add plots and working
points. This script needs only the CSV written by 02_train_track_scorer.py.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


DEFAULT_ENERGY_BIN_EDGES = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the columns and numerical ranges required by every metric."""
    required = {"track_score", "track", "energy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    result = frame.dropna(subset=list(required)).copy()
    result["track"] = result["track"].astype(int)
    if set(result["track"].unique()) != {0, 1}:
        raise ValueError("Evaluation requires both binary track labels 0 and 1")
    if not result["track_score"].between(0, 1).all():
        raise ValueError("track_score contains values outside [0, 1]")
    return result


def calculate_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    """Calculate global quality, calibration, and report working points."""
    truth = frame["track"].to_numpy()
    score = frame["track_score"].to_numpy()
    tracks = truth == 1
    cascades = truth == 0

    metrics = {
        "n_events": float(len(frame)),
        "n_tracks": float(tracks.sum()),
        "n_cascades": float(cascades.sum()),
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
        "binary_cross_entropy": float(log_loss(truth, score)),
        "brier_score": float(brier_score_loss(truth, score)),
        "accuracy_at_0.5": float(accuracy_score(truth, score >= 0.5)),
        # Values below correspond to Figure 7's two illustrative regions.
        "track_acceptance_score_gt_0.8": float((score[tracks] > 0.8).mean()),
        "cascade_leakage_score_gt_0.8": float(
            (score[cascades] > 0.8).mean()
        ),
        "cascade_acceptance_score_lt_0.4": float(
            (score[cascades] < 0.4).mean()
        ),
        "track_leakage_score_lt_0.4": float((score[tracks] < 0.4).mean()),
    }
    return metrics


def calculate_energy_metrics(
    frame: pd.DataFrame, edges: List[float]
) -> pd.DataFrame:
    """Measure whether the reduced training sample fails in an energy region."""
    working = frame.copy()
    working["energy_bin"] = pd.cut(
        working["energy"], bins=edges, include_lowest=True
    )

    rows = []
    for energy_bin, group in working.groupby("energy_bin", observed=True):
        # ROC AUC is undefined when a bin contains only one truth class.
        auc = np.nan
        if group["track"].nunique() == 2:
            auc = roc_auc_score(group["track"], group["track_score"])
        rows.append(
            {
                "energy_bin": str(energy_bin),
                "n_events": len(group),
                "n_tracks": int(group["track"].sum()),
                "n_cascades": int((group["track"] == 0).sum()),
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows)


def plot_roc(frame: pd.DataFrame, output: Path) -> None:
    """Plot signal efficiency against cascade false-positive rate."""
    false_positive, true_positive, _ = roc_curve(
        frame["track"], frame["track_score"]
    )
    auc = roc_auc_score(frame["track"], frame["track_score"])
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive, true_positive, label=f"TITO (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    axis.set(xlabel="Cascade false-positive rate", ylabel="Track efficiency")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_score_distributions(frame: pd.DataFrame, output: Path) -> None:
    """Show the learned score shapes for true tracks and cascades."""
    fig, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    axis.hist(
        frame.loc[frame["track"] == 1, "track_score"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="true track",
    )
    axis.hist(
        frame.loc[frame["track"] == 0, "track_score"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="true cascade",
    )
    axis.axvline(0.4, linestyle="--", color="grey")
    axis.axvline(0.8, linestyle="--", color="grey")
    axis.set(xlabel="Predicted track score", ylabel="Normalized density")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_calibration(frame: pd.DataFrame, output: Path) -> None:
    """Compare predicted scores with observed track fractions."""
    observed, predicted = calibration_curve(
        frame["track"], frame["track_score"], n_bins=10, strategy="quantile"
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(predicted, observed, marker="o", label="model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="ideal")
    axis.set(xlabel="Mean predicted score", ylabel="Observed track fraction")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    """Read predictions, calculate metrics, and write tables/plots."""
    frame = validate_predictions(pd.read_csv(args.predictions))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(frame)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)

    energy_metrics = calculate_energy_metrics(frame, args.energy_bin_edges)
    energy_metrics.to_csv(args.output_dir / "energy_metrics.csv", index=False)

    plot_roc(frame, args.output_dir / "roc_curve.png")
    plot_score_distributions(
        frame, args.output_dir / "track_score_distributions.png"
    )
    plot_calibration(frame, args.output_dir / "calibration.png")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nEvaluation outputs written to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Define paths and the same energy bins used during split creation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a validation_predictions.csv track-score file."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--energy-bin-edges",
        type=float,
        nargs="+",
        default=DEFAULT_ENERGY_BIN_EDGES,
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
