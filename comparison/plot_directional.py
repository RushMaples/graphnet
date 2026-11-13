"""Plot completed directional screens in the TANGO report's comparison style.

Usage: python comparison/plot_directional.py RUN_ROOT --output-dir FIGURES
Expects vmf/, king/, flow/ validation exports and metrics.csv training logs.
Plots are unweighted. The report's flux-weighted uncertainty figures are a
style reference; no astrophysical flux weighting is inferred from oneweight.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("vmf", "king", "flow")
COLORS = {"vmf": "#2465a8", "king": "#d27822", "flow": "#238568"}
LABELS = {"vmf": "IceMix vMF", "king": "Pulse-aware King", "flow": "Spherical flow (4 layers)"}
KEYS = ["source_database", "event_no"]


def arm_folder(root, arm):
    folder = root / arm
    return folder / "model" if (folder / "model").is_dir() else folder


def load_frames(root, expected_rows=72264):
    """Require identical event keys, truth, and actual held-out input features."""
    frames, reference = {}, None
    truth = ["input_sha256", "energy", "pid", "interaction_type", "azimuth", "zenith",
             "position_x", "position_y", "position_z", "n_pulses"]
    for arm in ARMS:
        frame = pd.read_csv(arm_folder(root, arm) / "validation" / "predictions.csv")
        if len(frame) != expected_rows or frame.duplicated(KEYS).any():
            raise ValueError(f"Incomplete or duplicate export: {arm}")
        frame = frame.set_index(KEYS).sort_index()
        current = frame[truth]
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(f"Validation keys, truth, counts or pulse inputs differ: {arm}")
        numeric = frame.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Nonfinite validation export: {arm}")
        frame["z_error_m"] = abs(frame.position_z_pred - frame.position_z)
        # Detector-centred cylindrical radius, not the transverse displacement.
        frame["rho_error_m"] = abs(
            np.hypot(frame.position_x_pred, frame.position_y_pred)
            - np.hypot(frame.position_x, frame.position_y))
        frame["track"] = (abs(frame.pid) == 14) & (frame.interaction_type == 1)
        frames[arm] = frame
    return frames


def bin_quantile(x, y, edges, quantile=0.5, min_count=30):
    result, counts = [], []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= low) & ((x <= high) if i == len(edges) - 2 else (x < high))
        count = int(np.count_nonzero(mask))
        counts.append(count)
        result.append(np.quantile(y[mask], quantile) if count >= min_count else np.nan)
    return np.asarray(result), np.asarray(counts)


def resolution_plots(frames, output, plt, min_count):
    edges = np.linspace(1, 4, 16)
    centers = (edges[:-1] + edges[1:]) / 2
    records = []
    for topology in ("track", "cascade"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        for col, (metric, ylabel) in enumerate((
            ("angle_deg", "Median opening angle (deg)"),
            ("z_error_m", "Median absolute z error (m)"),
            ("rho_error_m", "Median absolute radial error (m)"))):
            baseline = None
            for arm in ARMS:
                frame = frames[arm]
                mask = frame.track if topology == "track" else ~frame.track
                x = np.log10(frame.loc[mask, "energy"].to_numpy())
                y = frame.loc[mask, metric].to_numpy()
                median, counts = bin_quantile(x, y, edges, min_count=min_count)
                if baseline is None:
                    baseline = median
                ratio = np.divide(median, baseline, out=np.full_like(median, np.nan),
                                  where=np.isfinite(baseline) & (baseline > 0))
                axes[0, col].plot(centers, median, ".-", color=COLORS[arm], label=LABELS[arm])
                axes[1, col].plot(centers, ratio, ".-", color=COLORS[arm])
                for i, count in enumerate(counts):
                    records.append(dict(arm=arm, topology=topology, metric=metric,
                                        energy_min_GeV=10**edges[i], energy_max_GeV=10**edges[i+1],
                                        count=int(count), median=float(median[i]), ratio_to_vmf=float(ratio[i])))
            axes[0, col].set_ylabel(ylabel)
            axes[1, col].set_ylabel("Model / vMF")
            axes[1, col].set_xlabel(r"$\log_{10}(E_\nu/\mathrm{GeV})$")
            axes[1, col].axhline(1, color="0.3", linestyle="--", linewidth=1)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{topology.title()} events — frozen P14 validation (unweighted)")
        fig.tight_layout()
        fig.savefig(output / f"resolution_{topology}.png", dpi=180)
        fig.savefig(output / f"resolution_{topology}.pdf")
        plt.close(fig)
    pd.DataFrame(records).to_csv(output / "energy_resolution.csv", index=False)


def uncertainty_plots(frames, output, plt, min_count):
    probabilities = np.linspace(0.05, 0.95, 19)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rows = []
    for topology in ("all", "track", "cascade"):
        ax = axes[("all", "track", "cascade").index(topology)]
        for arm, frame in frames.items():
            mask = np.ones(len(frame), bool) if topology == "all" else (
                frame.track.to_numpy() if topology == "track" else ~frame.track.to_numpy())
            rank = frame.loc[mask, "hpd_mass_at_truth"].to_numpy()
            observed = [(rank <= p).mean() for p in probabilities]
            ax.plot(probabilities, observed, color=COLORS[arm], label=LABELS[arm])
            for p, cov in zip(probabilities, observed):
                rows.append(dict(arm=arm, topology=topology, probability=p, coverage=cov, count=len(rank)))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(title=topology.title(), xlabel="Nominal HPD probability", ylabel="Observed coverage",
               xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation calibration — flow coverage uses posterior sampling")
    fig.tight_layout()
    fig.savefig(output / "hpd_coverage.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "coverage.csv", index=False)
    # Coverage and region size must be viewed together; wider regions alone
    # can improve coverage without improving the density estimate.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    for col, variable in enumerate(("energy", "n_pulses")):
        edges = np.geomspace(10, 1e4, 12) if variable == "energy" else np.geomspace(2, 4096, 12)
        centers = np.sqrt(edges[:-1] * edges[1:])
        for arm, frame in frames.items():
            x = frame[variable].to_numpy()
            coverage = []
            for low, high in zip(edges[:-1], edges[1:]):
                keep = (x >= low) & (x < high)
                coverage.append((frame.loc[keep, "hpd_mass_at_truth"] <= .68).mean()
                                if keep.sum() >= min_count else np.nan)
            areas, _ = bin_quantile(x, frame["region_area_0.68_sr"].to_numpy(), edges, min_count=min_count)
            axes[0, col].plot(centers, coverage, ".-", color=COLORS[arm], label=LABELS[arm])
            axes[1, col].plot(centers, areas, ".-", color=COLORS[arm])
        axes[0, col].axhline(.68, color="0.3", linestyle="--")
        axes[0, col].set(ylabel="68% HPD coverage", xscale="log", ylim=(0, 1))
        axes[1, col].set(ylabel="Median 68% region area (sr)", xscale="log",
                         xlabel="True neutrino energy (GeV)" if variable == "energy" else "Cleaned parent pulse count")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "coverage_and_region_size.png", dpi=180)
    plt.close(fig)
    # TANGO Fig.12: true error conditional on predicted median angular error.
    # For flow this scalar cone is a descriptive summary; HPD regions above
    # retain its asymmetric or multiple-mode geometry.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    edges = np.linspace(0, 180, 25)
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, topology in zip(axes, ("track", "cascade")):
        for arm, frame in frames.items():
            mask = frame.track if topology == "track" else ~frame.track
            x = frame.loc[mask, "cone_radius_0.5_deg"].to_numpy()
            y = frame.loc[mask, "angle_deg"].to_numpy()
            median, _ = bin_quantile(x, y, edges, min_count=min_count)
            low, _ = bin_quantile(x, y, edges, .16, min_count)
            high, _ = bin_quantile(x, y, edges, .84, min_count)
            ax.plot(centers, median, ".-", color=COLORS[arm], label=LABELS[arm])
            ax.fill_between(centers, low, high, color=COLORS[arm], alpha=.10)
        ax.plot([0, 180], [0, 180], "k--", linewidth=1)
        ax.set(title=topology.title(), xlabel="Predicted median angular error (deg)",
               ylabel="Actual angular error: median and central 68% (deg)", xlim=(0, 180), ylim=(0, 180))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "predicted_vs_actual_error.png", dpi=180)
    plt.close(fig)


def training_plots(root, output, plt):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    found = False
    for arm in ARMS:
        candidates = sorted(arm_folder(root, arm).rglob("metrics.csv"))
        for path in candidates:
            frame = pd.read_csv(path)
            if "epoch" not in frame or "val_loss" not in frame:
                continue
            found = True
            for ax, metric in zip(axes, ("loss", "direction_nll", "vertex_error_m")):
                for prefix, style in (("train", "-"), ("val", "--")):
                    key = f"{prefix}_{metric}"
                    if key in frame:
                        rows = frame[["epoch", key]].dropna().groupby("epoch").last()
                        ax.plot(rows.index + 1, rows[key], style, color=COLORS[arm], label=f"{arm} {prefix}")
                ax.set(xlabel="Completed epoch", ylabel=metric.replace("_", " "))
            break
    if found:
        axes[0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-bin-count", type=int, default=30)
    args = parser.parse_args()
    if args.min_bin_count < 1:
        parser.error("--min-bin-count must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.alpha": .3,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.run_root)
    resolution_plots(frames, args.output_dir, plt, args.min_bin_count)
    uncertainty_plots(frames, args.output_dir, plt, args.min_bin_count)
    training_plots(args.run_root, args.output_dir, plt)
    (args.output_dir / "plot_protocol.json").write_text(json.dumps({
        "split": "validation", "weighting": "unweighted", "min_bin_count": args.min_bin_count,
        "ratio": "model median / vMF median; lower is better",
        "reference_style": "Jiyuan Liao TANGO technical report, Figures 2, 3 and 11",
        "flow_point_estimate": "sampled MAP approximation; consult prediction manifest for sample count",
        "flow_coverage": "Monte Carlo HPD rank; finite sampling uncertainty applies",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
