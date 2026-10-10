"""Refuse unmatched runs; report unweighted paired reconstruction metrics."""

import argparse
import json
from pathlib import Path

from contract import environment_identity, run_identity, sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def check_contracts(left, right):
    a, b = (
        read_json(root / "resolved_settings.json") for root in (left, right)
    )
    ignored = {"variant", "group_factor"}
    if {k: v for k, v in a.items() if k not in ignored} != {
        k: v for k, v in b.items() if k not in ignored
    }:
        raise ValueError("Run settings differ beyond variant/grouping")
    a, b = (
        read_json(root / "database_manifest.json") for root in (left, right)
    )
    identity = lambda rows: {x["database"]: x["sha256"] for x in rows}
    if identity(a) != identity(b):
        raise ValueError("Database content hashes differ")
    a, b = (read_json(root / "provenance.json") for root in (left, right))
    if environment_identity(a) != environment_identity(b):
        raise ValueError(
            "Runtime environment or GPU model differs; speed comparison would be confounded"
        )


def check_probes(left, right):
    check_contracts(left, right)
    a, b = (read_json(root / "input_probe.json") for root in (left, right))
    if a != b:
        raise ValueError(
            "Input probes differ: inspect retained pulses/feature normalization before training"
        )
    print(
        f"Matched {len(a)} validation input probes, including capped events where available."
    )


def event_metrics(frame):
    import numpy as np

    predicted = frame[[f"dir_{axis}_pred" for axis in "xyz"]].to_numpy()
    predicted = predicted / np.linalg.norm(predicted, axis=1)[:, None]
    zenith, azimuth = frame.zenith.to_numpy(), frame.azimuth.to_numpy()
    truth = np.column_stack(
        [
            np.sin(zenith) * np.cos(azimuth),
            np.sin(zenith) * np.sin(azimuth),
            np.cos(zenith),
        ]
    )
    cosine = np.clip((truth * predicted).sum(axis=1), -1, 1)
    residual = (
        frame[[f"position_{axis}_pred" for axis in "xyz"]].to_numpy()
        - frame[[f"position_{axis}" for axis in "xyz"]].to_numpy()
    )
    kappa = frame.direction_kappa.to_numpy()
    if (kappa < 0).any():
        raise ValueError("Negative kappa")
    # Exact vMF cone CDF, including its uniform-sphere limit.
    cdf = (1 - cosine) / 2
    positive = kappa > 1e-8
    cdf[positive] = np.expm1(
        kappa[positive] * (cosine[positive] - 1)
    ) / np.expm1(-2 * kappa[positive])
    return {
        "angle_deg": np.degrees(np.arccos(cosine)),
        "vertex_m": np.linalg.norm(residual, axis=1),
        "dx_m": residual[:, 0],
        "dy_m": residual[:, 1],
        "dz_m": residual[:, 2],
        "cone_cdf": cdf,
    }


def summarize(frame):
    import numpy as np

    arrays = event_metrics(frame)
    track = (frame.pid.abs().to_numpy() == 14) & (
        frame.interaction_type.to_numpy() == 1
    )
    rows = []
    for topology, mask in (
        ("all", np.ones(len(frame), dtype=bool)),
        ("track", track),
        ("cascade", ~track),
    ):
        for low, high in [
            (None, None),
            *zip(
                10.0 ** np.arange(0, 4, 0.5), 10.0 ** np.arange(0.5, 4.5, 0.5)
            ),
        ]:
            selected = mask.copy()
            if low is not None:
                selected &= (frame.energy.to_numpy() >= low) & (
                    frame.energy.to_numpy() < high
                )
            if not selected.any():
                continue
            row = {
                "topology": topology,
                "energy_min_GeV": low,
                "energy_max_GeV": high,
                "events": int(selected.sum()),
            }
            for key in ("angle_deg", "vertex_m"):
                row[key + "_median"] = float(np.median(arrays[key][selected]))
                row[key + "_p68"] = float(
                    np.quantile(arrays[key][selected], 0.68)
                )
            for key in ("dx_m", "dy_m", "dz_m"):
                row[key + "_median"] = float(np.median(arrays[key][selected]))
            for probability in (0.5, 0.68, 0.9, 0.95):
                row[f"coverage_{probability}"] = float(
                    np.mean(arrays["cone_cdf"][selected] <= probability)
                )
            rows.append(row)
    return rows


def compare(left, right, output):
    import numpy as np
    import pandas as pd

    check_probes(left, right)
    for root in (left, right):
        if read_json(root / "complete.json")["epochs"] != 10:
            raise ValueError("Require two completed ten-epoch runs")
    keys = ["source_database", "event_no"]
    frames = []
    for root in (left, right):
        manifest = read_json(root / "prediction_manifest.json")
        if sha256(root / "predictions.csv") != manifest["predictions_sha256"]:
            raise ValueError("Prediction CSV checksum mismatch")
        selected = read_json(root / "selected_checkpoint.json")
        if selected != manifest["checkpoint"]:
            raise ValueError("Prediction checkpoint metadata mismatch")
        checkpoint = Path(selected["path"])
        if checkpoint.exists() and sha256(checkpoint) != selected["sha256"]:
            raise ValueError("Selected checkpoint checksum mismatch")
        frame = pd.read_csv(root / "predictions.csv")
        if frame.duplicated(keys).any() or len(frame) != 72268:
            raise ValueError("Duplicate or incomplete prediction keys")
        if not np.isfinite(
            frame.select_dtypes(include="number").to_numpy()
        ).all():
            raise ValueError("Nonfinite predictions/truth")
        frames.append(frame.set_index(keys).sort_index())
    a, b = frames
    if not a.index.equals(b.index) or not a.input_sha256.equals(
        b.input_sha256
    ):
        raise ValueError(
            "Test event identities or actual retained model inputs differ"
        )
    truth = [
        "position_x",
        "position_y",
        "position_z",
        "azimuth",
        "zenith",
        "energy",
        "pid",
        "interaction_type",
        "oneweight",
        "n_pulses",
    ]
    if not np.array_equal(a[truth].to_numpy(), b[truth].to_numpy()):
        raise ValueError("Matched test truth differs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summarize(a)).to_csv(
        output / "carlos_metrics.csv", index=False
    )
    pd.DataFrame(summarize(b)).to_csv(output / "ajai_metrics.csv", index=False)
    am, bm = event_metrics(a), event_metrics(b)
    rng = np.random.default_rng(42)
    comparisons = {}
    for name in ("angle_deg", "vertex_m", "dz_m"):
        draws = []
        for _ in range(500):
            indices = rng.integers(0, len(a), len(a))
            draws.append(
                np.median(bm[name][indices]) - np.median(am[name][indices])
            )
        comparisons[name] = {
            "ajai_minus_carlos_median": float(
                np.median(bm[name]) - np.median(am[name])
            ),
            "paired_event_bootstrap_95_interval": np.quantile(
                draws, [0.025, 0.975]
            ).tolist(),
        }
    write_json(
        output / "paired_comparison.json",
        {
            "unweighted": True,
            "runs": [str(left), str(right)],
            "events": len(a),
            "uncertainty_scope": "Fixed-model finite-event uncertainty; excludes training-seed and detector systematics",
            "differences": comparisons,
        },
    )
    print(f"Matched ten-epoch comparison saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("carlos", type=Path)
    parser.add_argument("ajai", type=Path)
    parser.add_argument("--output-dir", default="comparison_summary")
    args = parser.parse_args()
    if args.probe:
        check_probes(args.carlos, args.ajai)
        for root in (args.carlos, args.ajai):
            if not read_json(root / "preflight.json")["passed"]:
                raise ValueError("Both GPU preflights must pass")
        receipt = args.carlos.parent / "paired_preflight.json"
        write_json(
            receipt,
            {
                "carlos": run_identity(args.carlos),
                "ajai": run_identity(args.ajai),
            },
        )
        print(f"Training receipt: {receipt}")
    else:
        compare(args.carlos, args.ajai, args.output_dir)


if __name__ == "__main__":
    main()
