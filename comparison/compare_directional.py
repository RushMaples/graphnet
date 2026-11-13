"""Validate and summarize all three completed directional screen arms.

python comparison/compare_directional.py RUN_ROOT --output-dir SUMMARY
This reports validation results. Ten epochs and one seed are a screen rather
than evidence of convergence or detector-systematic robustness.
"""

import argparse
import json
from pathlib import Path

from contract import environment_identity, sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_runs(root):
    from plot_directional import ARMS, arm_folder
    common = None
    for arm in ARMS:
        folder = arm_folder(root, arm)
        complete = read(folder / "complete.json")
        cfg = read(folder / "resolved_settings.json")
        selected = read(folder / "selected_checkpoint.json")
        manifest = read(folder / "validation" / "prediction_manifest.json")
        provenance = read(folder / "provenance.json")
        if (complete.get("epochs") != 10 or complete.get("test_exported")
                or not complete.get("validation_exported") or not manifest.get("complete")
                or cfg["arm"] != arm or manifest["split"] != "validation"
                or manifest["rows"] != 72264
                or manifest["sha256"] != sha256(folder / "validation" / "predictions.csv")):
            raise ValueError(f"Incomplete or changed directional screen: {folder}")
        if manifest["selected_checkpoint"] != selected:
            raise ValueError(f"Validation checkpoint differs from selected checkpoint: {arm}")
        for receipt in ("settings", "provenance", "database_manifest"):
            name = "resolved_settings.json" if receipt == "settings" else receipt + ".json"
            if manifest[receipt + "_sha256"] != sha256(folder / name):
                raise ValueError(f"Changed validation receipt: {arm}/{name}")
        comparable_cfg = {k: v for k, v in cfg.items() if k != "arm"}
        identity = {
            "settings": comparable_cfg,
            "initialization": read(folder / "initial_shared_state.json"),
            "source_files": provenance["source_files"],
            "harness_files": provenance["harness_files"],
            "environment": environment_identity(provenance),
            "databases": {r["database"]: r["sha256"] for r in read(folder / "database_manifest.json")},
            "input_probe": read(folder / "input_probe.json"),
        }
        if common is None:
            common = identity
        elif common != identity:
            raise ValueError(f"Common sources, environment, data, settings or initialization differ: {arm}")
    return common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from plot_directional import load_frames
    common = verify_runs(args.run_root)
    frames = load_frames(args.run_root)
    rows = []
    for arm, frame in frames.items():
        for topology in ("all", "track", "cascade"):
            values = frame if topology == "all" else frame[frame.track if topology == "track" else ~frame.track]
            row = dict(arm=arm, topology=topology, events=len(values))
            for metric in ("angle_deg", "vertex_error_m", "z_error_m", "rho_error_m"):
                for quantile in (.5, .68, .9, .95):
                    row[f"{metric}_p{round(100*quantile)}"] = values[metric].quantile(quantile)
            row["direction_nll_mean"] = values.direction_nll.mean()
            row["physical_joint_nll_mean"] = (values.direction_nll + .026*values.vertex_error_m).mean()
            for probability in (.5, .68, .9):
                row[f"coverage_{probability}"] = (values.hpd_mass_at_truth <= probability).mean()
                row[f"median_area_{probability}_sr"] = values[f"region_area_{probability}_sr"].median()
            rows.append(row)
    result = pd.DataFrame(rows)
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Empty subgroup or nonfinite summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "comparison_protocol.json").write_text(json.dumps({
        "split": "validation", "events": 72264, "seed": 42, "epochs": 10,
        "density_measure": "per steradian", "weighting": "unweighted",
        "common_identity": common,
        "limitations": ["single-seed ten-epoch screen", "validation has influenced model development",
                        "flow MAP, coverage and region area are posterior Monte Carlo approximations",
                        "baseline training uses original vMF approximation; this summary uses exact vMF density"],
    }, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
