"""Inference only for the validation-selected full-14534898 raw model.

Run validation parity before constructing a test loader. No fitting, checkpoint
selection, augmentation callbacks, or concentration calibration is performed.
"""

import argparse
import hashlib
import itertools
import time
from pathlib import Path

from augmentation import COUNTS, make_provenance, model_and_data, read_json
from continue_augmentation import (
    checkpoint_path,
    validate_export,
    verify_original_files,
)
from contract import DATABASES, FEATURES, environment_identity, load_splits, sha256, write_json
from run import activate_source, build_loaders, export_predictions

CHECKPOINT_SHA256 = "0d77854d1c72c0272bba9f8e01b8b4fb19265aa27f35d35ee276a2464cd2650c"
COMPLETE_SHA256 = "e4b59cde917def1e2ea143a2e4d16c41c14499d3af7270b0da4b72c898aa1d8a"
MODEL_CONFIG_SHA256 = "408dd78fc207b8533852667912a8da11abe9ef3a2b0ad93d0b52b13abc618be5"
PREDICTIONS = [
    "position_x_pred", "position_y_pred", "position_z_pred",
    "dir_x_pred", "dir_y_pred", "dir_z_pred", "direction_kappa",
]
KEYS = ["source_database", "event_no"]
PARITY_BATCHES = 4
# Fixed before test access: FP32 rounding allowances in metres, unit direction,
# and kappa, plus 2e-5 relative error. Exact input hashes are mandatory.
PARITY_RTOL = 2e-5
PARITY_ATOL = [2e-3] * 3 + [2e-6] * 3 + [2e-4]
SAMPLE = "reused-benchmark"


def verify_receipts(root):
    root = Path(root)
    if sha256(root / "complete.json") != COMPLETE_SHA256:
        raise ValueError("Require the audited full-14534898 completion receipt")
    complete = read_json(root / "complete.json")
    for name, digest in complete["artifact_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Completed artifact changed: {name}")
    if sha256(root / "model_config.yml") != MODEL_CONFIG_SHA256:
        raise ValueError("Recorded model configuration changed")
    cfg = read_json(root / "resolved_settings.json")
    selected = read_json(root / "selected_checkpoint.json")
    if selected["sha256"] != CHECKPOINT_SHA256 or selected["epoch"] != 30:
        raise ValueError("Require the frozen validation-selected epoch-31 checkpoint")
    validate_export(root, cfg, selected)
    return cfg, selected


def verify_databases(data_root, manifest):
    """Hash each staged DB once; original hashes already bind schema/membership."""
    if len(manifest) != len(DATABASES) or {row["database"] for row in manifest} != set(DATABASES):
        raise ValueError("Unexpected database manifest")
    records = []
    for row in manifest:
        path = Path(data_root) / row["database"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Database differs from the audited training data: {path}")
        records.append(dict(row, path=str(path.resolve())))
    return records


def expected_keys(selections):
    return {(database, int(event)) for database, ids in zip(DATABASES, selections) for event in ids}


def check_parity(reference, actual):
    import numpy as np

    if len(reference) == 0 or reference.duplicated(KEYS).any() or actual.duplicated(KEYS).any():
        raise ValueError("Empty or duplicate parity keys")
    left = reference.set_index(KEYS).sort_index()
    right = actual.set_index(KEYS).sort_index()
    if not left.index.equals(right.index) or not left.input_sha256.equals(right.input_sha256):
        raise ValueError("Validation parity event keys or actual six-input hashes differ")
    a, b = left[PREDICTIONS].to_numpy(), right[PREDICTIONS].to_numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isclose(
        b, a, rtol=PARITY_RTOL, atol=PARITY_ATOL
    ).all():
        raise ValueError("Validation prediction parity failed; do not access test")
    return {
        "passed": True, "rows": len(left), "rtol": PARITY_RTOL,
        "atol_by_prediction": dict(zip(PREDICTIONS, PARITY_ATOL)),
        "max_absolute_difference": dict(zip(PREDICTIONS, np.max(np.abs(a - b), axis=0).tolist())),
        "input_hashes_exact": True,
    }


def restore_model(model, path, selected, torch):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["epoch"] != selected["epoch"]:
        raise ValueError("Checkpoint epoch disagrees with selected receipt")
    model.load_state_dict(state["state_dict"], strict=True)
    model.float().eval().inference()


class PrefixLoader:
    """Keep original validation batch composition, padding, and event order."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return itertools.islice(iter(self.loader), PARITY_BATCHES)


def evaluate(args):
    started = time.perf_counter()
    phases = {}
    run = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    if output == run or run in output.parents or output in run.parents or output.exists():
        raise ValueError("Use a new output directory separate from the training run")
    cfg, selected = verify_receipts(run)
    selected_path = checkpoint_path(run, selected)
    original = read_json(run / "provenance.json")
    verify_original_files(original, args.source_root)
    splits = load_splits(args.selection_root, cfg)
    if {key: sum(map(len, values)) for key, values in splits.items()} != COUNTS:
        raise ValueError("Frozen selection row counts changed")
    source, graphnet_path = activate_source(args.source_root, "ajai")
    import torch
    import pytorch_lightning as pl
    import pandas as pd
    import resource
    from evaluate import summarize
    from full_diagnostics import diagnostics

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one CUDA GPU")
    provenance = make_provenance(source, graphnet_path, torch, pl)
    if environment_identity(provenance) != environment_identity(original):
        raise ValueError("Runtime environment differs from the audited validation export")
    output.mkdir(parents=True)
    write_json(output / "provenance.json", provenance)
    write_json(output / "protocol.json", {
        "training_run": str(run), "complete_sha256": COMPLETE_SHA256,
        "checkpoint": selected, "model_config_sha256": MODEL_CONFIG_SHA256,
        "model": "raw; no kappa rescaling", "test_rows": COUNTS["test"],
        "sample": SAMPLE,
        "features": FEATURES, "input_policy": "p14-eval-v1",
        "precision": "32-true", "training_augmentation": False,
        "fitting_or_test_based_selection": False,
        "parity_batches": PARITY_BATCHES, "parity_rtol": PARITY_RTOL,
        "parity_atol": PARITY_ATOL,
    })
    phases["receipt_source_environment_seconds"] = time.perf_counter() - started
    phase = time.perf_counter()
    databases = verify_databases(args.data_root, read_json(run / "database_manifest.json"))
    write_json(output / "database_manifest.json", databases)
    phases["database_hash_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    pl.seed_everything(cfg["seed"], workers=True)
    model, dataset_kwargs = model_and_data(cfg)
    if list(dataset_kwargs["features"]) != FEATURES:
        raise ValueError("Expected exactly the frozen six model features")
    restore_model(model, selected_path, selected, torch)
    # No training loader or callbacks are constructed. The test loader is not
    # constructed until the validation-only parity gate has passed.
    validation_loader = build_loaders(args, cfg, {"validation": splits["validation"]}, dataset_kwargs)["validation"]
    reference = pd.read_csv(run / "validation" / "predictions.csv", nrows=PARITY_BATCHES * cfg["batch_size"])
    parity_dir = output / "validation_parity"
    parity_dir.mkdir()
    export_predictions(model, PrefixLoader(validation_loader), parity_dir, selected,
                       expected_keys=set(reference[KEYS].itertuples(index=False, name=None)), split_name="validation-parity")
    parity = check_parity(reference, pd.read_csv(parity_dir / "predictions.csv"))
    write_json(output / "validation_parity.json", parity)
    del validation_loader
    phases["model_restore_validation_parity_seconds"] = time.perf_counter() - phase
    test_selections = splits["test"]
    write_json(output / "test_selections.json", {
        "sample": SAMPLE, "selections": {
            database: ids for database, ids in zip(DATABASES, test_selections)
        },
    })
    phase = time.perf_counter()
    loader = build_loaders(args, cfg, {"test": test_selections}, dataset_kwargs)["test"]
    torch.cuda.reset_peak_memory_stats()
    export_predictions(model, loader, output, selected,
                       expected_keys=expected_keys(test_selections), split_name="test")
    peak_gpu = torch.cuda.max_memory_allocated()
    del loader
    phases["test_export_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    frame = pd.read_csv(output / "predictions.csv")
    actual_keys = set(frame[KEYS].itertuples(index=False, name=None))
    if len(frame) != COUNTS["test"] or frame.duplicated(KEYS).any() or actual_keys != expected_keys(test_selections):
        raise ValueError("Test export keys/counts differ from frozen selections")
    pd.DataFrame(summarize(frame)).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(diagnostics(frame, cfg["vertex_weight"], azimuth_bins=True)).to_csv(output / "diagnostics.csv", index=False)
    phases["diagnostics_seconds"] = time.perf_counter() - phase
    key_digest = hashlib.sha256("".join(f"{db}:{event}\n" for db, event in sorted(actual_keys)).encode()).hexdigest()
    write_json(output / "complete.json", {
        "complete": True, "rows": len(frame), "unique_keys": len(actual_keys),
        "exact_frozen_test_keys": True, "sorted_keys_sha256": key_digest,
        "checkpoint": selected, "sample": SAMPLE, "raw_model": True, "validation_parity_passed": True,
        "peak_gpu_allocated_bytes": peak_gpu,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_worker_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
        "memory_note": "Linux ru_maxrss; worker peak is the maximum terminated child, not their sum. Slurm sacct MaxRSS complements this receipt.",
        "wall_seconds": time.perf_counter() - started, "phases": phases,
        "artifact_hashes": {name: sha256(output / name) for name in (
            "protocol.json", "provenance.json", "database_manifest.json", "test_selections.json",
            "validation_parity.json", "prediction_manifest.json", "predictions.csv", "metrics.csv", "diagnostics.csv",
        )},
    })
    print(f"Frozen raw-model test evaluation completed: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-root", "data-root", "selection-root", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    args.variant, args.group_factor = "ajai", 0
    evaluate(args)


if __name__ == "__main__":
    main()
