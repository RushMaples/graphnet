"""Continue the frozen both arm, stopping on validation loss, without test use.

Epochs is a maximum TOTAL horizon (at most 80), including the parent epochs.
The Lightning last checkpoint restores optimizer, scheduler, loops and AMP.
"""

import argparse
import copy
import csv
import gc
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from augmentation import (
    COUNTS,
    callbacks,
    checkpoint_record,
    make_provenance,
    model_and_data,
    read_json,
)
from continue_augmentation import (
    checkpoint_path,
    choose_checkpoint,
    continuation_settings,
    validate_checkpoint,
    validate_export,
    validate_settings,
    validate_timing,
    verify_data_identity,
    verify_original_files,
)
from contract import load_splits, sha256, write_json
from run import (
    activate_source,
    audit_databases,
    build_loaders,
    export_predictions,
    input_probe,
    numerical_preflight,
)


def stopping_policy(patience=12, min_delta=1e-4):
    if type(patience) is not int or patience < 1:
        raise ValueError("Patience must be a positive validation epoch count")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    return {
        "monitor": "val_loss",
        "mode": "min",
        "patience": patience,
        "min_delta": min_delta,
        "strict": True,
        "check_finite": True,
        "check_on_train_epoch_end": True,
        "arithmetic": "float64 score + absolute min_delta < best_score",
    }


def replay_stopping(rows, policy, initial=None):
    state = dict(
        initial or {"best_score": None, "wait_count": 0, "stopped_epoch": 0}
    )
    for row in rows:
        score = float(row["val_loss"])
        if not math.isfinite(score):
            raise FloatingPointError("Missing/nonfinite validation loss")
        if state["best_score"] is None or (
            score + policy["min_delta"] < state["best_score"]
        ):
            state.update(best_score=score, wait_count=0)
        else:
            state["wait_count"] += 1
    return state


def stopping_snapshot(callback):
    return {
        "best_score": float(callback.best_score),
        "wait_count": int(callback.wait_count),
        "stopped_epoch": int(callback.stopped_epoch),
    }


def early_stopping_callback(initial, policy):
    """A real Lightning EarlyStopping with reproducible scalar arithmetic.

    Float64 comparisons match the CSV replay, including strict delta ties.
    The original parent has no EarlyStopping state; seed it from its history.
    """
    import torch
    from pytorch_lightning.callbacks import EarlyStopping

    class CarriedEarlyStopping(EarlyStopping):
        def __init__(self):
            super().__init__(
                monitor="val_loss",
                mode="min",
                patience=policy["patience"],
                min_delta=policy["min_delta"],
                strict=True,
                check_finite=True,
                check_on_train_epoch_end=True,
            )
            self.best_score = torch.tensor(
                initial["best_score"], dtype=torch.float64
            )
            self.wait_count = initial["wait_count"]

        def _evaluate_stopping_criteria(self, current):
            state = replay_stopping(
                [{"val_loss": float(current)}], policy, stopping_snapshot(self)
            )
            self.best_score = torch.tensor(
                state["best_score"], dtype=torch.float64
            )
            self.wait_count = state["wait_count"]
            stop = self.wait_count >= self.patience
            return stop, (
                "Validation loss exhausted carried patience" if stop else None
            )

    return CarriedEarlyStopping()


def timing_rows(root, starting, completed):
    cfg = {"starting_epochs": starting, "epochs": completed}
    validate_timing(root, cfg)
    with (Path(root) / "timing.csv").open(newline="") as handle:
        return [
            dict(row, source_run=str(Path(root).resolve()))
            for row in csv.DictReader(handle)
        ]


def parent_history(root, seen=None):
    """Replay hash-bound ancestor timing, permitting relocation of a run tree."""
    root = Path(root).resolve()
    seen = set() if seen is None else seen
    if root in seen:
        raise ValueError("Cyclic continuation ancestry")
    seen.add(root)
    cfg = read_json(root / "resolved_settings.json")
    completed = read_json(root / "complete.json")
    validate_settings(cfg, completed)
    if cfg["arm"] != "both":
        raise ValueError("Full run ancestry must contain only both")
    rows = timing_rows(root, cfg.get("starting_epochs", 0), cfg["epochs"])
    ancestry = [
        {
            "run": str(root),
            "timing_sha256": sha256(root / "timing.csv"),
            "settings_sha256": sha256(root / "resolved_settings.json"),
            "complete_sha256": sha256(root / "complete.json"),
        }
    ]
    if cfg["stage"] == "screen":
        return rows, ancestry
    source = read_json(root / "checkpoint_source.json")
    ancestry[0]["checkpoint_source_sha256"] = sha256(
        root / "checkpoint_source.json"
    )
    recorded = Path(source["parent_run"])
    relocated = root.parent.parent / recorded.parent.name / recorded.name
    ancestor = recorded if recorded.is_dir() else relocated
    if not ancestor.is_dir():
        raise FileNotFoundError(
            f"Ancestor timing required for exact replay: {recorded}"
        )
    if (
        source["parent_run"] != cfg["parent_run"]
        or source["starting_epochs"] != cfg["starting_epochs"]
    ):
        raise ValueError("Ancestry settings and source receipt disagree")
    hashes = source.get("parent_artifact_hashes", {})
    if (
        not {"resolved_settings.json", "complete.json", "timing.csv"}
        <= hashes.keys()
    ):
        raise ValueError("Missing ancestor artifact hashes")
    for name, digest in hashes.items():
        if sha256(ancestor / name) != digest:
            raise ValueError(f"Ancestor artifact changed: {name}")
    previous, lineage = parent_history(ancestor, seen)
    if len(previous) != cfg["starting_epochs"]:
        raise ValueError("Ancestry does not cover the starting epoch")
    return previous + rows, lineage + ancestry


def completion_reason(starting, maximum, actual, final, policy):
    if not starting < actual <= maximum:
        raise RuntimeError("No complete additional epoch or invalid horizon")
    stopped = final["wait_count"] >= policy["patience"]
    if stopped and final["stopped_epoch"] == actual - 1:
        return "early_stopping"
    if actual == maximum and not stopped and final["stopped_epoch"] == 0:
        return "max_epochs"
    raise RuntimeError(
        "Training ended without reaching its cap or early stopping"
    )


def write_curve(output, history, current):
    rows = history + current
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (Path(output) / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_output(output, parent):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if (
        output == parent
        or output in parent.parents
        or parent in output.parents
    ):
        raise ValueError("Full-run output must be separate from its parent")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite artifacts: {output}")
    return output


def resume_checkpoint_path(parent, cfg, complete):
    """Bind full-state resume to the completed parent's last-file receipt."""
    path = Path(parent) / "checkpoints" / "last.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"Full resumable checkpoint required: {path}")
    recorded = complete.get("last_checkpoint")
    if recorded is None and cfg["stage"] == "screen":
        # The original ten-epoch screen predates last-checkpoint receipts.
        return path.resolve()
    if (
        not isinstance(recorded, dict)
        or not isinstance(recorded.get("sha256"), str)
        or len(recorded["sha256"]) != 64
        or Path(recorded.get("path", "")).name != "last.ckpt"
    ):
        raise ValueError(
            "Parent full resumable checkpoint receipt is required"
        )
    if sha256(path) != recorded["sha256"]:
        raise ValueError(
            f"Parent full resumable checkpoint changed from its receipt: {path}"
        )
    return path.resolve()


def validate_completion(root, cfg, complete):
    """Validate whole-epoch completion and the recorded stopping trajectory."""
    actual = complete["actual_completed_epochs"]
    starting = cfg["starting_epochs"]
    if (
        type(actual) is not int
        or not starting < actual <= cfg["epochs"] <= 80
        or complete.get("complete") is not True
        or complete["epochs"] != actual
        or complete["max_total_epochs"] != cfg["epochs"]
        or complete["starting_epochs"] != starting
        or complete["additional_epochs"] != actual - starting
        or complete["parent_run"] != cfg["parent_run"]
    ):
        raise ValueError("Incomplete or inconsistent completion receipt")
    policy = stopping_policy(
        cfg["early_stopping_patience"], cfg["early_stopping_min_delta"]
    )
    final = complete["early_stopping_final"]
    reason = completion_reason(starting, cfg["epochs"], actual, final, policy)
    if complete["stop_reason"] != reason:
        raise ValueError("Incorrect stop reason")
    timing = timing_rows(root, starting, actual)
    if (
        sha256(root / "training_curve.csv")
        != complete["training_curve_sha256"]
    ):
        raise ValueError("Training curve changed")
    with (root / "training_curve.csv").open(newline="") as handle:
        curve = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in curve] != list(range(actual)):
        raise ValueError("Training curve lacks complete ancestry")
    if [row["val_loss"] for row in curve[starting:]] != [
        row["val_loss"] for row in timing
    ]:
        raise ValueError("Training curve and epoch timing disagree")
    initial = replay_stopping(curve[:starting], policy)
    replayed = replay_stopping(curve[starting:], policy, initial)
    if initial != complete["early_stopping_initial"] or (
        replayed["best_score"],
        replayed["wait_count"],
    ) != (final["best_score"], final["wait_count"]):
        raise ValueError(
            "Stopping state differs from complete validation curve"
        )
    receipt = read_json(root / "early_stopping.json")
    if receipt != {
        "policy": policy,
        "initial": initial,
        "final": final,
        "stop_reason": reason,
    }:
        raise ValueError("Stopping receipts disagree")
    for name, digest in complete["artifact_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Completed artifact changed: {name}")
    return actual, reason


def train(args):
    wall_started = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    phases = {}
    parent = Path(args.parent_run).resolve()
    parent_cfg = read_json(parent / "resolved_settings.json")
    parent_complete = read_json(parent / "complete.json")
    starting = validate_settings(parent_cfg, parent_complete, args.epochs)
    if parent_cfg["arm"] != "both" or args.epochs > 80:
        raise ValueError("Only both is allowed, with an 80-total-epoch cap")
    policy = stopping_policy(args.patience, args.min_delta)
    history, ancestry = parent_history(parent)
    initial = replay_stopping(history, policy)
    if initial["wait_count"] >= policy["patience"]:
        raise ValueError("Parent already exhausts early-stopping patience")
    selected_parent = read_json(parent / "selected_checkpoint.json")
    validate_export(parent, parent_cfg, selected_parent)
    best_row = min(history, key=lambda row: float(row["val_loss"]))
    if selected_parent["score"] != float(
        best_row["val_loss"]
    ) or selected_parent["epoch"] != int(best_row["epoch"]):
        raise ValueError(
            "Parent selected checkpoint differs from its loss curve"
        )
    original_provenance = read_json(parent / "provenance.json")
    verify_original_files(original_provenance, args.source_root)
    best_parent_path = checkpoint_path(parent, selected_parent)
    last_parent_path = resume_checkpoint_path(
        parent, parent_cfg, parent_complete
    )
    output = validate_output(args.output_dir, parent)
    root, graphnet_path = activate_source(args.source_root, "ajai")
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from runtime import RunMonitor
    from full_diagnostics import (
        diagnostics,
        preflight_diagnostics,
        validation_callback,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one GPU before running")
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise RuntimeError("Single-task/single-GPU training only")
    state = torch.load(
        last_parent_path, map_location="cpu", weights_only=False
    )
    validate_checkpoint(state, starting, parent_cfg["precision"])
    del state
    cfg = continuation_settings(parent_cfg, parent, args.epochs)
    cfg.update(
        stage="full-continuation",
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
    )
    splits = load_splits(args.selection_root, cfg)
    if {key: sum(map(len, value)) for key, value in splits.items()} != COUNTS:
        raise ValueError("Frozen selection exposure counts differ")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir, args.variant, args.group_factor = str(output), "ajai", 16
    write_json(output / "resolved_settings.json", cfg)
    write_json(
        output / "provenance.json",
        make_provenance(root, graphnet_path, torch, pl),
    )
    source = {
        "parent_run": str(parent),
        "starting_epochs": starting,
        "target_total_epochs": cfg["epochs"],
        "resume_checkpoint": {
            "path": str(last_parent_path),
            "sha256": sha256(last_parent_path),
        },
        "parent_selected_checkpoint": selected_parent,
        "ancestry": ancestry,
        "early_stopping_initial": initial,
        "early_stopping_policy": policy,
        "parent_artifact_hashes": {
            name: sha256(parent / name)
            for name in (
                "resolved_settings.json",
                "complete.json",
                "provenance.json",
                "database_manifest.json",
                "input_probe.json",
                "timing.csv",
                "selected_checkpoint.json",
                "validation/prediction_manifest.json",
            )
        },
        "restore": "Lightning Trainer.fit ckpt_path: weights, optimizer, scheduler, loops, AMP",
    }
    write_json(output / "checkpoint_source.json", source)
    phases["startup_guards_seconds"] = time.perf_counter() - wall_started
    setup_started = time.perf_counter()
    pl.seed_everything(cfg["seed"], workers=True)
    probe, dataset_kwargs = model_and_data(cfg)
    write_json(
        output / "database_manifest.json",
        audit_databases(args.data_root, splits, dataset_kwargs["features"]),
    )
    loader_splits = {
        key: value for key, value in splits.items() if key != "test"
    }
    loaders = build_loaders(args, cfg, loader_splits, dataset_kwargs)
    input_probe(loaders["validation"], output / "input_probe.json")
    verify_data_identity(parent, output)
    # This model may receive gradients during the probe. It is never fitted.
    loss = numerical_preflight(probe, loaders["validation"], "ajai")
    diagnostic_smoke = preflight_diagnostics(
        probe, loaders["validation"], output, cfg
    )
    write_json(
        output / "preflight.json",
        {
            "passed": True,
            "probe_loss": loss,
            "separate_model": True,
            "diagnostic_smoke": diagnostic_smoke,
        },
    )
    del probe, loaders
    gc.collect()
    torch.cuda.empty_cache()
    pl.seed_everything(cfg["seed"], workers=True)
    loaders = build_loaders(args, cfg, loader_splits, dataset_kwargs)
    model, _ = model_and_data(cfg)
    model.save_config(str(output / "model_config.yml"))
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.8f}",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )
    monitor = RunMonitor(output, COUNTS["train"], COUNTS["validation"])
    early = early_stopping_callback(initial, policy)
    diagnostic_callback = validation_callback(
        output, cfg, COUNTS["validation"]
    )
    phases["setup_preflight_seconds"] = time.perf_counter() - setup_started
    fit_started = time.perf_counter()
    model.fit(
        loaders["train"],
        loaders["validation"],
        max_epochs=cfg["epochs"],
        ckpt_path=str(last_parent_path),
        early_stopping_patience=cfg["early_stopping_patience"],
        gpus=[0],
        distribution_strategy="auto",
        precision=cfg["precision"],
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        callbacks=[
            *callbacks(cfg),
            diagnostic_callback,
            monitor,
            early,
            checkpoint,
        ],
        logger=CSVLogger(str(output / "logs"), name="training"),
    )
    phases["fit_seconds"] = time.perf_counter() - fit_started
    selection_started = time.perf_counter()
    actual = starting + monitor.completed_epochs
    final = stopping_snapshot(early)
    reason = completion_reason(starting, cfg["epochs"], actual, final, policy)
    if not checkpoint.best_model_path:
        raise RuntimeError("No complete new validation checkpoint was saved")
    last = output / "checkpoints" / "last.ckpt"
    validate_checkpoint(
        torch.load(last, map_location="cpu", weights_only=False),
        actual,
        cfg["precision"],
    )
    current_rows = timing_rows(output, starting, actual)
    replayed = replay_stopping(current_rows, policy, initial)
    if (replayed["best_score"], replayed["wait_count"]) != (
        final["best_score"],
        final["wait_count"],
    ):
        raise RuntimeError("Early stopping differs from the saved loss curve")
    write_curve(output, history, current_rows)
    state, selected_new = checkpoint_record(checkpoint, torch)
    del state
    chosen = choose_checkpoint(selected_parent, selected_new)
    if chosen is selected_parent:
        copied = output / "checkpoints" / "parent-best.ckpt"
        shutil.copy2(best_parent_path, copied)
        chosen = dict(selected_parent, path=str(copied))
    selected = dict(chosen)
    if sha256(selected["path"]) != selected["sha256"]:
        raise ValueError("Selected checkpoint changed before inference")
    state = torch.load(
        selected["path"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    write_json(output / "selected_checkpoint.json", selected)
    write_json(
        output / "checkpoint_selection.json",
        {
            "rule": "minimum complete validation val_loss across parent and continuation; parent wins ties",
            "parent": selected_parent,
            "continuation": selected_new,
            "selected": selected,
        },
    )
    phases["checkpoint_selection_seconds"] = (
        time.perf_counter() - selection_started
    )
    export_started = time.perf_counter()
    validation = output / "validation"
    validation.mkdir()
    export_predictions(
        model,
        loaders["validation"],
        validation,
        selected,
        split_name="validation",
    )
    phases["validation_export_seconds"] = time.perf_counter() - export_started
    diagnostics_started = time.perf_counter()
    import pandas as pd
    from evaluate import summarize as physics_summary

    frame = pd.read_csv(validation / "predictions.csv")
    pd.DataFrame(physics_summary(frame)).to_csv(
        validation / "metrics.csv", index=False
    )
    pd.DataFrame(
        diagnostics(frame, cfg["vertex_weight"], azimuth_bins=True)
    ).to_csv(validation / "diagnostics.csv", index=False)
    phases["final_diagnostics_seconds"] = (
        time.perf_counter() - diagnostics_started
    )
    write_json(
        output / "wall_timing.json",
        {
            "started_utc": began,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "total_seconds": time.perf_counter() - wall_started,
            "phases": phases,
            "note": "Fit includes per-epoch diagnostic CPU reductions; inference manifest separately times FP32 forward export.",
        },
    )
    write_json(
        output / "early_stopping.json",
        {
            "policy": policy,
            "initial": initial,
            "final": final,
            "stop_reason": reason,
        },
    )
    write_json(
        output / "complete.json",
        {
            "complete": True,
            "epochs": actual,
            "actual_completed_epochs": actual,
            "max_total_epochs": cfg["epochs"],
            "stop_reason": reason,
            "early_stopping_initial": initial,
            "early_stopping_final": final,
            "training_curve_sha256": sha256(output / "training_curve.csv"),
            "starting_epochs": starting,
            "additional_epochs": monitor.completed_epochs,
            "parent_run": str(parent),
            "train_rows": COUNTS["train"],
            "validation_rows": len(frame),
            "test_exported": False,
            "test_rows": 0,
            "last_checkpoint": {"path": str(last), "sha256": sha256(last)},
            "artifact_hashes": {
                name: sha256(output / name)
                for name in (
                    "resolved_settings.json",
                    "provenance.json",
                    "checkpoint_source.json",
                    "database_manifest.json",
                    "input_probe.json",
                    "preflight.json",
                    "diagnostic_preflight/epoch_diagnostics.csv",
                    "timing.csv",
                    "selected_checkpoint.json",
                    "checkpoint_selection.json",
                    "epoch_diagnostics.csv",
                    "wall_timing.json",
                    "early_stopping.json",
                    "validation/prediction_manifest.json",
                    "validation/diagnostics.csv",
                )
            },
        },
    )


def summarize(run_root):
    import pandas as pd
    from full_diagnostics import diagnostics

    run_root = Path(run_root).resolve()
    root = (
        run_root
        if (run_root / "resolved_settings.json").is_file()
        else run_root / "both"
    )
    cfg = read_json(root / "resolved_settings.json")
    complete = read_json(root / "complete.json")
    if cfg.get("arm") != "both" or cfg.get("stage") != "full-continuation":
        raise ValueError("Require a completed both full continuation")
    original = copy.deepcopy(cfg)
    original.update(stage="continuation", early_stopping_patience=0)
    original.pop("early_stopping_min_delta")
    validate_settings(original, dict(complete, epochs=cfg["epochs"]))
    actual, reason = validate_completion(root, cfg, complete)
    selected = read_json(root / "selected_checkpoint.json")
    validate_export(root, dict(cfg, epochs=actual), selected)
    if Path(selected["path"]).is_file():
        checkpoint_path(root, selected)
    frame = pd.read_csv(root / "validation" / "predictions.csv")
    if (
        len(frame) != COUNTS["validation"]
        or frame.duplicated(["source_database", "event_no"]).any()
    ):
        raise ValueError("Incomplete or duplicate validation predictions")
    rows = diagnostics(frame, cfg["vertex_weight"], azimuth_bins=True)
    pd.DataFrame(rows).to_csv(
        run_root / "validation_diagnostics.csv", index=False
    )
    overall = next(
        row
        for row in rows
        if row["topology"] == "all"
        and row["energy_min_GeV"] is None
        and row["azimuth_min_deg"] is None
    )
    result = {
        "arm": "both",
        "actual_completed_epochs": actual,
        "max_total_epochs": cfg["epochs"],
        "stop_reason": reason,
        "best_epoch_1_based": selected["epoch"] + 1,
        "validation_selection_joint_loss": selected["score"],
        **overall,
    }
    pd.DataFrame([result]).to_csv(run_root / "summary.csv", index=False)
    receipt = {
        "run_root": str(run_root),
        "test_used": False,
        "automatic_extension": False,
        "result": result,
        "wall_timing": read_json(root / "wall_timing.json"),
        "loss_note": "Selection uses mixed-precision val_loss. Final prediction diagnostics use FP32 inference; epoch diagnostics reuse mixed-precision loss recorders with no extra forward.",
    }
    write_json(run_root / "full_summary.json", receipt)
    print(__import__("json").dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    for name in (
        "parent-run",
        "source-root",
        "data-root",
        "selection-root",
        "output-dir",
    ):
        fitting.add_argument(f"--{name}", required=True)
    fitting.add_argument("--epochs", type=int, default=80)
    fitting.add_argument("--patience", type=int, default=12)
    fitting.add_argument("--min-delta", type=float, default=1e-4)
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-root", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        summarize(args.run_root)


if __name__ == "__main__":
    main()
