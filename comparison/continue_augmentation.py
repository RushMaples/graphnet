"""Resume the paired baseline/both experiment; export validation only.

`epochs` is the total horizon, not the number of additional epochs. Training
always restores a full Lightning last.ckpt; selecting weights alone is never
used as a substitute for restoring optimizer, scheduler and AMP state.
"""

import argparse
import copy
import csv
import gc
import math
import os
import shutil
from pathlib import Path

from augmentation import (
    COUNTS,
    augmentation_settings,
    callbacks,
    checkpoint_record,
    identity,
    make_provenance,
    model_and_data,
    read_json,
)
from contract import (
    HERE,
    TRUTH,
    load_contract,
    load_splits,
    sha256,
    write_json,
)
from run import (
    activate_source,
    audit_databases,
    build_loaders,
    export_predictions,
    input_probe,
    numerical_preflight,
)

PAIRED_ARMS = ("baseline", "both")
STEPS_PER_EPOCH = 4517


def validate_settings(cfg, complete, target_epochs=None):
    """Validate the original frozen experiment without importing Torch."""
    arm = cfg.get("arm")
    if arm not in PAIRED_ARMS or cfg.get("stage") not in (
        "screen",
        "continuation",
    ):
        raise ValueError(
            "Require a baseline/both screen or continuation parent"
        )
    epochs = cfg.get("epochs")
    if type(epochs) is not int or epochs < 10:
        raise ValueError("Parent must have completed at least ten epochs")
    if cfg["stage"] == "screen" and epochs != 10:
        raise ValueError("The original screen parent must have ten epochs")
    expected = load_contract()
    aug = augmentation_settings(arm)
    expected.update(
        variant="ajai",
        arm=arm,
        stage=cfg["stage"],
        epochs=epochs,
        augmentation=aug,
        group_factor=16,
        drop_last=False,
        rotation_augmentation=aug["rotation"],
        heldout_policy="p14-eval-v1",
    )
    actual = copy.deepcopy(cfg)
    if cfg["stage"] == "continuation":
        starting = actual.pop("starting_epochs", None)
        parent = actual.pop("parent_run", None)
        if (
            type(starting) is not int
            or not 10 <= starting < epochs
            or not parent
        ):
            raise ValueError("Continuation parent has invalid ancestry")
    if actual != expected:
        raise ValueError("Parent settings differ from the frozen experiment")
    if (
        complete.get("epochs") != epochs
        or complete.get("train_rows") != COUNTS["train"]
        or complete.get("validation_rows") != COUNTS["validation"]
        or complete.get("test_exported") is not False
        or complete.get("test_rows") != 0
    ):
        raise ValueError("Parent is incomplete or used the test split")
    if target_epochs is not None and (
        type(target_epochs) is not int or target_epochs <= epochs
    ):
        raise ValueError(
            "Target total epochs must exceed the completed parent"
        )
    return epochs


def continuation_settings(cfg, parent_run, target_epochs):
    result = copy.deepcopy(cfg)
    result.update(
        stage="continuation",
        epochs=target_epochs,
        starting_epochs=cfg["epochs"],
        parent_run=str(Path(parent_run).resolve()),
    )
    return result


def verify_original_files(provenance, source_root, harness_root=None):
    """Every recorded file must survive byte-for-byte; additions are allowed."""
    roots = {
        "source_files": Path(source_root).resolve(),
        "harness_files": Path(harness_root or HERE.parent).resolve(),
    }
    for field, root in roots.items():
        rows = provenance.get(field)
        if not rows or len({row["path"] for row in rows}) != len(rows):
            raise ValueError(f"Missing or duplicate parent {field}")
        for row in rows:
            path = (root / row["path"]).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(
                    f"Missing/invalid original source file: {path}"
                )
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Original source file changed: {path}")


def validate_checkpoint(state, parent_epochs, precision="16-mixed"):
    """Reject weights-only, partial-epoch and inconsistent resume checkpoints."""
    if (
        state.get("epoch") != parent_epochs - 1
        or state.get("global_step") != STEPS_PER_EPOCH * parent_epochs
        or not state.get("state_dict")
        or not state.get("loops")
    ):
        raise ValueError(
            "Resume checkpoint epoch/step/model/loop state is invalid"
        )
    optimizers, schedulers = state.get("optimizer_states"), state.get(
        "lr_schedulers"
    )
    if (
        not isinstance(optimizers, list)
        or len(optimizers) != 1
        or not optimizers[0].get("state")
        or not optimizers[0].get("param_groups")
        or not isinstance(schedulers, list)
        or len(schedulers) != 1
        or not schedulers[0]
        or not {"best", "num_bad_epochs", "last_epoch"} <= schedulers[0].keys()
    ):
        raise ValueError(
            "Full optimizer and ReduceLROnPlateau state are required"
        )
    # Lightning 2.4 writes the plugin class name; retain its legacy spelling
    # because Trainer itself supports both checkpoint formats.
    scaler = state.get("MixedPrecision") or state.get(
        "native_amp_scaling_state"
    )
    if precision == "16-mixed" and not scaler:
        raise ValueError(
            "Mixed-precision resume requires the parent AMP scaler"
        )


def choose_checkpoint(parent, current):
    """Preserve the parent's best even if a new ModelCheckpoint resets state."""
    for record in (parent, current):
        if record.get("monitor") != "val_loss" or not math.isfinite(
            record["score"]
        ):
            raise ValueError(
                "Checkpoint selection requires finite validation loss"
            )
    return parent if parent["score"] <= current["score"] else current


def validate_timing(parent, cfg):
    with (Path(parent) / "timing.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    start = cfg.get("starting_epochs", 0)
    if [int(row["epoch"]) for row in rows] != list(
        range(start, cfg["epochs"])
    ):
        raise ValueError(
            "Parent timing does not prove the completed epoch horizon"
        )
    for row in rows:
        if (
            int(row["train_events"]) != COUNTS["train"]
            or int(row["validation_events"]) != COUNTS["validation"]
            or int(row["optimizer_steps"]) != STEPS_PER_EPOCH
            or not math.isfinite(float(row["val_loss"]))
        ):
            raise ValueError(
                "Parent timing has incomplete exposure or invalid loss"
            )


def validate_export(root, cfg, selected):
    root = Path(root)
    manifest = read_json(root / "validation" / "prediction_manifest.json")
    if (
        manifest.get("split") != "validation"
        or manifest.get("rows") != COUNTS["validation"]
        or manifest.get("precision") != "32-true"
        or manifest.get("evaluation_seed") != cfg["seed"]
        or manifest.get("checkpoint") != selected
        or manifest.get("predictions_sha256")
        != sha256(root / "validation" / "predictions.csv")
        or selected.get("monitor") != "val_loss"
        or not math.isfinite(selected["score"])
        or not 0 <= selected["epoch"] < cfg["epochs"]
    ):
        raise ValueError(
            "Invalid or changed parent validation export/checkpoint record"
        )


def checkpoint_path(parent, record):
    """Allow a copied run tree while still requiring the original file hash."""
    local = Path(parent) / "checkpoints" / Path(record["path"]).name
    path = local if local.is_file() else Path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(
            f"Parent selected checkpoint missing or changed: {path}"
        )
    return path.resolve()


def verify_data_identity(parent, output):
    old, new = identity(parent), identity(output)
    for field in ("environment", "databases", "input_probe"):
        if old[field] != new[field]:
            raise ValueError(f"Continuation changed parent {field}")


def train(args):
    parent = Path(args.parent_run).resolve()
    parent_cfg = read_json(parent / "resolved_settings.json")
    starting = validate_settings(
        parent_cfg, read_json(parent / "complete.json"), args.epochs
    )
    validate_timing(parent, parent_cfg)
    selected_parent = read_json(parent / "selected_checkpoint.json")
    validate_export(parent, parent_cfg, selected_parent)
    original_provenance = read_json(parent / "provenance.json")
    verify_original_files(original_provenance, args.source_root)
    best_parent_path = checkpoint_path(parent, selected_parent)
    last_parent_path = parent / "checkpoints" / "last.ckpt"
    if not last_parent_path.is_file():
        raise FileNotFoundError(
            f"Full resumable checkpoint required: {last_parent_path}"
        )
    output = Path(args.output_dir).resolve()
    if (
        output == parent
        or output in parent.parents
        or parent in output.parents
    ):
        raise ValueError(
            "Continuation output must be separate from its parent"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite artifacts: {output}")
    root, graphnet_path = activate_source(args.source_root, "ajai")
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from runtime import RunMonitor

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
    write_json(
        output / "preflight.json",
        {"passed": True, "probe_loss": loss, "separate_model": True},
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
    model.fit(
        loaders["train"],
        loaders["validation"],
        max_epochs=cfg["epochs"],
        ckpt_path=str(last_parent_path),
        early_stopping_patience=0,
        gpus=[0],
        distribution_strategy="auto",
        precision=cfg["precision"],
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        callbacks=[*callbacks(cfg), monitor, checkpoint],
        logger=CSVLogger(str(output / "logs"), name="training"),
    )
    if (
        monitor.completed_epochs != cfg["epochs"] - starting
        or not checkpoint.best_model_path
    ):
        raise RuntimeError(
            "Requested additional training horizon did not complete"
        )
    last = output / "checkpoints" / "last.ckpt"
    validate_checkpoint(
        torch.load(last, map_location="cpu", weights_only=False),
        cfg["epochs"],
        cfg["precision"],
    )
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
    validation = output / "validation"
    validation.mkdir()
    export_predictions(
        model,
        loaders["validation"],
        validation,
        selected,
        split_name="validation",
    )
    import pandas as pd
    from evaluate import summarize as physics_summary

    frame = pd.read_csv(validation / "predictions.csv")
    pd.DataFrame(physics_summary(frame)).to_csv(
        validation / "metrics.csv", index=False
    )
    write_json(
        output / "complete.json",
        {
            "epochs": cfg["epochs"],
            "starting_epochs": starting,
            "additional_epochs": monitor.completed_epochs,
            "parent_run": str(parent),
            "train_rows": COUNTS["train"],
            "validation_rows": len(frame),
            "test_exported": False,
            "test_rows": 0,
            "last_checkpoint": {"path": str(last), "sha256": sha256(last)},
        },
    )


def prediction_diagnostics(frame, vertex_weight):
    """CPU loss reconstruction from FP32 exports, with float64 reduction.

    Use the same scaled Euclidean + vMF tight/700 objective as the frozen
    model. This diagnostic is distinct from the mixed-precision selection loss.
    """
    import numpy as np
    from evaluate import event_metrics, summarize as physics_summary

    arrays = event_metrics(frame)
    direction = frame[[f"dir_{axis}_pred" for axis in "xyz"]].to_numpy()
    p = direction * frame.direction_kappa.to_numpy()[:, None]
    k = np.linalg.norm(p, axis=1)
    truth = np.column_stack(
        (
            np.sin(frame.zenith) * np.cos(frame.azimuth),
            np.sin(frame.zenith) * np.sin(frame.azimuth),
            np.cos(frame.zenith),
        )
    )
    log_c = np.empty_like(k)
    small = k < 1e-4
    log_c[small] = (
        -math.log(4 * math.pi) - k[small] ** 2 / 6 + k[small] ** 4 / 180
    )
    mid = (k >= 1e-4) & (k < 700)
    log_c[mid] = (
        np.log(k[mid])
        - k[mid]
        - np.log(-np.expm1(-2 * k[mid]))
        - math.log(2 * math.pi)
    )
    approx = lambda value: -np.sqrt(4 + value**2) + np.log(
        1 + np.sqrt(4 + value**2)
    )
    anchor = math.log(700) - 700 - math.log(2 * math.pi)
    log_c[k >= 700] = approx(k[k >= 700]) - (approx(700.0) - anchor)
    direction_loss = -log_c - (p * truth).sum(axis=1)
    position_loss = arrays["vertex_m"] / 100 * vertex_weight
    if not np.isfinite(direction_loss + position_loss).all():
        raise ValueError("Nonfinite reconstructed validation joint loss")
    track = (frame.pid.abs().to_numpy() == 14) & (
        frame.interaction_type.to_numpy() == 1
    )
    rows = physics_summary(frame)
    for row in rows:
        mask = {
            "all": np.ones(len(frame), dtype=bool),
            "track": track,
            "cascade": ~track,
        }[row["topology"]].copy()
        if row["energy_min_GeV"] is not None:
            mask &= (frame.energy.to_numpy() >= row["energy_min_GeV"]) & (
                frame.energy.to_numpy() < row["energy_max_GeV"]
            )
        for key in ("angle_deg", "vertex_m"):
            for quantile in (0.9, 0.95, 0.99):
                row[f"{key}_p{int(100 * quantile)}"] = float(
                    np.quantile(arrays[key][mask], quantile)
                )
        row["fp32_prediction_joint_loss"] = float(
            np.mean((direction_loss + position_loss)[mask])
        )
        row["direction_loss"] = float(np.mean(direction_loss[mask]))
        row["weighted_position_loss"] = float(np.mean(position_loss[mask]))
        row["angle_over_90_fraction"] = float(
            np.mean(arrays["angle_deg"][mask] > 90)
        )
        row["vertex_over_100m_fraction"] = float(
            np.mean(arrays["vertex_m"][mask] > 100)
        )
    return rows


def summarize(run_root):
    """Compare paired validation only, including topology/energy tail metrics."""
    import pandas as pd

    run_root = Path(run_root).resolve()
    reference = original_inputs = None
    results, diagnostics = [], []
    for arm in PAIRED_ARMS:
        root = run_root / arm
        cfg = read_json(root / "resolved_settings.json")
        validate_settings(cfg, read_json(root / "complete.json"))
        if cfg["stage"] != "continuation" or cfg["arm"] != arm:
            raise ValueError("Require the complete paired continuation arms")
        validate_timing(root, cfg)
        selected = read_json(root / "selected_checkpoint.json")
        validate_export(root, cfg, selected)
        if Path(selected["path"]).is_file():
            checkpoint_path(root, selected)
        current = identity(root)
        common = {
            key: value
            for key, value in cfg.items()
            if key
            not in {
                "arm",
                "augmentation",
                "rotation_augmentation",
                "parent_run",
            }
        }
        comparison = {"settings": common, **current}
        if reference is None:
            reference = comparison
        elif reference != comparison:
            raise ValueError(
                "Paired arms differ in settings, sources, environment, data or probe"
            )
        frame = pd.read_csv(root / "validation" / "predictions.csv")
        keys = ["source_database", "event_no"]
        if len(frame) != COUNTS["validation"] or frame.duplicated(keys).any():
            raise ValueError("Incomplete or duplicate validation events")
        inputs = frame.set_index(keys).sort_index()[["input_sha256", *TRUTH]]
        if original_inputs is None:
            original_inputs = inputs
        elif not original_inputs.equals(inputs):
            raise ValueError(
                "Paired validation keys, truth or retained inputs differ"
            )
        rows = prediction_diagnostics(frame, cfg["vertex_weight"])
        diagnostics.extend(dict(arm=arm, **row) for row in rows)
        overall = next(
            row
            for row in rows
            if row["topology"] == "all" and row["energy_min_GeV"] is None
        )
        results.append(
            {
                "arm": arm,
                "best_epoch_1_based": selected["epoch"] + 1,
                "validation_selection_joint_loss": selected["score"],
                **overall,
            }
        )
    receipt = {
        "run_root": str(run_root),
        "total_epochs": reference["settings"]["epochs"],
        "test_used": False,
        "automatic_extension": False,
        "loss_note": "Selection uses logged mixed-precision val_loss; fp32_prediction_joint_loss is reconstructed from FP32 inference exports using float64 CPU arithmetic.",
        "results": results,
    }
    write_json(run_root / "continuation_summary.json", receipt)
    pd.DataFrame(results).to_csv(run_root / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        run_root / "validation_diagnostics.csv", index=False
    )
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
    fitting.add_argument("--epochs", type=int, default=30)
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-root", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        summarize(args.run_root)


if __name__ == "__main__":
    main()
