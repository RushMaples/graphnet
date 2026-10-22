"""P14 augmentation screen and longer training on the frozen 10% selection.

Screen selection uses complete, unaugmented validation only. The longer run
starts from scratch and evaluates the frozen test sample only after fitting.
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path

from contract import (
    HERE,
    audit_databases,
    environment_identity,
    load_contract,
    load_splits,
    sha256,
    source_manifest,
    write_json,
)
from run import (
    activate_source,
    build_loaders,
    command_output,
    export_predictions,
    input_probe,
    numerical_preflight,
    physics_callback,
)

ARMS = ("baseline", "pulse-dropout", "rotation", "both")
COUNTS = {"train": 578121, "validation": 72264, "test": 72268}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def augmentation_settings(arm, token_drop=0.05, drop_chance=0.05):
    if arm not in ARMS:
        raise ValueError(f"Unknown augmentation arm: {arm}")
    for value in (token_drop, drop_chance):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Drop probabilities must be finite and in [0, 1]")
    dropping = arm in ("pulse-dropout", "both")
    return {
        "token_drop": token_drop if dropping else 0.0,
        "drop_chance": drop_chance if dropping else 0.0,
        "rotation": arm in ("rotation", "both"),
        "rotation_pivot_m": [46.3, -34.9],
        "application": "training only; token thinning after 256-pulse cap",
    }


def model_and_data(cfg):
    from graphnet.models.transformer.icemix import (
        FEATURES,
        IceMixReconstruction,
        build_data_representation,
    )
    from graphnet.training.labels import Direction

    model = IceMixReconstruction(
        lr=cfg["lr"],
        adam_eps=cfg["adam_eps"],
        weight_decay=cfg["weight_decay"],
        vertex_weight=cfg["vertex_weight"],
        direction_weight=1.0,
        scheduler_patience=cfg["scheduler_patience"],
        scheduler_factor=cfg["scheduler_factor"],
        dropout=0.1,
        attn_drop=0.05,
        proj_drop=0.1,
        drop_path_rate=0.2,
        token_drop=cfg["augmentation"]["token_drop"],
        drop_chance=cfg["augmentation"]["drop_chance"],
    )
    return model, {
        "data_representation": build_data_representation(),
        "features": FEATURES,
        "add_default_labels": False,
        "labels": {
            "direction": Direction(azimuth_key="azimuth", zenith_key="zenith")
        },
    }


def callbacks(cfg):
    from augmentation_callbacks import AugmentationCallback

    return [
        AugmentationCallback(cfg["seed"], cfg["augmentation"]["rotation"]),
        physics_callback("ajai"),
    ]


def checkpoint_record(callback, torch):
    path = Path(callback.best_model_path).resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state, {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": int(state["epoch"]),
        "monitor": "val_loss",
        "score": float(callback.best_model_score),
    }


def make_provenance(root, graphnet_path, torch, pl):
    return {
        "source_root": str(root),
        "graphnet_import": graphnet_path,
        "revision": command_output(["git", "rev-parse", "HEAD"], root),
        "status": command_output(["git", "status", "--porcelain"], root),
        "source_files": source_manifest(root),
        "harness_files": source_manifest(HERE.parent),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "lightning": pl.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
    }


def comparable_settings(cfg):
    return {
        k: v
        for k, v in cfg.items()
        if k not in {"arm", "augmentation", "rotation_augmentation"}
    }


def identity(root):
    provenance = read_json(root / "provenance.json")
    return {
        "source_files": provenance["source_files"],
        "harness_files": provenance["harness_files"],
        "environment": environment_identity(provenance),
        "databases": {
            row["database"]: row["sha256"]
            for row in read_json(root / "database_manifest.json")
        },
        "input_probe": read_json(root / "input_probe.json"),
    }


def summarize(screen_root):
    """Validate all arms before choosing minimum validation joint loss."""
    screen_root = Path(screen_root).resolve()
    rows = []
    reference = common_identity = validation_inputs = drop_rates = None
    for arm in ARMS:
        root = screen_root / arm
        complete = read_json(root / "complete.json")
        cfg = read_json(root / "resolved_settings.json")
        checkpoint = read_json(root / "selected_checkpoint.json")
        manifest = read_json(root / "validation" / "prediction_manifest.json")
        if (
            complete["epochs"] != cfg["epochs"]
            or complete["validation_rows"] != COUNTS["validation"]
            or complete["test_exported"]
            or cfg["stage"] != "screen"
            or cfg["arm"] != arm
            or manifest["split"] != "validation"
            or manifest["rows"] != COUNTS["validation"]
            or manifest["checkpoint"] != checkpoint
            or sha256(root / "validation" / "predictions.csv")
            != manifest["predictions_sha256"]
            or sha256(checkpoint["path"]) != checkpoint["sha256"]
            or not math.isfinite(checkpoint["score"])
        ):
            raise ValueError(f"Incomplete or changed screen artifacts: {root}")
        common = comparable_settings(cfg)
        aug = cfg["augmentation"]
        if aug != augmentation_settings(
            arm, aug["token_drop"], aug["drop_chance"]
        ):
            raise ValueError(
                f"Augmentation settings do not describe arm {arm}"
            )
        if arm in ("pulse-dropout", "both"):
            rates = (aug["token_drop"], aug["drop_chance"])
            if min(rates) <= 0:
                raise ValueError("Dropout arms require nonzero probabilities")
            if drop_rates is None:
                drop_rates = rates
            elif drop_rates != rates:
                raise ValueError(
                    "Pulse dropout rates differ between enabled arms"
                )
        current_identity = identity(root)
        if reference is None:
            reference, common_identity = common, current_identity
        if reference != common or common_identity != current_identity:
            raise ValueError("Screen arms differ beyond augmentation settings")
        import pandas as pd
        from contract import TRUTH

        keys = ["source_database", "event_no"]
        frame = pd.read_csv(root / "validation" / "predictions.csv")
        if len(frame) != COUNTS["validation"] or frame.duplicated(keys).any():
            raise ValueError("Incomplete or duplicate validation events")
        actual_inputs = frame.set_index(keys).sort_index()[
            ["input_sha256", *TRUTH]
        ]
        if validation_inputs is None:
            validation_inputs = actual_inputs
        elif not validation_inputs.equals(actual_inputs):
            raise ValueError(
                "Validation event keys, truth or retained pulses differ"
            )
        with (root / "validation" / "metrics.csv").open(newline="") as handle:
            overall = next(
                row
                for row in csv.DictReader(handle)
                if row["topology"] == "all" and row["energy_min_GeV"] == ""
            )
        rows.append(
            {
                "arm": arm,
                "best_epoch_1_based": checkpoint["epoch"] + 1,
                "validation_joint_loss": checkpoint["score"],
                "angle_deg_median": float(overall["angle_deg_median"]),
                "vertex_m_median": float(overall["vertex_m_median"]),
                "coverage_0.68": float(overall["coverage_0.68"]),
                "augmentation": cfg["augmentation"],
            }
        )
    selected = min(rows, key=lambda row: row["validation_joint_loss"])
    receipt = {
        "selection_rule": "minimum complete unaugmented validation joint loss; ties use ARMS order",
        "selected_arm": selected["arm"],
        "augmentation": selected["augmentation"],
        "screen_root": str(screen_root),
        "common_settings": reference,
        "identity": common_identity,
        "results": rows,
        "test_used_for_selection": False,
    }
    write_json(screen_root / "selected_arm.json", receipt)
    with (screen_root / "summary.csv").open("w", newline="") as handle:
        fields = [key for key in rows[0] if key != "augmentation"]
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {"selected_arm": selected["arm"], "results": rows}, indent=2
        )
    )
    return receipt


def train(args):
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
    cfg = load_contract()
    splits = load_splits(args.selection_root, cfg)
    selection = None
    if args.stage == "full":
        selection = read_json(Path(args.screen_root) / "selected_arm.json")
        # Recheck files rather than trusting an editable selection receipt.
        checked = summarize(args.screen_root)
        if selection != checked:
            raise ValueError(
                "Screen selection changed; inspect the new summary"
            )
        args.arm = selection["selected_arm"]
        augmentation = selection["augmentation"]
    else:
        augmentation = augmentation_settings(
            args.arm, args.token_drop, args.drop_chance
        )
    cfg.update(
        variant="ajai",
        arm=args.arm,
        stage=args.stage,
        epochs=args.epochs,
        augmentation=augmentation,
        group_factor=16,
        drop_last=False,
        rotation_augmentation=augmentation["rotation"],
        heldout_policy="p14-eval-v1",
    )
    if selection:
        expected = dict(selection["common_settings"])
        actual = comparable_settings(cfg)
        for key in ("stage", "epochs", "rotation_augmentation"):
            expected.pop(key, None)
            actual.pop(key, None)
        if expected != actual:
            raise ValueError(
                "Long-run settings differ from screen beyond horizon"
            )
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir, args.variant, args.group_factor = str(output), "ajai", 16
    write_json(output / "resolved_settings.json", cfg)
    write_json(
        output / "provenance.json",
        make_provenance(root, graphnet_path, torch, pl),
    )
    pl.seed_everything(cfg["seed"], workers=True)
    model, dataset_kwargs = model_and_data(cfg)
    write_json(
        output / "database_manifest.json",
        audit_databases(args.data_root, splits, dataset_kwargs["features"]),
    )
    # Never even build a test loader during the augmentation screen.
    loader_splits = {
        k: v for k, v in splits.items() if args.stage == "full" or k != "test"
    }
    loaders = build_loaders(args, cfg, loader_splits, dataset_kwargs)
    input_probe(loaders["validation"], output / "input_probe.json")
    if selection and identity(output) != selection["identity"]:
        raise ValueError(
            "Sources, environment, data or held-out inputs changed since screen"
        )
    loss = numerical_preflight(model, loaders["validation"], "ajai")
    smoke = output / "trainer_smoke"
    smoke.mkdir()
    smoke_checkpoint = ModelCheckpoint(
        dirpath=str(smoke / "checkpoints"), monitor="val_loss", mode="min"
    )
    smoke_monitor = RunMonitor(smoke, cfg["batch_size"], cfg["batch_size"])
    model.fit(
        loaders["train"],
        loaders["validation"],
        max_epochs=1,
        early_stopping_patience=0,
        gpus=[0],
        distribution_strategy="auto",
        precision=cfg["precision"],
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        use_distributed_sampler=False,
        logger=False,
        callbacks=[*callbacks(cfg), smoke_monitor, smoke_checkpoint],
    )
    if (
        smoke_monitor.completed_epochs != 1
        or not smoke_checkpoint.best_model_path
    ):
        raise RuntimeError("Inline augmentation/Trainer preflight failed")
    # Exercise config reconstruction, checkpoint loading and truth-free export
    # shape before spending the full training allocation.
    from graphnet.models.transformer.icemix import load_icemix_model

    smoke_config = smoke / "model_config.yml"
    model.save_config(str(smoke_config))
    restored = load_icemix_model(str(smoke_config))
    if (restored.backbone.token_drop, restored.backbone.drop_chance) != (
        augmentation["token_drop"],
        augmentation["drop_chance"],
    ):
        raise RuntimeError("Augmentation configuration did not round-trip")
    smoke_state, _ = checkpoint_record(smoke_checkpoint, torch)
    restored.load_state_dict(smoke_state["state_dict"], strict=True)
    restored.to("cuda").eval()
    restored.inference()
    with torch.inference_mode():
        sample = next(iter(loaders["validation"])).to("cuda")
        prediction = torch.cat(restored(sample), dim=1)
        if (
            prediction.shape != (sample.num_graphs, 7)
            or not torch.isfinite(prediction).all()
        ):
            raise RuntimeError("Inline inference output check failed")
    del restored, smoke_state, sample, prediction
    write_json(
        output / "preflight.json",
        {"passed": True, "probe_loss": loss, "augmentation": augmentation},
    )
    del model
    # Rebuild loaders too: worker RNGs used by smoke must not leak into fitting.
    del loaders
    gc.collect()
    torch.cuda.empty_cache()
    pl.seed_everything(cfg["seed"], workers=True)
    loaders = build_loaders(args, cfg, loader_splits, dataset_kwargs)
    model, _ = model_and_data(cfg)
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.8f}",
        save_top_k=1,
        save_last=True,
    )
    monitor = RunMonitor(output, COUNTS["train"], COUNTS["validation"])
    model.save_config(str(output / "model_config.yml"))
    model.fit(
        loaders["train"],
        loaders["validation"],
        max_epochs=cfg["epochs"],
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
        monitor.completed_epochs != cfg["epochs"]
        or not checkpoint.best_model_path
    ):
        raise RuntimeError("Requested training horizon did not complete")
    state, selected = checkpoint_record(checkpoint, torch)
    model.load_state_dict(state["state_dict"], strict=True)
    write_json(output / "selected_checkpoint.json", selected)
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
    if args.stage == "full":
        export_predictions(model, loaders["test"], output, selected)
        pd.DataFrame(
            physics_summary(pd.read_csv(output / "predictions.csv"))
        ).to_csv(output / "test_metrics.csv", index=False)
        write_json(output / "screen_selection.json", selection)
    write_json(
        output / "complete.json",
        {
            "epochs": cfg["epochs"],
            "train_rows": COUNTS["train"],
            "validation_rows": len(frame),
            "test_exported": args.stage == "full",
            "test_rows": COUNTS["test"] if args.stage == "full" else 0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--screen-root", required=True)
    fitting = subparsers.add_parser("train")
    fitting.add_argument("--stage", choices=("screen", "full"), required=True)
    fitting.add_argument("--arm", choices=ARMS, default="baseline")
    fitting.add_argument("--epochs", type=int, required=True)
    fitting.add_argument("--source-root", required=True)
    fitting.add_argument("--data-root", required=True)
    fitting.add_argument("--selection-root", required=True)
    fitting.add_argument("--output-dir", required=True)
    fitting.add_argument("--screen-root")
    fitting.add_argument("--token-drop", type=float, default=0.05)
    fitting.add_argument("--drop-chance", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.screen_root)
    else:
        if args.epochs < 1 or (args.stage == "full" and not args.screen_root):
            parser.error(
                "Positive epochs required; full stage also needs --screen-root"
            )
        train(args)


if __name__ == "__main__":
    main()
