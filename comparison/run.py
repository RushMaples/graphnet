"""Ten-epoch Carlos/Ajai comparison in the existing PACE environment.

Each variant runs in a fresh process with its own source checkout. Both use
the same audited inputs, loader policy, training settings and export code.
"""

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

from contract import (
    DATABASES,
    FEATURES,
    TRUTH,
    HERE,
    audit_databases,
    load_contract,
    load_splits,
    run_identity,
    sha256,
    source_manifest,
    write_json,
)


def command_output(arguments, cwd=None):
    result = subprocess.run(arguments, cwd=cwd, text=True, capture_output=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def activate_source(root, variant):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / "src"))
    if variant == "carlos":
        sys.path.insert(0, str(root / "ice_mix"))
    import graphnet

    actual = Path(graphnet.__file__).resolve()
    if root / "src" / "graphnet" not in actual.parents:
        raise RuntimeError(
            f"Wrong GraphNeT import: {actual}; expected {root}/src/graphnet"
        )
    return root, str(actual)


def build_model(variant, cfg):
    import torch
    from graphnet.models import StandardModel
    from graphnet.models.detector.icecube import IceCube86
    from graphnet.training.loss_functions import (
        EuclideanDistanceLoss,
        VonMisesFisher3DLoss,
    )

    if variant == "ajai":
        from graphnet.models.transformer.icemix import (
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
        )
        dataset_kwargs = {
            "data_representation": build_data_representation(),
            "features": FEATURES,
            "add_default_labels": False,
            "labels": {
                "direction": Direction(
                    azimuth_key="azimuth", zenith_key="zenith"
                )
            },
        }
    else:
        from graphnet.models.graphs import GraphDefinition
        from graphnet.models.graphs.nodes import IceMixNodes
        from graphnet.models.task.reconstruction import (
            JointPositionandDirectionReco,
        )
        from graphnet.training.labels import JointLabel
        from src.models.transformer import IceMix
        from src.metrics_logging import JointLossWithMetrics

        # The archived model configuration carries seven raw features; the
        # backbone encodes the first six. Preserve that original contract.
        features = FEATURES + ["pmt_area"]
        representation = GraphDefinition(
            detector=IceCube86(),
            input_feature_names=features,
            node_definition=IceMixNodes(
                input_feature_names=features,
                max_pulses=256,
                z_name="sensor_pos_z",
                hlc_name=None,
                add_ice_properties=False,
            ),
        )
        backbone = IceMix(
            hidden_dim=384,
            seq_length=128,
            depth=12,
            head_size=32,
            depth_rel=4,
            n_rel=4,
            scaled_emb=True,
            include_dynedge=False,
            n_features=6,
            maha_encoder=False,
            dropout=0.1,
            attn_drop=0.05,
            proj_drop=0.1,
            drop_path_rate=0.2,
            token_drop=0.0,
            drop_chance=1.0,
            pos_time_multiplier=4096.0,
            charge_rde_multiplier=1024.0,
            spacetime_distance_scale=18.0,
            spacetime_distance_clip_min=-4.0,
            spacetime_distance_clip_max=4.0,
            spacetime_distance_multiplier=1024.0,
            mlp_ratio=4.0,
            init_values=1.0,
            n_freq=10000.0,
        )
        task = JointPositionandDirectionReco(
            hidden_size=backbone.nb_outputs,
            target_labels=["joint_labels"],
            loss_function=JointLossWithMetrics(
                alpha=cfg["alpha"],
                position_loss=EuclideanDistanceLoss(),
                direction_loss=VonMisesFisher3DLoss(),
            ),
        )
        model = StandardModel(
            graph_definition=representation,
            backbone=backbone,
            tasks=[task],
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": cfg["lr"],
                "eps": cfg["adam_eps"],
                "weight_decay": cfg["weight_decay"],
            },
            scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={
                "mode": "min",
                "patience": cfg["scheduler_patience"],
                "factor": cfg["scheduler_factor"],
            },
            scheduler_config={"monitor": "val_loss", "frequency": 1},
        )
        dataset_kwargs = {
            "graph_definition": representation,
            "features": features,
            "labels": {
                "joint_labels": JointLabel(
                    azimuth_key="azimuth",
                    zenith_key="zenith",
                    position_keys=("position_x", "position_y", "position_z"),
                    key="joint_labels",
                )
            },
        }
    return model, dataset_kwargs


def build_loaders(args, cfg, splits, dataset_kwargs):
    from torch.utils.data import ConcatDataset
    from graphnet.data.dataloader import DataLoader
    from runtime import PilotDataset, strict_collate, worker_init

    loaders = {}
    for split, selections in splits.items():
        datasets = [
            PilotDataset(
                path=str(Path(args.data_root) / name),
                selection=selection,
                comparison_split=split,
                evaluation_seed=cfg["seed"],
                pulsemaps="SRTInIcePulses",
                truth=TRUTH,
                truth_table="truth",
                index_column="event_no",
                **dataset_kwargs,
            )
            for name, selection in zip(DATABASES, selections)
        ]
        sampler = None
        if args.variant == "ajai" and split == "train" and args.group_factor:
            from graphnet.data.dataset.samplers import LengthGroupedSampler
            from graphnet.data.utilities.lengths import event_lengths

            lengths = event_lengths(
                paths=[str(Path(args.data_root) / name) for name in DATABASES],
                selections=selections,
                pulsemap="SRTInIcePulses",
                index_column="event_no",
                cache_dir=str(Path(args.output_dir) / "length_cache"),
                max_pulses=256,
            )
            sampler = LengthGroupedSampler(
                lengths,
                batch_size=cfg["batch_size"],
                num_replicas=1,
                rank=0,
                seed=cfg["seed"],
                shuffle=True,
                group_factor=args.group_factor,
            )
        # Single-GPU; keep the final 73 training events and every validation event.
        loaders[split] = DataLoader(
            ConcatDataset(datasets),
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            drop_last=False,
            shuffle=split == "train" and sampler is None,
            sampler=sampler,
            multiprocessing_context="spawn",
            worker_init_fn=worker_init,
            collate_fn=strict_collate,
        )
    return loaders


def numerical_preflight(model, loader, variant):
    """Exercise extensions, the real model, FP16 loss/backward, and vMF zero."""
    import torch

    for name in (
        "torch_geometric",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
    ):
        importlib.import_module(name)
    from graphnet.training.loss_functions import VonMisesFisher3DLoss

    kappa = torch.tensor(
        [0.0, 1e-8, 1e-4, 1.0, 699.0, 701.0, 10000.0],
        device="cuda",
        requires_grad=True,
    )
    values = VonMisesFisher3DLoss.log_cmk(3, kappa)
    values.sum().backward()
    if (
        not torch.isfinite(values).all()
        or not torch.isfinite(kappa.grad).all()
    ):
        raise FloatingPointError(
            f"{variant}: nonfinite vMF boundary; fix before training"
        )
    model.to("cuda").train()
    batch = next(iter(loader)).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model.shared_step([batch], 0)
    loss.backward()
    if not torch.isfinite(loss) or any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        raise FloatingPointError(
            "Real-data mixed-precision forward/backward failed"
        )
    model.zero_grad(set_to_none=True)
    # Training will reconstruct the model and reseed after this probe.
    return float(loss.detach())


def physics_callback(variant):
    if variant == "ajai":
        from graphnet.training.callbacks import PhysicsMetricsCallback

        return PhysicsMetricsCallback(
            position_scale=100.0, sel_energy_min=10.0
        )
    from src.metrics_logging import PhysicsMetricsCallback

    return PhysicsMetricsCallback()


def trainer_preflight(model, loaders, output, variant):
    """Exercise the actual Trainer, callbacks, checkpoint reload and exporter."""
    import torch
    from torch.utils.data import ConcatDataset, Subset
    from graphnet.data.dataloader import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint
    from runtime import RunMonitor, strict_collate

    smoke = Path(output) / "trainer_smoke"
    smoke.mkdir()
    checkpoint = ModelCheckpoint(
        dirpath=str(smoke / "checkpoints"), monitor="val_loss", mode="min"
    )
    monitor = RunMonitor(smoke, 128, 128)
    model.fit(
        loaders["train"],
        loaders["validation"],
        max_epochs=1,
        early_stopping_patience=0,
        gpus=[0],
        distribution_strategy="auto",
        precision="16-mixed",
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        use_distributed_sampler=False,
        logger=False,
        callbacks=[physics_callback(variant), monitor, checkpoint],
    )
    if monitor.completed_epochs != 1 or not checkpoint.best_model_path:
        raise RuntimeError("Trainer/checkpoint preflight did not finish")
    state = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["state_dict"], strict=True)
    datasets = loaders["validation"].dataset.datasets
    subset_loader = DataLoader(
        ConcatDataset([Subset(d, [0, 1]) for d in datasets]),
        batch_size=6,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        collate_fn=strict_collate,
        shuffle=False,
    )
    keys = {
        (Path(d.path).name, int(d._indices[i]))
        for d in datasets
        for i in (0, 1)
    }
    export_predictions(
        model,
        subset_loader,
        smoke,
        {
            "path": checkpoint.best_model_path,
            "sha256": sha256(checkpoint.best_model_path),
            "epoch": state["epoch"],
            "monitor": "val_loss",
            "score": float(checkpoint.best_model_score),
        },
        expected_keys=keys,
        split_name="validation-smoke",
    )


def input_probe(loader, output):
    """Export a reproducibility check without inspecting test predictions."""
    records = []
    for dataset in loader.dataset.datasets:
        # Add capped events, so the probe exercises random pulse selection too.
        import sqlite3

        with closing(
            sqlite3.connect(
                Path(dataset.path).resolve().as_uri() + "?mode=ro", uri=True
            )
        ) as connection:
            capped = []
            # Restrict counts to selected validation IDs, rather than grouping
            # every pulse in the full production database.
            for start in range(0, len(dataset._indices), 500):
                chunk = dataset._indices[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT event_no FROM SRTInIcePulses WHERE event_no IN ({placeholders}) GROUP BY event_no HAVING count(*) >= 256 ORDER BY event_no"
                capped.extend(
                    row[0] for row in connection.execute(query, chunk)
                )
                if len(capped) >= 3:
                    capped = capped[:3]
                    break
        indices = sorted(
            set(range(min(3, len(dataset))))
            | {dataset._indices.index(x) for x in capped}
        )
        for index in indices:
            first, second = dataset[index], dataset[index]
            if first.input_sha256 != second.input_sha256:
                raise RuntimeError(
                    "Held-out preprocessing is not deterministic"
                )
            records.append(
                {
                    "source_database": Path(dataset.path).name,
                    "event_no": int(first.event_no),
                    "input_sha256": first.input_sha256,
                }
            )
        dataset._close_connection()
    write_json(output, records)


def export_predictions(
    model,
    loader,
    output_dir,
    checkpoint,
    expected_keys=None,
    split_name="test",
):
    """One pass pairs each prediction with its own batch's truth and key."""
    import torch
    import numpy as np

    model.to("cuda").eval()
    model.inference()
    prediction_names = [
        "position_x_pred",
        "position_y_pred",
        "position_z_pred",
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "direction_kappa",
    ]
    fields = [
        "source_database",
        "event_no",
        "input_sha256",
        "n_pulses",
        *TRUTH,
        *prediction_names,
    ]
    # Warm up model kernels before timing. Inference remains FP32.
    with torch.inference_mode():
        model(next(iter(loader)).to("cuda"))
    torch.cuda.synchronize()
    started = time.perf_counter()
    rows = 0
    seen = set()
    destination = Path(output_dir) / "predictions.csv"
    partial = destination.with_suffix(".csv.partial")
    with partial.open("w", newline="") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in loader:
            values = (
                torch.cat(model(batch.to("cuda")), dim=1).float().cpu().numpy()
            )
            if (
                values.shape != (batch.num_graphs, 7)
                or not np.isfinite(values).all()
            ):
                raise FloatingPointError(
                    "Invalid prediction shape or nonfinite values"
                )
            if (values[:, 6] < 0).any() or not np.allclose(
                np.linalg.norm(values[:, 3:6], axis=1), 1.0, atol=1e-4
            ):
                raise ValueError("Invalid direction normalization or kappa")
            attributes = {
                name: getattr(batch, name).detach().cpu().reshape(-1).tolist()
                for name in ["event_no", "n_pulses", *TRUTH]
            }
            for i, prediction in enumerate(values):
                row = {name: attributes[name][i] for name in attributes}
                row.update(
                    source_database=batch.source_database[i],
                    input_sha256=batch.input_sha256[i],
                )
                row.update(zip(prediction_names, prediction.tolist()))
                row["event_no"] = int(row["event_no"])
                key = (row["source_database"], row["event_no"])
                if key in seen:
                    raise ValueError(f"Duplicate prediction key: {key}")
                seen.add(key)
                writer.writerow(row)
                rows += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    expected = (
        expected_keys
        if expected_keys is not None
        else {
            (Path(d.path).name, int(e))
            for d in loader.dataset.datasets
            for e in d._indices
        }
    )
    if seen != expected:
        raise RuntimeError(
            "Prediction keys do not match the frozen test split"
        )
    partial.replace(destination)
    write_json(
        Path(output_dir) / "prediction_manifest.json",
        {
            "checkpoint": checkpoint,
            "selection": "minimum complete validation joint loss",
            "split": split_name,
            "rows": rows,
            "precision": "32-true",
            "evaluation_seed": 42,
            "pulse_policy": "p14-eval-v1 composite-key CPU RNG; actual six-input SHA256 in each row",
            "seconds_including_loading_transfer_and_csv": elapsed,
            "events_per_second_including_loading_transfer_and_csv": rows
            / elapsed,
            "predictions_sha256": sha256(destination),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["carlos", "ajai"], required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-factor", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--preflight-receipt",
        help="paired_preflight.json from a successful check job",
    )
    args = parser.parse_args()
    if args.group_factor < 0:
        parser.error("--group-factor must be nonnegative")
    if not args.preflight_only and not args.preflight_receipt:
        parser.error(
            "Training requires --preflight-receipt from the paired check job"
        )
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output)
    root, graphnet_path = activate_source(args.source_root, args.variant)
    import torch
    import pytorch_lightning as pl

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Comparison requires exactly one visible allocated GPU"
        )
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise RuntimeError("This benchmark is single-task/single-GPU")
    cfg = load_contract()
    splits = load_splits(args.selection_root, cfg)
    cfg["variant"] = args.variant
    cfg["group_factor"] = args.group_factor if args.variant == "ajai" else 0
    cfg["heldout_policy"] = "p14-eval-v1"
    cfg["drop_last"] = False
    write_json(output / "resolved_settings.json", cfg)
    write_json(
        output / "provenance.json",
        {
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
            "pip_freeze": command_output(
                [sys.executable, "-m", "pip", "freeze"]
            ),
            "nvidia_smi": command_output(["nvidia-smi"]),
            "harness_revision": command_output(
                ["git", "rev-parse", "HEAD"], HERE.parent
            ),
        },
    )
    pl.seed_everything(cfg["seed"], workers=True)
    model, dataset_kwargs = build_model(args.variant, cfg)
    write_json(
        output / "database_manifest.json",
        audit_databases(args.data_root, splits, dataset_kwargs["features"]),
    )
    loaders = build_loaders(args, cfg, splits, dataset_kwargs)
    input_probe(loaders["validation"], output / "input_probe.json")
    if not args.preflight_only:
        receipt = json.loads(Path(args.preflight_receipt).read_text())
        if receipt.get(args.variant) != run_identity(output):
            raise ValueError(
                "Source/settings/environment/data changed since paired preflight; rerun check"
            )
    probe_loss = numerical_preflight(
        model, loaders["validation"], args.variant
    )
    if args.preflight_only:
        trainer_preflight(model, loaders, output, args.variant)
    write_json(
        output / "preflight.json",
        {
            "passed": True,
            "probe_loss": probe_loss,
            "train_events": len(loaders["train"].dataset),
            "validation_events": len(loaders["validation"].dataset),
        },
    )
    if args.preflight_only:
        return
    # Discard probe gradients and state. Start both ten-epoch runs from scratch.
    del model
    torch.cuda.empty_cache()
    pl.seed_everything(cfg["seed"], workers=True)
    model, _ = build_model(args.variant, cfg)
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from runtime import RunMonitor

    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.8f}",
        save_top_k=1,
        save_last=True,
    )
    monitor = RunMonitor(
        output,
        len(loaders["train"].dataset),
        len(loaders["validation"].dataset),
    )
    model.save_config(str(output / "model_config.yml"))
    model.fit(
        loaders["train"],
        loaders["validation"],
        max_epochs=10,
        early_stopping_patience=0,
        gpus=[0],
        callbacks=[physics_callback(args.variant), monitor, checkpoint],
        distribution_strategy="auto",
        precision="16-mixed",
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        logger=CSVLogger(str(output / "logs"), name="training"),
    )
    if monitor.completed_epochs != 10 or not checkpoint.best_model_path:
        raise RuntimeError(
            "Run did not complete ten epochs with a valid best checkpoint"
        )
    selected = Path(checkpoint.best_model_path).resolve()
    state = torch.load(selected, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    checkpoint_record = {
        "path": str(selected),
        "sha256": sha256(selected),
        "epoch": state["epoch"],
        "monitor": "val_loss",
        "score": float(checkpoint.best_model_score),
    }
    write_json(output / "selected_checkpoint.json", checkpoint_record)
    export_predictions(model, loaders["test"], output, checkpoint_record)
    write_json(
        output / "complete.json",
        {"epochs": 10, "test_rows": len(loaders["test"].dataset)},
    )


if __name__ == "__main__":
    main()
