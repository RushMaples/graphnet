"""Matched directional-head screen on the frozen 10% P14 subset.

Training and optional exports use training/validation only. Never builds a test
loader. Run --help without GraphNeT imports; GPU preflight runs inside the job.
"""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from contract import (DATABASES, TRUTH, audit_databases, load_contract, load_splits,
                      sha256, write_json)
from run import activate_source, build_loaders, input_probe

ARMS = ("vmf", "king", "flow")
COUNTS = {"train": 578121, "validation": 72264, "test": 72268}
FLOW_SOURCE_COMMIT = "51f70764c12272715c3f47c593cccd1701d978c7"
FOLLOWUP_MODEL_API = 1


def settings(arm, flow_layers=4):
    if arm not in ARMS or flow_layers not in (4, 16):
        raise ValueError("Unknown arm or unsupported flow depth")
    cfg = load_contract()
    cfg.update(arm=arm, epochs=10, variant="ajai", group_factor=16,
               drop_last=False, heldout_policy="p14-eval-v1", flow_layers=flow_layers,
               augmentation={"token_drop": .05, "drop_chance": .05, "rotation": True},
               rotation_augmentation=True, test_used=False,
               parent_count_policy="cleaned SRTInIcePulses before cap/dropout",
               initialization="seed42 common backbone and vertex; independent head",
               flow_source_commit=FLOW_SOURCE_COMMIT)
    return cfg


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(cfg):
    import torch
    import pytorch_lightning as pl
    from augmentation import model_and_data
    from directional_heads import KingHead, SphericalFlowHead, vmf_log_prob

    # Instantiate the same original model FIRST for every arm, ensuring its
    # backbone and vertex state are identical before head-specific RNG use.
    pl.seed_everything(cfg["seed"], workers=True)
    original, dataset_kwargs = model_and_data(cfg)

    class DirectionalModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.successful_optimizer_steps = 0
            self.overflow_batches = 0
            self.backbone = original.backbone
            self.position_task = original._tasks[0]
            self.direction_task = original._tasks[1] if cfg["arm"] == "vmf" else None
            self.density_head = (KingHead() if cfg["arm"] == "king" else
                                 SphericalFlowHead(num_layers=cfg["flow_layers"])
                                 if cfg["arm"] == "flow" else None)
            if cfg.get("task", "joint") == "direction":
                self.position_task.requires_grad_(False)

        def forward(self, batch):
            normalized, raw = self.backbone(batch)
            position = (self.position_task(normalized)
                        if cfg.get("task", "joint") == "joint" else None)
            count = batch.n_pulses
            if cfg["arm"] == "king" and not cfg.get("king_parent_count", True):
                # Same King architecture/initialization; log(32/32) is zero.
                # Actual parent counts remain unchanged in data and exports.
                count = torch.full_like(count, 32)
            density = (self.direction_task(raw) if cfg["arm"] == "vmf" else
                       self.density_head(raw, count))
            return position, density

        def components(self, batch):
            position, density = self(batch)
            targets = {name: batch[name] for name in (*TRUTH, "direction")}
            truth = batch.direction.reshape(-1, 3)
            if cfg["arm"] == "vmf":
                direction = self.direction_task.compute_loss(density, targets)
                exact = -vmf_log_prob(density[:, :3], density[:, 3], truth).mean()
            else:
                direction = -density.log_prob(truth).mean()
                exact = direction
            vertex = (self.position_task.compute_loss(position, targets)
                      if position is not None else direction.new_zeros(()))
            total = vertex + direction
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite joint loss")
            error = None
            if position is not None:
                true_position = torch.stack([batch[f"position_{a}"] for a in "xyz"], -1)
                error = torch.linalg.vector_norm(position.float() * 100 - true_position, dim=-1)
            return total, vertex, direction, exact, error

        def step(self, batch, stage):
            total, vertex, direction, exact, error = self.components(batch)
            values = [("loss", total), ("direction_loss", direction), ("direction_nll", exact)]
            if error is not None:
                values.extend((("vertex_loss", vertex), ("vertex_error_m", error.mean())))
            for key, value in values:
                self.log(f"{stage}_{key}", value, on_step=False, on_epoch=True,
                         batch_size=batch.num_graphs, prog_bar=key == "loss")
            return total

        def on_save_checkpoint(self, checkpoint):
            checkpoint["directional_counters"] = {
                "successful_optimizer_steps": self.successful_optimizer_steps,
                "amp_overflow_batches": self.overflow_batches}

        def on_load_checkpoint(self, checkpoint):
            values = checkpoint.get("directional_counters")
            if values:
                self.successful_optimizer_steps = int(values["successful_optimizer_steps"])
                self.overflow_batches = int(values["amp_overflow_batches"])

        def training_step(self, batch, batch_idx):
            return self.step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self.step(batch, "val")

        def on_before_optimizer_step(self, optimizer):
            nonfinite = any(p.grad is not None and not torch.isfinite(p.grad).all()
                            for p in self.parameters())
            if nonfinite:
                # GradScaler normally skips an overflowing update and lowers
                # its scale. Do not turn that recoverable AMP event into a
                # fatal job failure; losses themselves must remain finite.
                if getattr(self.trainer.precision_plugin, "scaler", None) is None:
                    raise FloatingPointError("Nonfinite gradients without GradScaler")
                self.overflow_batches += 1

        def _updated(self, optimizer, args, kwargs):
            # The optimizer post-hook is not called for GradScaler skips.
            self.successful_optimizer_steps += 1

        def on_train_epoch_start(self):
            self._epoch_updates = self.successful_optimizer_steps
            self._epoch_overflows = self.overflow_batches

        def on_train_epoch_end(self):
            updates = self.successful_optimizer_steps - self._epoch_updates
            if updates == 0:
                raise FloatingPointError("Epoch completed without an optimizer update")
            self.log("train_successful_optimizer_steps", float(updates))
            self.log("train_amp_overflow_batches", float(self.overflow_batches - self._epoch_overflows))

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=cfg["lr"],
                        eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])
            optimizer.register_step_post_hook(self._updated)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        patience=cfg["scheduler_patience"], factor=cfg["scheduler_factor"])
            return {"optimizer": optimizer, "lr_scheduler": {
                    "scheduler": scheduler, "monitor": "val_loss", "frequency": 1}}

    model = DirectionalModel()
    initial = tensor_hash(model.backbone.state_dict())
    vertex_initial = tensor_hash(model.position_task.state_dict())
    del original
    # Equal starting global RNG after head construction; independent head sizes
    # still mean later stochastic streams need not remain identical.
    pl.seed_everything(cfg["seed"], workers=True)
    return model, dataset_kwargs, {"backbone": initial, "vertex": vertex_initial}


def check_environment():
    import torch
    import pytorch_lightning
    from directional_heads import SphericalFlowHead
    from check_imports import check_worker
    import graphnet
    source_root = Path(graphnet.__file__).resolve().parents[2]
    # Catch the previous IceMix/lmdb/PyG import failures on the login node
    # and before any allocation copies the large databases.
    check_worker("ajai", source_root)
    import scipy
    import matplotlib
    import torchdiffeq
    version = importlib.metadata.version("jammy_flows")
    if version != "1.1.0":
        raise RuntimeError(f"Require jammy_flows==1.1.0, found {version}")
    distribution = importlib.metadata.distribution("jammy_flows")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = source.get("vcs_info", {}).get("commit_id")
    if commit != FLOW_SOURCE_COMMIT:
        raise RuntimeError("Unverified jammy_flows source; install the pinned Git dependency "
                           "from comparison/requirements-directional.txt")
    # Constructor/API compatibility before database staging. No GPU required.
    head = SphericalFlowHead(num_layers=4)
    with torch.no_grad():
        truth = torch.tensor([[0.,0.,1.], [0.,0.,-1.], [1.,0.,0.],
                              [-1.,0.,0.], [0.,1.,0.], [0.,-1.,0.],
                              [1e-9,-1e-9,1.], [-1e-9,1e-9,-1.]])
        dist = head(torch.zeros(len(truth), 384), torch.full((len(truth),), 32))
        values = dist.log_prob(truth)
        samples = dist.sample(2)
        if not torch.isfinite(values).all() or samples.shape != (len(truth), 2, 3) or not torch.isfinite(samples).all():
            raise RuntimeError("Spherical flow API smoke check failed")
    result = {"torch": torch.__version__, "lightning": pytorch_lightning.__version__,
              "jammy_flows": version, "jammy_flows_source_commit": commit,
              "jammy_flows_direct_url": source, "flow_api": "passed",
              "flow_coordinate_fix": "stable-s2-atan2-v1",
              "directional_heads_sha256": sha256(Path(__file__).with_name("directional_heads.py"))}
    print(json.dumps(result))
    return result


def trainer(cfg, output, callbacks, smoke=False):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    return pl.Trainer(accelerator="gpu", devices=[0], strategy="auto",
        max_epochs=1 if smoke else 10, precision=cfg["precision"],
        accumulate_grad_batches=1, num_sanity_val_steps=0,
        limit_train_batches=8 if smoke else 1.0,
        limit_val_batches=1 if smoke else 1.0,
        use_distributed_sampler=False, callbacks=callbacks,
        logger=False if smoke else CSVLogger(str(output / "logs"), name="training"))


def train(args):
    root, graphnet_path = activate_source(args.source_root, "ajai")
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from augmentation import make_provenance
    from augmentation_callbacks import AugmentationCallback
    from runtime import RunMonitor
    environment_check = check_environment()  # same environment for all arms
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one visible GPU")
    cfg = settings(args.arm, args.flow_layers)
    splits = load_splits(args.selection_root, cfg)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.variant, args.group_factor, args.output_dir = "ajai", 16, str(output)
    write_json(output / "resolved_settings.json", cfg)
    provenance = make_provenance(root, graphnet_path, torch, pl)
    provenance["directional_environment_check"] = environment_check
    provenance["directional_requirements_sha256"] = sha256(Path(__file__).with_name("requirements-directional.txt"))
    write_json(output / "provenance.json", provenance)
    model, dataset_kwargs, initial = make_model(cfg)
    write_json(output / "initial_shared_state.json", initial)
    write_json(output / "database_manifest.json",
               audit_databases(args.data_root, splits, dataset_kwargs["features"]))
    eligible = {k: splits[k] for k in ("train", "validation")}
    loaders = build_loaders(args, cfg, eligible, dataset_kwargs)
    input_probe(loaders["validation"], output / "input_probe.json")
    smoke_root = output / "preflight"
    checkpoint = ModelCheckpoint(dirpath=str(smoke_root), monitor="val_loss", mode="min")
    probe_initial_hash = tensor_hash(model.state_dict())
    probe = trainer(cfg, smoke_root,
        [AugmentationCallback(42, True), checkpoint], smoke=True)
    probe.fit(model, loaders["train"], loaders["validation"])
    probe_updated_hash = tensor_hash(model.state_dict())
    if model.successful_optimizer_steps < 1 or probe_initial_hash == probe_updated_hash:
        raise RuntimeError("Preflight did not execute an optimizer update")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Preflight left nonfinite parameters")
    if not checkpoint.best_model_path:
        raise RuntimeError("Trainer preflight failed to select a checkpoint")
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to("cuda").eval()
    with torch.no_grad():
        batch = next(iter(loaders["validation"])).to("cuda")
        result = model.components(batch)
        if not all(torch.isfinite(v).all() for v in result):
            raise RuntimeError("Preflight checkpoint restore produced nonfinite values")
        if cfg["arm"] == "king":
            _, density = model(batch)
            density.check_quadrature()
    write_json(output / "preflight.json", {"passed": True, "restore": True,
               "train_batches": 8, "validation_batches": 1,
               "global_step": int(probe.global_step),
               "successful_optimizer_steps": model.successful_optimizer_steps,
               "amp_overflow_batches": model.overflow_batches,
               "initial_parameter_sha256": probe_initial_hash,
               "updated_parameter_sha256": probe_updated_hash})
    del model, loaders, probe, state, batch, result
    gc.collect()
    torch.cuda.empty_cache()
    model, _, repeated_initial = make_model(cfg)
    if initial != repeated_initial:
        raise RuntimeError("Preflight changed from-scratch initialization")
    loaders = build_loaders(args, cfg, eligible, dataset_kwargs)
    best = ModelCheckpoint(dirpath=str(output / "checkpoints"), monitor="val_loss",
                           mode="min", save_last=True, save_top_k=1,
                           filename="best-{epoch:02d}-{val_loss:.8f}")
    monitor = RunMonitor(output, COUNTS["train"], COUNTS["validation"])
    fitting = trainer(cfg, output, [AugmentationCallback(42, True), monitor, best])
    fitting.fit(model, loaders["train"], loaders["validation"])
    if monitor.completed_epochs != 10 or not best.best_model_path:
        raise RuntimeError("Training did not complete all ten epochs")
    state = torch.load(best.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    selected = {"path": str(Path(best.best_model_path).resolve()),
                "sha256": sha256(best.best_model_path), "epoch": int(state["epoch"]),
                "monitor": "val_loss", "score": float(best.best_model_score)}
    write_json(output / "selected_checkpoint.json", selected)
    if args.export_validation:
        export_validation(model, loaders["validation"], cfg, output, args.samples)
    write_json(output / "complete.json", {"epochs": 10, "train_rows": COUNTS["train"],
               "validation_rows": COUNTS["validation"], "test_exported": False,
               "validation_exported": args.export_validation,
               "successful_optimizer_steps": model.successful_optimizer_steps,
               "amp_overflow_batches": model.overflow_batches,
               "selected_checkpoint": selected})


def export_validation(model, loader, cfg, output, samples):
    import torch
    from directional_heads import angular_distance, unit_vector, vmf_log_prob
    destination = output / "validation"
    destination.mkdir(exist_ok=True)
    model.to("cuda").eval()
    rows, keys = 0, set()
    started = time.perf_counter()
    path = destination / "predictions.csv"
    expected_keys = {(Path(dataset.path).name, int(event))
                     for dataset in loader.dataset.datasets for event in dataset._indices}
    writer = None
    with path.open("w", newline="") as handle, torch.inference_mode():
        for batch in loader:
            batch = batch.to("cuda")
            truth = batch.direction.reshape(-1, 3)
            position, density = model(batch)
            if cfg["arm"] == "vmf":
                direction = unit_vector(density[:, :3])
                k = (density[:, :3].double() * density[:, 3, None].double()).norm(dim=-1)
                nll = -vmf_log_prob(density[:, :3], density[:, 3], truth)
                angle = angular_distance(direction, truth)
                # Exact spherical cap mass; uniform limit uses (1-cos)/2.
                z = 1 - angle.cos()
                hpd = torch.where(k < 1e-6, z / 2,
                        -torch.expm1(-k*z) / (-torch.expm1(-2*k)).clamp_min(1e-300))
                areas = {p: 2*math.pi*torch.where(k < 1e-6, torch.full_like(k, 2*p),
                        -torch.log1p(-p*(-torch.expm1(-2*k))) / k.clamp_min(1e-300))
                         for p in (.5, .68, .9)}
                radius50 = torch.acos((1-areas[.5]/(2*math.pi)).clamp(-1,1))*180/math.pi
            elif cfg["arm"] == "king":
                direction = density.direction
                density.check_quadrature()
                nll = -density.log_prob(truth)
                angle = angular_distance(direction, truth)
                hpd = density.cdf(angle)
                areas = {p: 2*math.pi*(1-density.cone_radius(p).cos()) for p in (.5, .68, .9)}
                radius50 = density.cone_radius(.5)*180/math.pi
            else:
                # Each event has its own stable RNG, independent of export batch
                # size. Process contexts one by one to make this explicit.
                raw = density.context
                estimates, masses, logs, radii, area_lists = [], [], [], [], {p: [] for p in (.5,.68,.9)}
                from contract import event_seed
                for i in range(batch.num_graphs):
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(event_seed(batch.source_database[i], int(batch.event_no[i]), 42))
                        dist = model.density_head(raw[i:i+1], batch.n_pulses[i:i+1])
                        draws, logq = dist.sample_log_prob(samples)
                        values, sample_logq = draws[0], logq[0]
                        log_truth = dist.log_prob(truth[i:i+1])[0]
                        estimate = values[sample_logq.argmax()]
                        estimates.append(estimate)
                        radii.append(torch.quantile(angular_distance(estimate.expand_as(values), values), .5))
                        masses.append((sample_logq >= log_truth).double().mean())
                        logs.append(-log_truth)
                        for p in area_lists:
                            cutoff = torch.quantile(sample_logq, 1-p)
                            # Importance estimate under posterior draws; noisy
                            # for small sample counts, recorded as MC estimate.
                            area_lists[p].append(torch.where(sample_logq >= cutoff,
                                                     (-sample_logq).exp(), 0.).mean())
                direction, hpd, nll = torch.stack(estimates), torch.stack(masses), torch.stack(logs)
                areas = {p: torch.stack(v) for p,v in area_lists.items()}
                radius50 = torch.stack(radii)*180/math.pi
                angle = angular_distance(direction, truth)
            predictions = vertex_error = None
            if position is not None:
                true_pos = torch.stack([batch[f"position_{a}"] for a in "xyz"], -1)
                predictions = position.float()*100
                vertex_error = torch.linalg.vector_norm(predictions-true_pos, dim=-1)
            for i in range(batch.num_graphs):
                key = (batch.source_database[i], int(batch.event_no[i]))
                if key in keys:
                    raise ValueError("Duplicate validation key")
                keys.add(key)
                row = {"source_database": key[0], "event_no": key[1],
                       "input_sha256": batch.input_sha256[i], "n_pulses": int(batch.n_pulses[i])}
                row.update({name: float(batch[name][i]) for name in TRUTH})
                row.update({f"direction_{a}": float(truth[i,j]) for j,a in enumerate("xyz")})
                if predictions is not None:
                    row.update({f"position_{a}_pred": float(predictions[i,j]) for j,a in enumerate("xyz")})
                row.update({f"dir_{a}_pred": float(direction[i,j]) for j,a in enumerate("xyz")})
                row.update(direction_nll=float(nll[i]), angle_deg=float(angle[i]*180/math.pi),
                           hpd_mass_at_truth=float(hpd[i]))
                if vertex_error is not None:
                    row["vertex_error_m"] = float(vertex_error[i])
                row["cone_radius_0.5_deg"] = float(radius50[i])
                row.update({f"region_area_{p}_sr": float(v[i]) for p,v in areas.items()})
                if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise FloatingPointError("Nonfinite validation export")
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                rows += 1
    if rows != COUNTS["validation"] or keys != expected_keys:
        raise RuntimeError("Incomplete validation export")
    write_json(destination / "prediction_manifest.json", {"split": "validation", "rows": rows,
        "complete": True,
        "selected_checkpoint": json.loads((output / "selected_checkpoint.json").read_text()),
        "settings_sha256": sha256(output / "resolved_settings.json"),
        "provenance_sha256": sha256(output / "provenance.json"),
        "database_manifest_sha256": sha256(output / "database_manifest.json"),
        "sha256": sha256(path), "samples_per_event": samples if cfg["arm"] == "flow" else None,
        "point_estimate": "maximum density among posterior samples" if cfg["arm"] == "flow" else "mode",
        "radius50_method": "MC median opening angle about sampled mode" if cfg["arm"] == "flow" else "exact mass cone",
        "flow_region_method": "posterior MC density-rank HPD and importance area estimate",
        "inference_including_export_seconds": time.perf_counter()-started,
        "density_measure": "per steradian", "test_used": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-environment", "check-inputs", "train"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-root")
    parser.add_argument("--selection-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--flow-layers", type=int, choices=(4,16), default=4)
    parser.add_argument("--export-validation", action="store_true")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if args.command == "check-environment":
        activate_source(args.source_root, "ajai")
        check_environment()
    elif args.command == "check-inputs":
        if not args.selection_root or not args.data_root:
            parser.error("check-inputs requires selection-root and data-root")
        splits = load_splits(args.selection_root)
        counts = {name: sum(map(len, lists)) for name, lists in splits.items()}
        if counts != COUNTS:
            raise ValueError(f"Unexpected frozen split counts: {counts}")
        missing = [name for name in DATABASES if not (Path(args.data_root) / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Source databases missing under {args.data_root}: {missing}")
        print(json.dumps({"split_counts": counts, "frozen_csv_checksums": "passed",
                          "database_paths": "present", "database_contents": "audited inside allocation"}))
    else:
        if not all((args.arm, args.data_root, args.selection_root, args.output_dir)):
            parser.error("train requires arm, data-root, selection-root and output-dir")
        if args.samples < 32:
            parser.error("At least 32 posterior samples per event")
        train(args)


if __name__ == "__main__":
    main()
