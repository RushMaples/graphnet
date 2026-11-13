"""Five-run follow-up to screen-15407679. Train/validation only, seed 42.

The old screen remains immutable. Continuations restore Lightning last.ckpt,
including optimizer, scheduler, loops and AMP; new objectives start afresh.
"""
import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

from contract import (DATABASES, environment_identity, load_splits, sha256,
                      source_manifest, write_json)
from directional_run import (COUNTS, FOLLOWUP_MODEL_API, check_environment, export_validation,
                             make_model, settings as screen_settings, tensor_hash)

TASKS = (
    ("joint_vmf", "vmf", "joint", True),
    ("joint_king_no_count", "king", "joint", False),
    ("joint_flow", "flow", "joint", True),
    ("direction_vmf", "vmf", "direction", False),
    ("direction_flow", "flow", "direction", False),
)
TASK_NAMES = tuple(row[0] for row in TASKS)
PATIENCE, MIN_DELTA, MAX_EPOCHS, SAMPLES = 12, 1e-4, 50, 128


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    write_json(temporary, value)
    temporary.replace(path)


def settings(name):
    if name not in TASK_NAMES:
        raise ValueError("Unknown follow-up task")
    _, arm, task, resume = TASKS[TASK_NAMES.index(name)]
    cfg = screen_settings(arm)
    cfg.update(study="directional-followup-v1", run_name=name, task=task,
               king_parent_count=False, epochs=MAX_EPOCHS,
               starting_epochs=10 if resume else 0,
               early_stopping_patience=PATIENCE, early_stopping_min_delta=MIN_DELTA,
               selection_monitor="val_loss" if task == "joint" else "val_direction_nll",
               scheduler_monitor="val_loss", posterior_samples=SAMPLES,
               snapshots=[10, 30], additional_training_seeds=False,
               sampling_convergence_study=False)
    return cfg


def validation_history(folder):
    paths = list((Path(folder) / "logs" / "training").glob("version_*/metrics.csv"))
    if len(paths) != 1:
        raise ValueError("Require one unambiguous parent metrics.csv")
    by_epoch = {}
    with paths[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"]))
            values = by_epoch.setdefault(epoch, {"epoch": epoch})
            for key in ("val_loss", "val_direction_nll"):
                if row.get(key):
                    values[key] = float(row[key])
    result = [by_epoch[key] for key in sorted(by_epoch)]
    if [row["epoch"] for row in result] != list(range(10)):
        raise ValueError("Parent history must contain ten complete epochs")
    if any(set(row) != {"epoch", "val_loss", "val_direction_nll"}
           or not all(math.isfinite(row[key]) for key in ("val_loss", "val_direction_nll"))
           for row in result):
        raise ValueError("Missing/nonfinite parent validation metrics")
    return result


def stopping_state(rows, monitor):
    best, wait = None, 0
    for row in rows:
        best, wait = advance_stopping(float(row[monitor]), best, wait)
    return {"best_score": best, "wait_count": wait}


def advance_stopping(score, best, wait):
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite validation stopping score")
    if best is None or score + MIN_DELTA < best:
        return score, 0
    return best, wait + 1


def screen_folder(screen, arm):
    return Path(screen) / arm / "model"


def selected_file(folder, selected):
    path = Path(folder) / "checkpoints" / Path(selected["path"]).name
    if not path.is_file() or sha256(path) != selected["sha256"]:
        raise ValueError("Missing/changed selected checkpoint: " + str(path))
    return path


def check_screen(screen, source_root=None, require_weights=True):
    """Verify archived receipts without importing Torch or scanning databases."""
    screen = Path(screen).resolve()
    parents, common = {}, None
    for arm in ("vmf", "king", "flow"):
        folder = screen_folder(screen, arm)
        cfg, complete = read(folder / "resolved_settings.json"), read(folder / "complete.json")
        selected = read(folder / "selected_checkpoint.json")
        manifest = read(folder / "validation" / "prediction_manifest.json")
        provenance = read(folder / "provenance.json")
        if cfg != screen_settings(arm) or complete.get("epochs") != 10:
            raise ValueError("Parent settings differ from the completed screen: " + arm)
        if (complete.get("train_rows") != COUNTS["train"]
                or complete.get("validation_rows") != COUNTS["validation"]
                or complete.get("test_exported") is not False
                or not complete.get("validation_exported") or not manifest.get("complete")
                or manifest.get("split") != "validation" or manifest.get("rows") != COUNTS["validation"]
                or manifest.get("test_used") is not False
                or manifest.get("selected_checkpoint") != selected
                or complete.get("selected_checkpoint") != selected
                or manifest["sha256"] != sha256(folder / "validation" / "predictions.csv")):
            raise ValueError("Invalid/incomplete screen export: " + arm)
        for kind, filename in (("settings", "resolved_settings.json"), ("provenance", "provenance.json"),
                               ("database_manifest", "database_manifest.json")):
            if manifest[kind + "_sha256"] != sha256(folder / filename):
                raise ValueError("Changed parent receipt: " + filename)
        identity = {"cfg": {k: v for k, v in cfg.items() if k != "arm"},
                    "initial": read(folder / "initial_shared_state.json"),
                    "source": provenance["source_files"], "harness": provenance["harness_files"],
                    "environment": environment_identity(provenance),
                    "databases": {r["database"]: r["sha256"] for r in read(folder / "database_manifest.json")},
                    "probe": read(folder / "input_probe.json")}
        if common is not None and common != identity:
            raise ValueError("The original screen arms do not share one protocol")
        common = identity
        history = validation_history(folder)
        best_direction = min(history, key=lambda r: r["val_direction_nll"])
        if best_direction["epoch"] != selected["epoch"]:
            raise ValueError("Parent direction-selected weights were not retained: " + arm)
        record = {"folder": str(folder), "selected": selected, "history": history,
                  "best_direction_score": best_direction["val_direction_nll"],
                  "receipts": {filename: sha256(folder / filename) for filename in (
                      "resolved_settings.json", "complete.json", "provenance.json",
                      "database_manifest.json", "initial_shared_state.json", "input_probe.json",
                      "selected_checkpoint.json", "validation/prediction_manifest.json",
                      "validation/predictions.csv")}}
        if require_weights and arm in ("vmf", "flow"):
            best_path = selected_file(folder, selected)
            last = folder / "checkpoints" / "last.ckpt"
            if not last.is_file():
                raise FileNotFoundError("Full resume checkpoint required: " + str(last))
            record.update(best_path=str(best_path), last_path=str(last), last_sha256=sha256(last))
        parents[arm] = record
    if source_root is not None:
        records = {r["path"]: r["sha256"] for r in (*common["source"], *common["harness"])}
        for relative, expected in records.items():
            path = Path(source_root) / relative
            # The default-compatible task/count/export extension is intentional.
            if relative in {"comparison/directional_run.py", "comparison/pace_directional.sbatch",
                            "comparison/submit_directional.sh"}:
                continue
            if not path.is_file() or sha256(path) != expected:
                raise ValueError("Original model/data source changed: " + str(path))
    return parents, common


def prepare(args):
    if FOLLOWUP_MODEL_API != 1:
        raise ValueError("Upload the complete follow-up kit, including directional_run.py")
    source = Path(args.source_root).resolve()
    parents, common = check_screen(args.screen_root, source)
    splits = load_splits(args.selection_root)
    if {key: sum(map(len, values)) for key, values in splits.items()} != COUNTS:
        raise ValueError("Unexpected selection counts")
    for database in DATABASES:
        if not (Path(args.data_root) / database).is_file():
            raise FileNotFoundError(database)
    historical = Path(args.historical_root).resolve()
    old_csv = historical / "validation" / "predictions.csv"
    if not old_csv.is_file():
        raise FileNotFoundError("Historical validation export required: " + str(old_csv))
    historical_complete = read(historical / "complete.json")
    historical_selected = read(historical / "selected_checkpoint.json")
    if (historical_complete.get("test_exported") is not False
            or historical_complete.get("validation_rows") != COUNTS["validation"]
            or not 0 <= historical_selected["epoch"] < historical_complete["epochs"]):
        raise ValueError("Historical vMF completion/checkpoint receipt is inconsistent")
    original_source = source
    if args.plan:
        frozen = Path(args.plan).resolve().parent / "source"
        if frozen.exists():
            raise FileExistsError("Refusing to overwrite frozen source")
        if any((source / name).resolve() in frozen.parents for name in ("src", "comparison", "ice_mix")):
            raise ValueError("Submission output must be outside the source directories being copied")
        expected = source_manifest(source)
        for name in ("src", "comparison"):
            shutil.copytree(source / name, frozen / name,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        # Geometry/config resources remain the same read-only inputs. Code is
        # copied, so subsequent edits to the working checkout cannot mix runs.
        for name in ("data", "assets", "configs", "ice_mix"):
            if (source / name).exists():
                os.symlink(str(source / name), str(frozen / name), target_is_directory=True)
        for name in ("pyproject.toml", "setup.cfg", "setup.py"):
            if (source / name).is_file():
                shutil.copy2(source / name, frozen / name)
        if source_manifest(frozen) != expected:
            raise ValueError("Source changed while freezing the submission")
        source = frozen
    plan = {"protocol": "directional-followup-v1", "source_root": str(source),
            "source_checkout": str(original_source),
            "screen_root": str(Path(args.screen_root).resolve()), "parents": parents,
            "historical_root": str(historical), "historical_predictions_sha256": sha256(old_csv),
            "historical_epochs": historical_complete["epochs"],
            "historical_selected_epoch": historical_selected["epoch"]+1,
            "historical_receipts": {name: sha256(historical / name) for name in ("complete.json", "selected_checkpoint.json")},
            "common_initialization": common["initial"], "parent_environment": common["environment"],
            "parent_databases": common["databases"], "parent_input_probe": common["probe"],
            "source_files": source_manifest(source),
            "runs": {name: settings(name) for name in TASK_NAMES},
            "seed": 42, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
            "posterior_samples": SAMPLES, "test_used": False,
            "deferred": ["sampling convergence study", "additional training seeds"],
            "continuation_rng_limit": "Original checkpoints do not record all RNG states; bitwise continuous-run equivalence is not claimed."}
    if args.plan:
        destination = Path(args.plan)
        if destination.exists():
            raise FileExistsError("Refusing to overwrite submission plan")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, plan)
    print(json.dumps({"preflight": "passed", "runs": list(TASK_NAMES), "max_epochs": MAX_EPOCHS,
                      "validation_rows": COUNTS["validation"], "test_used": False,
                      "plan": args.plan}, indent=2))


def verify_plan(plan, name=None):
    if (plan.get("protocol") != "directional-followup-v1" or plan.get("test_used") is not False
            or plan.get("runs") != {n: settings(n) for n in TASK_NAMES}):
        raise ValueError("Submission plan is not the agreed five-run experiment")
    for row in plan["source_files"]:
        path = Path(plan["source_root"]) / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("Source changed after submission: " + str(path))
    for arm, parent in plan["parents"].items():
        for filename, expected in parent["receipts"].items():
            path = Path(parent["folder"]) / filename
            if not path.is_file() or sha256(path) != expected:
                raise ValueError("Parent artifact changed after submission: " + str(path))
    if name in ("joint_vmf", "joint_flow"):
        arm = settings(name)["arm"]
        parent = plan["parents"][arm]
        if sha256(parent["last_path"]) != parent["last_sha256"]:
            raise ValueError("Resume checkpoint changed after submission")


def checkpoint_record(callback):
    import re
    path = Path(callback.best_model_path)
    match = re.search(r"epoch=(\d+)", path.name)
    if not path.is_file() or match is None or callback.best_model_score is None:
        raise ValueError("Missing selected checkpoint/epoch")
    return {"path": str(path.resolve()), "sha256": sha256(path), "epoch": int(match.group(1)),
            "monitor": callback.monitor, "score": float(callback.best_model_score)}


def seal_last_checkpoint(trainer, path, completed):
    """Save after Lightning's final plateau-scheduler update, atomically.

    Native epoch-end ModelCheckpoint precedes that update in Lightning 2.4.
    Normalize the public epoch metadata to the last completed zero-based epoch;
    the complete Lightning loop progress is retained unchanged.
    """
    import torch
    path = Path(path)
    temporary = path.with_name("last-post-fit.partial.ckpt")
    trainer.save_checkpoint(str(temporary))
    state = torch.load(temporary, map_location="cpu", weights_only=False)
    state["epoch"] = completed-1
    state["directional_completed_epochs"] = completed
    state["directional_last_checkpoint_phase"] = "after fit and final scheduler update"
    torch.save(state, temporary)
    del state
    temporary.replace(path)


def make_callbacks(cfg, output, parent=None):
    import torch
    from pytorch_lightning import Callback
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from augmentation_callbacks import AugmentationCallback
    from runtime import RunMonitor

    output = Path(output)
    history = copy.deepcopy(parent["history"]) if parent else []
    monitor_name = cfg["selection_monitor"]
    initial_stop = stopping_state(history, monitor_name)

    class StudyEarlyStopping(EarlyStopping):
        @property
        def state_key(self):
            return "DirectionalFollowupEarlyStopping:" + monitor_name

        def _evaluate_stopping_criteria(self, current):
            best, wait = advance_stopping(float(current), float(self.best_score), self.wait_count)
            self.best_score = torch.tensor(best, dtype=torch.float64)
            self.wait_count = wait
            return wait >= self.patience, "Follow-up stopping patience exhausted"

    stopping = StudyEarlyStopping(monitor=monitor_name, mode="min", patience=PATIENCE,
                                 min_delta=MIN_DELTA, strict=True, check_finite=True,
                                 check_on_train_epoch_end=True)
    if initial_stop["best_score"] is not None:
        stopping.best_score = torch.tensor(initial_stop["best_score"], dtype=torch.float64)
        stopping.wait_count = initial_stop["wait_count"]

    class StudyCheckpoint(ModelCheckpoint):
        @property
        def state_key(self):
            # Do not let a differently rooted legacy ModelCheckpoint overwrite
            # the explicitly carried best-so-far state or delete old artifacts.
            return "DirectionalFollowupCheckpoint:" + self.monitor

    objective = StudyCheckpoint(dirpath=str(output / "checkpoints"), monitor=monitor_name,
                                mode="min", save_top_k=1, save_last=True,
                                save_on_train_epoch_end=True,
                                filename="objective-{epoch:02d}-{" + monitor_name + ":.8f}")
    direction = objective if cfg["task"] == "direction" else StudyCheckpoint(
        dirpath=str(output / "checkpoints"), monitor="val_direction_nll", mode="min",
        save_top_k=1, save_last=False, save_weights_only=True, save_on_train_epoch_end=True,
        filename="direction-{epoch:02d}-{val_direction_nll:.8f}")
    checkpoints = {"objective": objective, "direction": direction}
    if parent:
        for label, callback in checkpoints.items():
            target = output / "checkpoints" / (label + "-parent-epoch=%02d.ckpt" % parent["selected"]["epoch"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent["best_path"], target)
            score = (parent["selected"]["score"] if label == "objective" else parent["best_direction_score"])
            value = torch.tensor(score, dtype=torch.float64)
            callback.best_model_path = callback.kth_best_model_path = str(target)
            callback.best_model_score = callback.kth_value = value
            callback.best_k_models = {str(target): value}

    class Snapshots(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            completed = trainer.current_epoch
            if completed in cfg["snapshots"]:
                self.snapshot("epoch%d" % completed, completed)

        def on_train_epoch_end(self, trainer, pl_module):
            row = {"epoch": trainer.current_epoch}
            for key in ("val_loss", "val_direction_nll"):
                row[key] = float(trainer.callback_metrics[key])
                if not math.isfinite(row[key]):
                    raise FloatingPointError("Nonfinite full validation metric")
            history.append(row)
            atomic_json(output / "validation_history.json", history)

        def snapshot(self, tag, completed):
            folder = output / "snapshots" / tag
            folder.mkdir(parents=True, exist_ok=True)
            records = {}
            for label, callback in checkpoints.items():
                record = checkpoint_record(callback)
                target = folder / (label + ".ckpt")
                if target.exists():
                    raise FileExistsError("Snapshot already exists: " + str(target))
                shutil.copy2(record["path"], target)
                record["path"] = str(target.resolve())
                records[label] = record
            atomic_json(folder / "selection.json", {"budget_epochs": completed, "checkpoints": records})

    snapshots = Snapshots()
    monitor = RunMonitor(output, COUNTS["train"], COUNTS["validation"])
    callbacks = [AugmentationCallback(42, True), monitor, snapshots, stopping, objective]
    if parent:
        class CompleteParentScheduler(Callback):
            def on_train_start(self, trainer, pl_module):
                scheduler = trainer.lr_scheduler_configs[0].scheduler
                completed = cfg["starting_epochs"]
                before = int(scheduler.last_epoch)
                if before == completed-1:
                    scheduler.step(parent["history"][-1]["val_loss"])
                elif before != completed:
                    raise ValueError("Parent scheduler progress is inconsistent with its completed epochs")
                atomic_json(output / "scheduler_restore.json", {
                    "restored_last_epoch": before, "completed_parent_epochs": completed,
                    "replayed_pending_validation_check": before == completed-1,
                    "last_epoch_after_reconciliation": int(scheduler.last_epoch)})
        callbacks.insert(0, CompleteParentScheduler())
    if direction is not objective:
        callbacks.append(direction)
    return callbacks, monitor, snapshots, checkpoints, stopping


def parent_parity(model, loader, cfg, parent):
    """Check restored selected weights against actual archived FP32 predictions."""
    import torch
    from directional_heads import vmf_log_prob
    state = torch.load(parent["best_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.to("cuda").eval()
    batch = next(iter(loader)).to("cuda")
    with torch.inference_mode():
        position, density = model(batch)
        truth = batch.direction.reshape(-1, 3)
        nll = (-vmf_log_prob(density[:, :3], density[:, 3], truth)
               if cfg["arm"] == "vmf" else -density.log_prob(truth))
    keys = {(batch.source_database[i], int(batch.event_no[i])): i for i in range(batch.num_graphs)}
    found, max_nll, max_position = set(), 0., 0.
    csv_path = Path(parent["folder"]) / "validation" / "predictions.csv"
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_database"], int(row["event_no"]))
            if key not in keys:
                continue
            i = keys[key]
            if row["input_sha256"] != batch.input_sha256[i]:
                raise ValueError("Parent parity input fingerprint changed")
            expected = torch.tensor(float(row["direction_nll"]), dtype=torch.float64, device="cuda")
            torch.testing.assert_close(nll[i], expected, rtol=1e-4, atol=1e-5)
            max_nll = max(max_nll, abs(float(nll[i])-float(expected)))
            predicted = position[i].float()*100
            expected_pos = torch.tensor([float(row["position_"+a+"_pred"]) for a in "xyz"], device="cuda")
            torch.testing.assert_close(predicted, expected_pos, rtol=1e-4, atol=1e-3)
            max_position = max(max_position, float((predicted-expected_pos).abs().max()))
            found.add(key)
    if found != set(keys):
        raise ValueError("Incomplete parent prediction parity check")
    return {"passed": True, "events": len(found), "max_nll_absolute_difference": max_nll,
            "max_position_coordinate_difference_m": max_position}


def export_snapshot(model, loader, cfg, output, tag, criterion):
    snapshot = read(Path(output) / "snapshots" / tag / "selection.json")
    selected = snapshot["checkpoints"][criterion]
    destination = Path(output) / "exports" / tag / criterion
    if destination.exists():
        raise FileExistsError("Refusing to overwrite export: " + str(destination))
    destination.mkdir(parents=True)
    import torch
    if sha256(selected["path"]) != selected["sha256"]:
        raise ValueError("Snapshot changed before inference")
    state = torch.load(selected["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    for filename in ("resolved_settings.json", "provenance.json", "database_manifest.json"):
        shutil.copy2(Path(output) / filename, destination / filename)
    write_json(destination / "selected_checkpoint.json", selected)
    write_json(destination / "export_protocol.json", {
        "run_name": cfg["run_name"], "task": cfg["task"], "tag": tag,
        "budget_epochs": snapshot["budget_epochs"], "criterion": criterion,
        "test_used": False, "vertex_predictions": cfg["task"] == "joint"})
    export_validation(model, loader, cfg, destination, SAMPLES)
    manifest = read(destination / "validation" / "prediction_manifest.json")
    return {"folder": str(destination.resolve()), "manifest_sha256": sha256(destination / "validation" / "prediction_manifest.json"),
            "prediction_sha256": manifest["sha256"], "criterion": criterion,
            "budget_epochs": snapshot["budget_epochs"], "selected_checkpoint": selected,
            "weights_sha256": tensor_hash(model.state_dict())}


def train(args):
    plan = read(args.plan)
    verify_plan(plan, args.run_name)
    cfg = plan["runs"][args.run_name]
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite model artifacts: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    from run import activate_source, build_loaders
    from augmentation import make_provenance
    from contract import audit_databases
    from run import input_probe
    activate_source(plan["source_root"], "ajai")
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from augmentation_callbacks import AugmentationCallback
    from continue_augmentation import validate_checkpoint
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one visible CUDA GPU")
    environment_check = check_environment()
    provenance = make_provenance(Path(plan["source_root"]), str(Path(plan["source_root"]) / "src/graphnet/__init__.py"), torch, pl)
    if environment_identity(provenance) != plan["parent_environment"]:
        raise ValueError("The working PACE environment changed from the original screen")
    provenance["directional_environment_check"] = environment_check
    provenance["followup_plan_sha256"] = sha256(args.plan)
    write_json(output / "resolved_settings.json", cfg)
    write_json(output / "provenance.json", provenance)
    splits = load_splits(args.selection_root, cfg)
    eligible = {key: splits[key] for key in ("train", "validation")}
    args.source_root, args.variant, args.group_factor = plan["source_root"], "ajai", 16
    args.output_dir = str(output)
    model, dataset_kwargs, initial = make_model(cfg)
    if initial != plan["common_initialization"]:
        raise ValueError("Common initial backbone/vertex state changed")
    write_json(output / "initial_shared_state.json", initial)
    databases = audit_databases(args.data_root, splits, dataset_kwargs["features"])
    if {r["database"]: r["sha256"] for r in databases} != plan["parent_databases"]:
        raise ValueError("Staged databases differ from original screen")
    write_json(output / "database_manifest.json", databases)
    loaders = build_loaders(args, cfg, eligible, dataset_kwargs)
    input_probe(loaders["validation"], output / "input_probe.json")
    if read(output / "input_probe.json") != plan["parent_input_probe"]:
        raise ValueError("Held-out inputs differ from original screen")
    # A separate, disposable smoke fit exercises real CUDA AMP/backward/restore.
    smoke_root = output / "preflight"
    smoke_checkpoint = ModelCheckpoint(dirpath=str(smoke_root), monitor="val_loss", mode="min")
    position_before = tensor_hash(model.position_task.state_dict())
    from directional_run import trainer as screen_trainer
    probe = screen_trainer(cfg, smoke_root, [AugmentationCallback(42, True), smoke_checkpoint], smoke=True)
    probe.fit(model, loaders["train"], loaders["validation"])
    if not model.successful_optimizer_steps or not smoke_checkpoint.best_model_path:
        raise RuntimeError("Preflight did not train and save a checkpoint")
    if cfg["task"] == "direction" and tensor_hash(model.position_task.state_dict()) != position_before:
        raise RuntimeError("Direction-only preflight changed vertex weights")
    state = torch.load(smoke_checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to("cuda").eval()
    with torch.no_grad():
        values = model.components(next(iter(loaders["validation"])).to("cuda"))
        if not all(torch.isfinite(v).all() for v in values if v is not None):
            raise FloatingPointError("Nonfinite restored preflight loss")
    write_json(output / "preflight.json", {"passed": True, "successful_updates": model.successful_optimizer_steps,
               "checkpoint_restore": True, "vertex_supervision": cfg["task"] == "joint"})
    del model, loaders, probe, state, values
    gc.collect()
    torch.cuda.empty_cache()
    model, _, repeated = make_model(cfg)
    if repeated != initial:
        raise ValueError("Smoke fit changed production initialization")
    loaders = build_loaders(args, cfg, eligible, dataset_kwargs)
    parent = plan["parents"][cfg["arm"]] if cfg["starting_epochs"] else None
    if parent:
        parity = parent_parity(model, loaders["validation"], cfg, parent)
        write_json(output / "parent_prediction_parity.json", parity)
        state = torch.load(parent["last_path"], map_location="cpu", weights_only=False)
        validate_checkpoint(state, 10, cfg["precision"])
        del state
        complete = read(Path(parent["folder"]) / "complete.json")
        model.successful_optimizer_steps = complete["successful_optimizer_steps"]
        model.overflow_batches = complete["amp_overflow_batches"]
        write_json(output / "checkpoint_source.json", {"parent": parent["folder"],
                   "last_checkpoint_sha256": parent["last_sha256"], "last_checkpoint": parent["last_path"],
                   "restore": "Lightning weights, optimizer, scheduler, loop and AMP state",
                   "rng_limit": plan["continuation_rng_limit"]})
    callbacks, monitor, snapshots, checkpoints, stopping = make_callbacks(cfg, output, parent)
    fitting = pl.Trainer(accelerator="gpu", devices=1, strategy="auto", max_epochs=MAX_EPOCHS,
                         precision=cfg["precision"], accumulate_grad_batches=1, num_sanity_val_steps=0,
                         use_distributed_sampler=False, callbacks=callbacks,
                         logger=CSVLogger(str(output / "logs"), name="training"))
    fit_started = time.perf_counter()
    fitting.fit(model, loaders["train"], loaders["validation"],
                ckpt_path=parent["last_path"] if parent else None)
    fit_seconds = time.perf_counter()-fit_started
    completed = cfg["starting_epochs"] + monitor.completed_epochs
    if not 10 <= completed <= MAX_EPOCHS or (completed < MAX_EPOCHS and not fitting.should_stop):
        raise RuntimeError("Training neither reached its cap nor stopped under the declared rule")
    last = output / "checkpoints" / "last.ckpt"
    seal_last_checkpoint(fitting, last, completed)
    state = torch.load(last, map_location="cpu", weights_only=False)
    validate_checkpoint(state, completed, cfg["precision"])
    del state
    if not (output / "snapshots" / "epoch10" / "selection.json").is_file():
        if completed != 10:
            raise RuntimeError("The matched ten-epoch snapshot was not retained")
        snapshots.snapshot("epoch10", 10)
    if completed == 30 and not (output / "snapshots" / "epoch30" / "selection.json").is_file():
        snapshots.snapshot("epoch30", 30)
    snapshots.snapshot("final", completed)
    successful, overflows = model.successful_optimizer_steps, model.overflow_batches
    exports = {}
    if parent:
        # These exact weights were already exported; preserve the original
        # export rather than rerunning 51 minutes of identical flow diagnostics.
        for criterion in ("objective", "direction"):
            manifest_path = Path(parent["folder"]) / "validation" / "prediction_manifest.json"
            manifest = read(manifest_path)
            selected = dict(parent["selected"])
            if criterion == "direction":
                selected.update(monitor="val_direction_nll", score=parent["best_direction_score"])
            exports["epoch10/"+criterion] = {"folder": parent["folder"],
                "manifest_sha256": sha256(manifest_path), "prediction_sha256": manifest["sha256"],
                "criterion": criterion, "budget_epochs": 10, "selected_checkpoint": selected,
                "reused_parent_export": True}
    else:
        criterion = "objective" if cfg["task"] == "joint" else "direction"
        exports["epoch10/"+criterion] = export_snapshot(model, loaders["validation"], cfg, output, "epoch10", criterion)
    criteria = ("objective", "direction") if cfg["task"] == "joint" and cfg["arm"] != "king" else (
        "objective" if cfg["task"] == "joint" else "direction",)
    cache = {}
    for criterion in criteria:
        selected = read(output / "snapshots" / "final" / "selection.json")["checkpoints"][criterion]
        # Identical selected epochs have identical weights even if one ckpt
        # also holds optimizer state. Reuse the full export in that case.
        cache_key = selected["epoch"]
        if cache_key in cache:
            state = torch.load(selected["path"], map_location="cpu", weights_only=False)
            if tensor_hash(state["state_dict"]) != cache[cache_key]["weights_sha256"]:
                raise ValueError("Same-epoch checkpoints do not have identical weights")
            del state
            exports["final/"+criterion] = dict(cache[cache_key], criterion=criterion, selected_checkpoint=selected,
                                              reused_same_epoch_export=True)
        else:
            value = export_snapshot(model, loaders["validation"], cfg, output, "final", criterion)
            cache[cache_key] = value
            exports["final/"+criterion] = value
    atomic_json(output / "complete.json", {"protocol": "directional-followup-v1", "run_name": cfg["run_name"],
        "epochs": completed, "starting_epochs": cfg["starting_epochs"], "epoch_cap": MAX_EPOCHS,
        "stopped_early": completed < MAX_EPOCHS, "stop_wait_count": stopping.wait_count,
        "train_rows": COUNTS["train"], "validation_rows": COUNTS["validation"],
        "test_exported": False, "test_rows": 0, "validation_exported": True,
        "successful_optimizer_steps": successful, "amp_overflow_batches": overflows,
        "fit_seconds": fit_seconds, "last_checkpoint": {"path": str(last), "sha256": sha256(last)},
        "exports": exports, "plan_sha256": sha256(args.plan),
        "settings_sha256": sha256(output / "resolved_settings.json"),
        "provenance_sha256": sha256(output / "provenance.json")})
    print(json.dumps({"completed": cfg["run_name"], "epochs": completed, "exports": list(exports)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "train"))
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--screen-root")
    parser.add_argument("--historical-root")
    parser.add_argument("--data-root")
    parser.add_argument("--selection-root")
    parser.add_argument("--plan")
    parser.add_argument("--run-name", choices=TASK_NAMES)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.command in ("prepare", "check"):
        if not all((args.screen_root, args.historical_root, args.data_root, args.selection_root)):
            parser.error("prepare/check require screen-root, historical-root, data-root and selection-root")
        if args.command == "prepare" and not args.plan:
            parser.error("prepare requires --plan")
        if args.command == "check":
            args.plan = None
        prepare(args)
    else:
        if not all((args.plan, args.run_name, args.output_dir, args.data_root, args.selection_root)):
            parser.error("train requires plan, run-name, output-dir, data-root and selection-root")
        train(args)


if __name__ == "__main__":
    main()
