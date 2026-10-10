"""Shared dataset and timing policy, imported from either source checkout."""

import csv
import hashlib
import time
from pathlib import Path

import torch
from pytorch_lightning import Callback
from torch_geometric.data import Batch
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset

from contract import event_seed


def worker_init(_worker_id):
    torch.set_num_threads(1)


def strict_collate(graphs):
    # GraphNeT's default collator silently drops <2-pulse events.
    if any(int(graph.n_pulses) < 2 for graph in graphs):
        raise ValueError(
            "Pilot contains a <2-pulse event; agree on a common selection before training"
        )
    return Batch.from_data_list(graphs)


class PilotDataset(SQLiteDataset):
    """Keep training sampling; freeze held-out CPU RNG by composite event key."""

    def __init__(self, *args, comparison_split, evaluation_seed=42, **kwargs):
        self.comparison_split = comparison_split
        self.evaluation_seed = evaluation_seed
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self.comparison_split == "train":
            return super().__getitem__(index)
        event_id = self._get_event_index(index)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                event_seed(self.path, event_id, self.evaluation_seed)
            )
            graph = super().__getitem__(index)
        # Hash the actual six model inputs, not just the RNG seed. Carlos also
        # carries pmt_area, which its six-feature backbone does not encode.
        inputs = (
            graph.x[:, :6].detach().cpu().contiguous().numpy().astype("<f4")
        )
        graph.input_sha256 = hashlib.sha256(inputs.tobytes()).hexdigest()
        graph.source_database = Path(self.path).name
        return graph


class RunMonitor(Callback):
    def __init__(self, output_dir, expected_train, expected_validation):
        self.output_dir = Path(output_dir)
        self.expected_train = expected_train
        self.expected_validation = expected_validation
        self.completed_epochs = 0

    def on_train_epoch_start(self, trainer, pl_module):
        sampler = trainer.train_dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.train_events = self.val_events = self.steps = 0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite training loss")
        self.train_events += int(batch.num_graphs)
        self.steps += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.train_seconds = time.perf_counter() - self.started

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.val_events += int(batch.num_graphs)

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        if (self.train_events, self.val_events) != (
            self.expected_train,
            self.expected_validation,
        ):
            raise RuntimeError(
                f"Wrong exposure: train={self.train_events}, validation={self.val_events}"
            )
        value = trainer.callback_metrics.get("val_loss")
        if value is None or not torch.isfinite(value).all():
            raise FloatingPointError("Missing/nonfinite validation loss")
        row = {
            "epoch": trainer.current_epoch,
            "train_events": self.train_events,
            "validation_events": self.val_events,
            "optimizer_steps": self.steps,
            "train_seconds": self.train_seconds,
            "epoch_seconds": time.perf_counter() - self.started,
            "train_events_per_second": self.train_events / self.train_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
            "val_loss": float(value),
        }
        path = self.output_dir / "timing.csv"
        exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self.completed_epochs += 1
