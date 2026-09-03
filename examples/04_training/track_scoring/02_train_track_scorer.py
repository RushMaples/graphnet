"""Train a DynEdgeTITO binary track-scoring model.

This is an annotated adaptation of ``02_train_tito_model.py``. The overall
GraphNeT workflow is intentionally kept recognizable; the reconstruction task
has been replaced by binary classification.
"""

import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.data.constants import FEATURES
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.callbacks import ProgressBar
from graphnet.training.labels import Track
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger


# Node features are the measurements available to the learner. Copy the list
# so this script never mutates GraphNeT's global constants.
FEATURES_USED = list(FEATURES.ICECUBE86)

# These are event-level simulation values, not model inputs. PID and
# interaction_type define the label; energy is retained for diagnostic plots.
TRUTH_USED = ["energy", "pid", "interaction_type"]


def read_manifest(
    path: Path,
    limit_train: Optional[int],
    limit_validation: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Load and validate the event selections made by 01_make_splits.py."""
    manifest = pd.read_csv(path)
    required = {"source_database", "event_no", "split", "track"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    unexpected = set(manifest["split"].unique()) - {
        "train",
        "validation",
        "test",
    }
    if unexpected:
        raise ValueError(f"Unexpected split labels: {sorted(unexpected)}")

    duplicate_key = ["source_database", "event_no"]
    if manifest.duplicated(duplicate_key).any():
        raise ValueError("Manifest contains duplicate database/event keys")

    # Limits create a fast smoke test while preserving the complete manifest.
    # Sampling, rather than taking the first rows, avoids ordering bias.
    pieces = []
    for split, limit in [
        ("train", limit_train),
        ("validation", limit_validation),
    ]:
        rows = manifest[manifest["split"] == split]
        if limit is not None and len(rows) > limit:
            if limit < 2:
                raise ValueError(
                    f"--limit-{split} must be at least 2 to retain both classes"
                )
            rows, _ = train_test_split(
                rows,
                train_size=limit,
                random_state=seed,
                stratify=rows["track"],
            )
        pieces.append(rows)

    selected = pd.concat(pieces, ignore_index=True)
    for split in ["train", "validation"]:
        split_rows = selected[selected["split"] == split]
        if split_rows.empty:
            raise ValueError(f"Manifest has no selected {split} events")
        if split_rows["track"].nunique() != 2:
            raise ValueError(f"{split} split does not contain both classes")
    return selected


def selections_by_database(
    manifest: pd.DataFrame,
) -> Tuple[List[str], List[List[int]], List[List[int]]]:
    """Convert the manifest into the lists expected by GraphNeTDataModule."""
    database_paths = manifest["source_database"].drop_duplicates().tolist()
    train_selections = []
    validation_selections = []

    for database in database_paths:
        database_rows = manifest[manifest["source_database"] == database]
        train_selections.append(
            database_rows.loc[
                database_rows["split"] == "train", "event_no"
            ]
            .astype(int)
            .tolist()
        )
        validation_selections.append(
            database_rows.loc[
                database_rows["split"] == "validation", "event_no"
            ]
            .astype(int)
            .tolist()
        )

    missing_databases = [path for path in database_paths if not Path(path).is_file()]
    if missing_databases:
        preview = "\n".join(missing_databases[:5])
        raise FileNotFoundError(
            "Manifest database paths are not visible in this environment:\n"
            f"{preview}"
        )
    return database_paths, train_selections, validation_selections


def build_graph_definition() -> KNNGraph:
    """Define how one event's pulses become a graph.

    ``IceCube86`` applies the same detector-specific feature scaling used by
    the existing track/cascade inference workflow. The first three feature
    columns are DOM x/y/z, so the initial KNN edges are spatial.
    """
    return KNNGraph(
        detector=IceCube86(),
        node_definition=NodesAsPulses(),
        input_feature_names=FEATURES_USED,
        nb_nearest_neighbours=8,
        columns=[0, 1, 2],
    )


def build_dataloaders(
    manifest: pd.DataFrame,
    graph_definition: KNNGraph,
    pulsemap: str,
    truth_table: str,
    batch_size: int,
    num_workers: int,
):
    """Create shuffled training and deterministic validation data loaders."""
    paths, train_selections, validation_selections = selections_by_database(
        manifest
    )

    # Track computes the binary label from the raw truth attached to each
    # graph. Keeping this explicit makes the physics definition visible here.
    labels = {
        "track": Track(
            key="track",
            pid_key="pid",
            interaction_key="interaction_type",
        )
    }

    # Each SQLiteDataset owns one database connection and one event selection.
    # EnsembleDataset presents all of those per-file datasets as one dataset.
    # Skip an empty selection: a smoke-test subset may not use every database
    # in both splits.
    def make_ensemble(selections: List[List[int]]) -> EnsembleDataset:
        datasets = []
        for path, selection in zip(paths, selections):
            if not selection:
                continue
            datasets.append(
                SQLiteDataset(
                    path=path,
                    selection=selection,
                    pulsemaps=pulsemap,
                    features=FEATURES_USED,
                    truth=TRUTH_USED,
                    truth_table=truth_table,
                    index_column="event_no",
                    graph_definition=graph_definition,
                    labels=labels,
                )
            )
        if not datasets:
            raise ValueError("No non-empty SQLite selections were constructed")
        return EnsembleDataset(datasets)

    train_dataset = make_ensemble(train_selections)
    validation_dataset = make_ensemble(validation_selections)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_loader, validation_loader


def build_model(
    graph_definition: KNNGraph,
    learning_rate: float,
    scheduler_patience: int,
) -> StandardModel:
    """Build the TITO backbone, binary head, loss, and optimizer.

    The backbone matches the dimensions in the existing TITO training example.
    Keeping it fixed initially makes the 10% data study easier to interpret.
    """
    backbone = DynEdgeTITO(
        nb_inputs=graph_definition.nb_outputs,
        features_subset=[0, 1, 2, 3],
        dyntrans_layer_sizes=[
            (256, 256),
            (256, 256),
            (256, 256),
            (256, 256),
        ],
        global_pooling_schemes=["max"],
        use_global_features=True,
        use_post_processing_layers=True,
    )

    # BinaryClassificationTask learns one output and applies sigmoid, so the
    # prediction is a probability-like track score in [0, 1].
    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels="track",
        prediction_labels="track_score",
        loss_function=BinaryCrossEntropyLoss(),
    )

    return StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": learning_rate, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience": scheduler_patience,
            "factor": 0.5,
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    )


def check_one_batch(model: StandardModel, train_loader) -> None:
    """Run a cheap forward/loss check before starting a long training job."""
    batch = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        loss = model.compute_loss(predictions, [batch])

    scores = predictions[0]
    labels = batch.track
    print("\nSingle-batch preflight")
    print(f"  node feature shape: {tuple(batch.x.shape)}")
    print(f"  number of events:   {batch.num_graphs}")
    print(f"  label shape:        {tuple(labels.shape)}")
    print(f"  prediction shape:   {tuple(scores.shape)}")
    print(f"  labels present:     {torch.unique(labels).tolist()}")
    print(f"  finite loss:        {loss.item():.6f}")

    if scores.shape[-1] != 1:
        raise ValueError("Binary task should produce exactly one score per event")
    if not torch.isfinite(loss):
        raise ValueError("Non-finite loss in the first batch")
    if torch.any(scores < 0) or torch.any(scores > 1):
        raise ValueError("Track scores are outside [0, 1]")
    model.train()


def save_run_arguments(args: Namespace, output_dir: Path) -> None:
    """Record the settings needed to understand and reproduce this run."""
    serializable: Dict[str, object] = {}
    for key, value in vars(args).items():
        serializable[key] = str(value) if isinstance(value, Path) else value
    with (output_dir / "run_arguments.json").open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, sort_keys=True)


def main(args: Namespace) -> None:
    """Load selections, build the model, train it, predict, and save outputs."""
    # This GraphNeT version always configures persistent workers and a prefetch
    # factor, both of which require at least one DataLoader worker.
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1 in this repository")

    logger = Logger()
    seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_arguments(args, args.output_dir)

    logger.info(f"Node features: {FEATURES_USED}")
    logger.info(f"Truth fields: {TRUTH_USED}")
    manifest = read_manifest(
        args.manifest,
        args.limit_train,
        args.limit_validation,
        args.seed,
    )
    # Save the resolved selection too. This protects reproducibility if the
    # original manifest is later moved or a smoke-test limit was applied.
    manifest.to_csv(args.output_dir / "selected_manifest.csv", index=False)
    logger.info(
        "Selected events: "
        f"train={(manifest['split'] == 'train').sum()}, "
        f"validation={(manifest['split'] == 'validation').sum()}"
    )

    graph_definition = build_graph_definition()
    train_loader, validation_loader = build_dataloaders(
        manifest,
        graph_definition,
        args.pulsemap,
        args.truth_table,
        args.batch_size,
        args.num_workers,
    )
    model = build_model(
        graph_definition,
        args.learning_rate,
        args.scheduler_patience,
    )
    check_one_batch(model, train_loader)

    if args.dry_run:
        logger.info("Dry run complete; training was intentionally skipped.")
        return

    # Because callbacks are supplied explicitly, include every behavior we
    # want: progress display, early stopping, and durable best/last checkpoints.
    callbacks = [
        ProgressBar(),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.early_stopping_patience,
        ),
        ModelCheckpoint(
            dirpath=str(args.output_dir / "checkpoints"),
            filename="tito-track-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
    ]
    csv_logger = CSVLogger(
        save_dir=str(args.output_dir),
        name="lightning",
    )

    model.fit(
        train_loader,
        validation_loader,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        callbacks=callbacks,
        logger=csv_logger,
        gradient_clip_val=args.gradient_clip_val,
        distribution_strategy="auto",
        ckpt_path=str(args.resume_from) if args.resume_from else None,
    )

    # StandardModel.fit reloads the best checkpoint when both EarlyStopping
    # and ModelCheckpoint are present, so these predictions use the best epoch.
    predictions = model.predict_as_dataframe(
        validation_loader,
        prediction_columns=["track_score"],
        additional_attributes=[
            "track",
            "energy",
            "pid",
            "interaction_type",
            "event_no",
            "dataset_path",
        ],
        gpus=args.gpus,
        distribution_strategy="auto",
    )
    predictions = predictions.rename(columns={"dataset_path": "source_database"})
    predictions.to_csv(
        args.output_dir / "validation_predictions.csv",
        index=False,
    )

    # The config + state dict is the safer long-term format. model.pth is also
    # saved because existing GraphNeT inference workflows load complete models.
    model.save_state_dict(str(args.output_dir / "state_dict.pth"))
    model.save_config(str(args.output_dir / "model_config.yml"))
    model.save(str(args.output_dir / "model.pth"))
    logger.info(f"Training outputs written to {args.output_dir}")


def build_parser() -> ArgumentParser:
    """Define user-facing settings, with conservative first-run defaults."""
    parser = ArgumentParser(
        description="""
Train a DynEdgeTITO binary track/cascade classifier from a split manifest.

Run 01_make_splits.py first. Use --dry-run or small --limit-* values before
submitting the full reduced-data job.
"""
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pulsemap",
        default="SplitRTCleanedInIcePulses",
        help="SQLite pulse table; must match the manifest/database schema.",
    )
    parser.add_argument("--truth-table", default="truth")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
        help="Epochs before reducing LR; keep below early-stop patience.",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional Lightning last.ckpt path for an interrupted run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check one batch and model forward pass, then stop.",
    )
    parser.with_standard_arguments(
        ("gpus", [0]),
        ("max-epochs", 40),
        ("early-stopping-patience", 6),
        ("batch-size", 32),
        ("num-workers", 4),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
