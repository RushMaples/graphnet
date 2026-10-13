"""Check comparison imports in isolated CPU processes before submitting jobs.

Run with the same pilot Python used by pace_compare.sbatch. This only imports
the selected source code; it does not read databases or construct models.
"""

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys


CARLOS_ROOT = (
    "/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet"
)
HERE = Path(__file__).resolve().parent

COMMON_MODEL = (
    ("torch", ()),
    ("graphnet.models", ("StandardModel",)),
    ("graphnet.models.detector.icecube", ("IceCube86",)),
    (
        "graphnet.training.loss_functions",
        ("EuclideanDistanceLoss", "VonMisesFisher3DLoss"),
    ),
)
VARIANT_MODEL = {
    "carlos": (
        ("graphnet.models.graphs", ("GraphDefinition",)),
        ("graphnet.models.graphs.nodes", ("IceMixNodes",)),
        (
            "graphnet.models.task.reconstruction",
            ("JointPositionandDirectionReco",),
        ),
        ("graphnet.training.labels", ("JointLabel",)),
        ("src.models.transformer", ("IceMix",)),
        (
            "src.metrics_logging",
            ("JointLossWithMetrics", "PhysicsMetricsCallback"),
        ),
    ),
    "ajai": (
        (
            "graphnet.models.transformer.icemix",
            ("IceMixReconstruction", "build_data_representation"),
        ),
        ("graphnet.training.labels", ("Direction",)),
        ("graphnet.training.callbacks", ("PhysicsMetricsCallback",)),
    ),
}
LOADERS = (
    ("torch.utils.data", ("ConcatDataset", "Subset")),
    ("graphnet.data.dataloader", ("DataLoader",)),
    (
        "runtime",
        ("PilotDataset", "strict_collate", "worker_init", "RunMonitor"),
    ),
)
AJAI_LENGTHS = (
    ("graphnet.data.dataset.samplers", ("LengthGroupedSampler",)),
    ("graphnet.data.utilities.lengths", ("event_lengths",)),
)
TRAINING = (
    ("pytorch_lightning", ()),
    ("pytorch_lightning.callbacks", ("ModelCheckpoint",)),
    ("pytorch_lightning.loggers", ("CSVLogger",)),
    ("torch_geometric", ()),
    ("torch_scatter", ()),
    ("torch_sparse", ()),
    ("torch_cluster", ()),
)


def check_worker(variant, source_root):
    """Use the harness source activation and resolve its imported symbols."""
    from run import activate_source

    root, actual = activate_source(source_root, variant)
    print(f"[{variant}] Python: {sys.executable}", flush=True)
    print(f"[{variant}] Source: {root}; graphnet: {actual}", flush=True)
    imports = COMMON_MODEL + VARIANT_MODEL[variant] + LOADERS
    if variant == "ajai":
        imports += AJAI_LENGTHS
    imports += TRAINING
    for module_name, attributes in imports:
        print(f"[{variant}] Importing {module_name}", flush=True)
        module = importlib.import_module(module_name)
        for attribute in attributes:
            getattr(module, attribute)
    print(f"[{variant}] Import check passed", flush=True)
    return 0


def run_checks(variants, carlos_root, ajai_root):
    """Check every requested variant in a fresh process, reporting failures."""
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    roots = {"carlos": carlos_root, "ajai": ajai_root}
    failed = []
    for variant in variants:
        source_root = Path(roots[variant]).resolve()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(source_root / "src"), str(HERE))
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            variant,
            "--source-root",
            str(source_root),
        ]
        result = subprocess.run(command, env=environment)
        if result.returncode:
            failed.append(variant)
            print(
                f"[{variant}] Import check failed (exit {result.returncode})",
                file=sys.stderr,
                flush=True,
            )
    if failed:
        print(f"Failed variants: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(
        "All requested import checks passed. GPU preflight is still required."
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("carlos", "ajai"),
        default=["carlos", "ajai"],
    )
    parser.add_argument(
        "--carlos-root", default=os.environ.get("CARLOS_ROOT", CARLOS_ROOT)
    )
    parser.add_argument(
        "--ajai-root",
        default=os.environ.get("ICEMIX_CHECKOUT", str(HERE.parent)),
    )
    parser.add_argument(
        "--worker", choices=("carlos", "ajai"), help=argparse.SUPPRESS
    )
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.source_root:
            parser.error("--source-root is required with --worker")
        return check_worker(args.worker, args.source_root)
    return run_checks(args.variants, args.carlos_root, args.ajai_root)


if __name__ == "__main__":
    raise SystemExit(main())
