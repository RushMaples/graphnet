"""Frozen P14 pilot inputs; deliberately usable without GraphNeT or a GPU."""

import csv
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATABASES = [
    f"P14_{flavour}_database_part_1.db" for flavour in ("numu", "nue", "nutau")
]
SPLITS = ("train", "validation", "test")
FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde"]
TRUTH = [
    "position_x",
    "position_y",
    "position_z",
    "azimuth",
    "zenith",
    "energy",
    "pid",
    "interaction_type",
    "oneweight",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_contract():
    return json.loads(
        (HERE / "pilot_contract.json").read_text(encoding="utf-8-sig")
    )


def read_ids(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["event_no"]:
            raise ValueError(f"Expected one event_no column: {path}")
        values = [int(row["event_no"]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate event IDs: {path}")
    return values


def load_splits(selection_root, contract=None):
    """Require exact archived CSV bytes, counts, and disjoint per-DB lists."""
    contract = contract or load_contract()
    result = {split: [] for split in SPLITS}
    for database in DATABASES:
        seen = set()
        for split in SPLITS:
            name = f"{Path(database).stem}_{split}.csv"
            path = Path(selection_root) / name
            expected = contract["selections"][name]
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"Frozen pilot CSV checksum mismatch: {path}")
            ids = read_ids(path)
            if len(ids) != expected["count"] or seen.intersection(ids):
                raise ValueError(f"Wrong split count or leakage: {path}")
            seen.update(ids)
            result[split].append(ids)
    return result


def audit_databases(data_root, splits, required_features=FEATURES):
    """Read-only schema/membership audit and full DB hashes (startup cost)."""
    records = []
    for index, name in enumerate(DATABASES):
        path = (Path(data_root) / name).resolve()
        requested = set().union(*(set(splits[s][index]) for s in SPLITS))
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        ) as connection:
            for table, required in (
                ("truth", ["event_no", *TRUTH]),
                ("SRTInIcePulses", ["event_no", *required_features]),
            ):
                columns = {
                    row[1]
                    for row in connection.execute(
                        f'PRAGMA table_info("{table}")'
                    )
                }
                if not set(required) <= columns:
                    raise ValueError(
                        f"{name}/{table} missing {set(required) - columns}"
                    )
            found = set()
            total = 0
            for (event_id,) in connection.execute(
                "SELECT event_no FROM truth"
            ):
                total += 1
                if event_id in requested:
                    if event_id in found:
                        raise ValueError(
                            f"Duplicate selected truth ID in {name}: {event_id}"
                        )
                    found.add(event_id)
            if found != requested:
                raise ValueError(
                    f"{name}: {len(requested - found)} selected events missing"
                )
        records.append(
            {
                "database": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "truth_rows": total,
                "selected": len(requested),
                "sha256": sha256(path),
            }
        )
    return records


def event_seed(database, event_no, seed=42):
    """Portable seed, independent of worker/batch order and Python hash salt."""
    key = f"p14-eval-v1:{seed}:{Path(database).name}:{int(event_no)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (
        2**63 - 1
    )


def source_manifest(root):
    """Hash executed source/config, including untracked pilot modifications."""
    root = Path(root).resolve()
    files = []
    for folder in ("src/graphnet", "ice_mix", "comparison"):
        for path in sorted((root / folder).rglob("*")):
            relative = path.relative_to(root)
            if any(
                part in {"outputs", "checkpoints", "__pycache__", ".git"}
                for part in relative.parts
            ):
                continue
            if path.is_file() and path.suffix in {
                ".py",
                ".yaml",
                ".yml",
                ".json",
                ".sh",
                ".sbatch",
            }:
                files.append(
                    {"path": relative.as_posix(), "sha256": sha256(path)}
                )
    return files


def environment_identity(provenance):
    """Comparable runtime properties, excluding per-job GPU UUIDs/paths."""
    return {
        key: provenance[key]
        for key in (
            "python",
            "torch",
            "cuda_runtime",
            "lightning",
            "gpu",
            "pip_freeze",
        )
    }


def run_identity(root):
    """Bind a passed preflight to sources, settings, runtime and actual data."""
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    provenance = read("provenance.json")
    return {
        "settings": read("resolved_settings.json"),
        "source_files": provenance["source_files"],
        "harness_files": provenance["harness_files"],
        "environment": environment_identity(provenance),
        "databases": {
            row["database"]: row["sha256"]
            for row in read("database_manifest.json")
        },
        "input_probe": read("input_probe.json"),
    }
