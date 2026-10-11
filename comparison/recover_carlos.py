"""Restore pinned Carlos source and archived pilot selections; no GPU needed."""

import argparse
import hashlib
import io
import csv
import subprocess
import tarfile
from pathlib import Path

from contract import DATABASES, SPLITS, load_contract, load_splits, sha256

CARLOS_REVISION = "1d83975e21b5b1c7f02efb797d5fc39f5f5735c6"
CARLOS_URL = "https://github.com/carlosm-silva/graphnet.git"
DEFAULT_ROOT = (
    "/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet"
)


def read_archive(path):
    """Validate bytes, identities, and counts before writing any files."""
    contract = load_contract()
    expected = contract["selections"]
    contents = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name == "manifest.json" and member.isfile():
                continue
            if member.name not in expected or not member.isfile():
                raise ValueError(f"Unexpected archive entry: {member.name}")
            if member.name in contents or member.size > 10_000_000:
                raise ValueError(
                    f"Duplicate or oversized entry: {member.name}"
                )
            with archive.extractfile(member) as handle:
                data = handle.read()
            if (
                hashlib.sha256(data).hexdigest()
                != expected[member.name]["sha256"]
            ):
                raise ValueError(
                    f"Archived split checksum mismatch: {member.name}"
                )
            contents[member.name] = data
    if set(contents) != set(expected):
        raise ValueError("Archive must contain all nine original pilot CSVs")
    for database in DATABASES:
        seen = set()
        for split in SPLITS:
            name = f"{Path(database).stem}_{split}.csv"
            rows = csv.DictReader(
                io.StringIO(contents[name].decode("utf-8-sig"))
            )
            if rows.fieldnames != ["event_no"]:
                raise ValueError(f"Invalid columns: {name}")
            ids = [int(row["event_no"]) for row in rows]
            if (
                len(ids) != expected[name]["count"]
                or len(ids) != len(set(ids))
                or seen.intersection(ids)
            ):
                raise ValueError(
                    f"Invalid membership/count or split leakage: {name}"
                )
            seen.update(ids)
    return contents


def git(*arguments):
    return subprocess.check_output(
        ["git", *map(str, arguments)], text=True
    ).strip()


def restore_source(root):
    if root.exists():
        if not (root / ".git").exists():
            raise FileExistsError(
                f"Refusing to overwrite existing directory: {root}"
            )
        if git("-C", root, "rev-parse", "HEAD") != CARLOS_REVISION:
            raise ValueError(
                f"Existing checkout is not the pinned Carlos revision: {root}"
            )
        if git("-C", root, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(
                f"Existing checkout has source edits; preserve them and choose another --carlos-root: {root}"
            )
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "cursor",
                "--single-branch",
                CARLOS_URL,
                str(root),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--detach", CARLOS_REVISION],
            check=True,
        )
    for relative in (
        "src/graphnet/__init__.py",
        "ice_mix/src/models/transformer.py",
        "ice_mix/src/metrics_logging.py",
    ):
        if not (root / relative).is_file():
            raise FileNotFoundError(
                f"Missing baseline source: {root / relative}"
            )


def restore_selections(contents, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # Check every existing file before adding any missing file.
    for name, data in contents.items():
        path = destination / name
        if path.exists() and sha256(path) != hashlib.sha256(data).hexdigest():
            raise FileExistsError(
                f"Refusing to overwrite a different selection: {path}"
            )
    for name, data in contents.items():
        path = destination / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)
    return load_splits(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help="Uploaded p14_pilot_frozen_selections.tar.gz",
    )
    parser.add_argument("--carlos-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--selection-root", type=Path)
    args = parser.parse_args()
    contents = read_archive(args.archive)
    root = args.carlos_root.resolve()
    destination = (
        args.selection_root
        or root / "ice_mix/outputs/pilot_10pct_p14_three_seed42/selections"
    )
    restore_source(root)
    splits = restore_selections(contents, destination)
    print(f"Carlos source: {root} at {CARLOS_REVISION}")
    print(f"Frozen selections: {destination}")
    print({name: sum(map(len, values)) for name, values in splits.items()})
    print(
        "Source and selections restored. Environment, databases and old checkpoints were not recreated."
    )
    print(
        "If custom paths were used, export CARLOS_ROOT and PILOT_SELECTION_ROOT before sbatch."
    )


if __name__ == "__main__":
    main()
