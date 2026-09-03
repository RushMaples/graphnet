"""Create reproducible train/validation selections for track scoring.

This script deliberately handles *physics selection* separately from model
training. Its CSV output is the contract that says exactly which simulated
events were used in each split.

The defaults follow the TANGO technical report: use nu_e and nu_mu events,
label nu_mu charged-current events as tracks, draw 10% of each class/energy
stratum, and divide the result 9:1 into training and validation.
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


# These are the pulse columns expected by FEATURES.ICECUBE86. The split script
# only reads the truth table, but checking the pulse table here catches a
# mismatched database before a GPU job is submitted.
EXPECTED_PULSE_FEATURES = [
    "dom_x",
    "dom_y",
    "dom_z",
    "dom_time",
    "charge",
    "rde",
    "pmt_area",
]

# The report covers approximately 1 GeV to 10 TeV. Sampling inside these bins
# keeps a random 10% sample from accidentally distorting the energy spectrum.
DEFAULT_ENERGY_BIN_EDGES = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]


def _validate_identifier(name: str) -> str:
    """Reject unsafe SQL table/column names supplied on the command line."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise ValueError(f"Invalid SQLite identifier: {name!r}")
    return name


def resolve_databases(
    input_dir: Optional[Path],
    explicit_databases: Optional[List[Path]],
    max_files: Optional[int],
) -> List[Path]:
    """Resolve either an explicit file list or every database in a directory.

    Explicit paths are safest on a shared filesystem such as PACE: the parent
    directory can contain databases for other reconstruction tasks that should
    not silently enter this training sample.
    """
    if explicit_databases is not None:
        databases = explicit_databases
    else:
        assert input_dir is not None  # Enforced by argparse's required group.
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory does not exist: {input_dir}"
            )
        databases = sorted(input_dir.glob("*.db"))

    if max_files is not None:
        databases = databases[:max_files]
    if not databases:
        raise FileNotFoundError("No SQLite databases were selected")

    missing = [database for database in databases if not database.is_file()]
    if missing:
        rendered = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Database path(s) do not exist:\n{rendered}")

    non_sqlite = [database for database in databases if database.suffix != ".db"]
    if non_sqlite:
        rendered = "\n".join(f"  {path}" for path in non_sqlite)
        raise ValueError(f"Expected .db files, received:\n{rendered}")

    # Resolve symlinks and relative paths before they are written to the
    # manifest. Compute nodes must be able to reopen these exact paths later.
    databases = [database.resolve() for database in databases]
    if len(databases) != len(set(databases)):
        raise ValueError("The database list contains the same file more than once")
    return databases


def inspect_database(
    database: Path,
    truth_table: str,
    pulsemap: str,
    event_column: str,
    pid_column: str,
    interaction_column: str,
    energy_column: str,
) -> None:
    """Check that one database contains the tables and columns we will use."""
    required_truth = {
        event_column,
        pid_column,
        interaction_column,
        energy_column,
    }
    required_pulses = {event_column, *EXPECTED_PULSE_FEATURES}

    with sqlite3.connect(str(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = {truth_table, pulsemap} - tables
        if missing_tables:
            raise ValueError(
                f"{database} is missing tables {sorted(missing_tables)}; "
                f"available tables are {sorted(tables)}"
            )

        truth_columns = {
            row[1]
            for row in connection.execute(f'PRAGMA table_info("{truth_table}")')
        }
        pulse_columns = {
            row[1]
            for row in connection.execute(f'PRAGMA table_info("{pulsemap}")')
        }

    missing_truth = required_truth - truth_columns
    missing_pulses = required_pulses - pulse_columns
    if missing_truth:
        raise ValueError(
            f"{database}:{truth_table} is missing {sorted(missing_truth)}; "
            f"available columns are {sorted(truth_columns)}"
        )
    if missing_pulses:
        raise ValueError(
            f"{database}:{pulsemap} is missing {sorted(missing_pulses)}; "
            f"available columns are {sorted(pulse_columns)}"
        )


def read_candidate_events(
    database: Path,
    truth_table: str,
    event_column: str,
    pid_column: str,
    interaction_column: str,
    energy_column: str,
) -> pd.DataFrame:
    """Read only nu_e and nu_mu candidates and standardize column names."""
    # Aliases make the manifest independent of the original database's column
    # names. Adjust command-line arguments rather than editing later stages.
    query = f"""
        SELECT
            "{event_column}" AS event_no,
            "{pid_column}" AS pid,
            "{interaction_column}" AS interaction_type,
            "{energy_column}" AS energy
        FROM "{truth_table}"
        WHERE ABS("{pid_column}") IN (12, 14)
    """
    with sqlite3.connect(str(database)) as connection:
        events = pd.read_sql_query(query, connection)

    events["source_database"] = str(database.resolve())
    return events


def add_track_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Implement the report's binary truth definition."""
    result = events.copy()
    is_numu = result["pid"].abs() == 14
    is_charged_current = result["interaction_type"] == 1
    result["track"] = (is_numu & is_charged_current).astype(int)
    return result


def add_energy_bins(events: pd.DataFrame, edges: List[float]) -> pd.DataFrame:
    """Attach an energy-bin label used only to stratify sampling/splitting."""
    if len(edges) < 2 or edges != sorted(edges):
        raise ValueError("Energy-bin edges must be an increasing list")

    result = events.copy()
    result["energy_bin"] = pd.cut(
        result["energy"],
        bins=edges,
        include_lowest=True,
    ).astype(str)

    outside = result["energy_bin"] == "nan"
    if outside.any():
        count = int(outside.sum())
        print(
            f"WARNING: {count} events fall outside the requested energy bins; "
            "they will share an 'nan' stratum. Adjust --energy-bin-edges if "
            "this is unexpected."
        )
    return result


def sample_strata(
    events: pd.DataFrame, fraction: float, seed: int
) -> pd.DataFrame:
    """Sample a fraction independently inside every class/energy group."""
    if not 0 < fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")

    sampled_groups = []
    for _, group in events.groupby(["track", "energy_bin"], observed=True):
        # Retain at least two members when possible so a stratum can appear in
        # both train and validation. No event is sampled more than once.
        minimum = 2 if len(group) >= 2 else 1
        count = min(len(group), max(minimum, round(len(group) * fraction)))
        sampled_groups.append(group.sample(n=count, random_state=seed))

    sampled = pd.concat(sampled_groups, ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def choose_stratification(events: pd.DataFrame) -> pd.Series:
    """Prefer class+energy stratification, falling back to class if needed."""
    detailed = events["track"].astype(str) + ":" + events["energy_bin"]
    if detailed.value_counts().min() >= 2:
        return detailed

    print(
        "WARNING: at least one class/energy stratum has only one sampled "
        "event; splitting will be stratified by track label only."
    )
    return events["track"]


def assign_splits(
    events: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign disjoint train, validation, and optional test labels."""
    if validation_fraction <= 0 or test_fraction < 0:
        raise ValueError("Validation must be positive and test cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")

    remaining = events
    test = None
    if test_fraction > 0:
        strata = choose_stratification(events)
        remaining, test = train_test_split(
            events,
            test_size=test_fraction,
            random_state=seed,
            stratify=strata,
        )

    # validation_fraction is expressed relative to the original sample. After
    # removing test events, convert it to the corresponding remaining fraction.
    adjusted_validation = validation_fraction / (1 - test_fraction)
    strata = choose_stratification(remaining)
    train, validation = train_test_split(
        remaining,
        test_size=adjusted_validation,
        random_state=seed,
        stratify=strata,
    )

    train = train.assign(split="train")
    validation = validation.assign(split="validation")
    parts = [train, validation]
    if test is not None:
        parts.append(test.assign(split="test"))

    return pd.concat(parts, ignore_index=True).sort_values(
        ["split", "source_database", "event_no"]
    )


def print_summary(manifest: pd.DataFrame) -> None:
    """Print the counts that should always be inspected before training."""
    summary = (
        manifest.groupby(["split", "track"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "cascade", 1: "track"})
    )
    summary["total"] = summary.sum(axis=1)
    print("\nManifest summary:")
    print(summary.to_string())
    print(f"\nDatabases: {manifest['source_database'].nunique()}")
    print(f"Events: {len(manifest)}")


def print_candidate_summary(candidates: pd.DataFrame) -> None:
    """Show which flavors/interactions came from each selected database."""
    summary = (
        candidates.assign(
            database=candidates["source_database"].map(
                lambda path: Path(path).name
            )
        )
        .groupby(["database", "pid", "interaction_type", "track"])
        .size()
        .rename("events")
        .reset_index()
    )
    print("\nCandidate truth summary (before 10% sampling):")
    print(summary.to_string(index=False))


def main(args: argparse.Namespace) -> None:
    """Inspect databases, select events, split them, and write the manifest."""
    identifier_arguments = [
        args.truth_table,
        args.pulsemap,
        args.event_column,
        args.pid_column,
        args.interaction_column,
        args.energy_column,
    ]
    for identifier in identifier_arguments:
        _validate_identifier(identifier)

    databases = resolve_databases(
        args.input_dir,
        args.databases,
        args.max_files,
    )
    print(f"Selected {len(databases)} SQLite database(s).")

    frames = []
    for index, database in enumerate(databases, start=1):
        print(f"[{index}/{len(databases)}] Reading {database}")
        inspect_database(
            database,
            args.truth_table,
            args.pulsemap,
            args.event_column,
            args.pid_column,
            args.interaction_column,
            args.energy_column,
        )
        frames.append(
            read_candidate_events(
                database,
                args.truth_table,
                args.event_column,
                args.pid_column,
                args.interaction_column,
                args.energy_column,
            )
        )

    candidates = pd.concat(frames, ignore_index=True)
    candidates = add_track_labels(candidates)
    candidates = add_energy_bins(candidates, args.energy_bin_edges)
    print_candidate_summary(candidates)

    if candidates["track"].nunique() != 2:
        raise ValueError(
            "The selected databases did not produce both track and cascade "
            "labels. Verify PID and interaction_type conventions before training."
        )

    duplicate_key = ["source_database", "event_no"]
    if candidates.duplicated(duplicate_key).any():
        raise ValueError(
            "Duplicate (source_database, event_no) keys found in truth data"
        )

    print(f"Candidate nu_e/nu_mu events: {len(candidates)}")
    sampled = sample_strata(candidates, args.sample_fraction, args.seed)
    manifest = assign_splits(
        sampled,
        args.validation_fraction,
        args.test_fraction,
        args.seed,
    )

    columns = [
        "source_database",
        "event_no",
        "split",
        "track",
        "energy",
        "energy_bin",
        "pid",
        "interaction_type",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, columns=columns, index=False)
    print_summary(manifest)
    print(f"\nWrote manifest to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface and its dataset-specific knobs."""
    parser = argparse.ArgumentParser(
        description="Create a reproducible track-scoring event manifest."
    )
    database_source = parser.add_mutually_exclusive_group(required=True)
    database_source.add_argument(
        "--input-dir",
        type=Path,
        help="Directory whose direct children are GraphNeT SQLite .db files.",
    )
    database_source.add_argument(
        "--databases",
        type=Path,
        nargs="+",
        help=(
            "Exact SQLite files to use. Prefer this on PACE so unrelated files "
            "in the shared parent directory cannot enter the sample."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV manifest to create.",
    )
    parser.add_argument(
        "--truth-table",
        default="truth",
        help="Event-level truth table (change if PACE data uses another name).",
    )
    parser.add_argument(
        "--pulsemap",
        default="SplitRTCleanedInIcePulses",
        help="Pulse table that the later training job will read.",
    )
    parser.add_argument("--event-column", default="event_no")
    parser.add_argument("--pid-column", default="pid")
    parser.add_argument("--interaction-column", default="interaction_type")
    parser.add_argument("--energy-column", default="energy")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.10,
        help="Fraction sampled within each class/energy stratum (default: 0.10).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
        help="Fraction of sampled events assigned to validation (default: 0.10).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Optional fraction of sampled events reserved for testing.",
    )
    parser.add_argument(
        "--energy-bin-edges",
        type=float,
        nargs="+",
        default=DEFAULT_ENERGY_BIN_EDGES,
        help="Energy-bin edges in GeV used for stratification.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit it for the real manifest.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
