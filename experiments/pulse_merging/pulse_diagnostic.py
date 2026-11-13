"""Measure same-sensor pulse timing and simulate anchored merging on frozen P14 training IDs.

Input SQLite databases are opened read-only. The sole output is a new ZIP.
Times are in ns and charge in photoelectrons (PE), as exported by GraphNeT.
"""

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import zipfile
from collections import Counter
from contextlib import closing

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

CORE = Path(__file__).resolve().parents[2] / "comparison"
sys.path.insert(0, str(CORE))
from contract import DATABASES, read_ids, sha256

TABLE = "SRTInIcePulses"
GAPS = [0, 1, 2, 5, 10, 25, 50, 100, 300, 1000, 3000, 11000, float("inf")]
CHARGES = [0, .1, .25, .5, 1, 2, 5, float("inf")]


def selected_training_events(root, selections, database, count, seed):
    name = f"{Path(database).stem}_train.csv"
    path = root / name
    expected = selections[name]
    if sha256(path) != expected["sha256"]:
        raise ValueError(f"Training selection checksum mismatch: {path}")
    ids = read_ids(path)
    if len(ids) != expected["count"] or len(ids) < count:
        raise ValueError(f"Unexpected training selection size: {path}")

    def rank(event):
        key = f"pulse-diagnostic-v1:{seed}:{database}:{event}".encode()
        return hashlib.sha256(key).digest(), event

    return sorted(sorted(ids, key=rank)[:count])


def quoted(column):
    return '"' + column.replace('"', '""') + '"'


def pulse_rows(connection, ids, label, record):
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({TABLE})")]
    if {"string", "dom_number"} <= set(columns):
        sensor = ["string", "dom_number"]
    else:
        sensor = ["dom_x", "dom_y", "dom_z"]  # Exact location proxy for single-PMT P14 DOMs.
    if "pmt_number" in columns:
        sensor.append("pmt_number")
    required = {"event_no", "dom_time", "charge", *sensor}
    if label:
        required.add(label)
    if required.difference(columns):
        raise ValueError(f"{TABLE} missing {sorted(required.difference(columns))}")
    record.update(sensor_key=sensor, pulse_columns=columns,
                  tables=[r[0] for r in connection.execute(
                      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")])
    # Single SELECT for all sampled IDs; SQLite can use an event index when
    # available and can scan the pulse table once when there is no index.
    connection.execute("CREATE TEMP TABLE sample (event_no INTEGER PRIMARY KEY)")
    connection.executemany("INSERT INTO sample VALUES (?)", [(e,) for e in ids])
    key_sql = ", ".join(map(quoted, sensor))
    flag = quoted(label) if label else "NULL"
    query = (f"SELECT event_no, {key_sql}, dom_time, charge, {flag} FROM {TABLE} "
             "WHERE event_no IN (SELECT event_no FROM sample) "
             f"ORDER BY event_no, {key_sql}, dom_time, rowid")
    return connection.execute(query)


def brightness_band(charge):
    return "<1" if charge < 1 else "1–<5" if charge < 5 else "≥5"


def analyze_event(database, event, rows, windows, labeled, pairs, brightest, anchors, sensor_columns):
    # Each pulse is (time, charge, optional binary aftereffect flag).
    sensors = {}
    for row in rows:
        key, (time, charge, flag) = tuple(row[1:-3]), row[-3:]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (*key, time, charge)):
            raise ValueError(f"Missing/non-finite pulse values in {database}, event {event}")
        if sensor_columns[0] == "string" and (key[0] <= 0 or key[1] <= 0):
            raise ValueError(f"Invalid physical sensor ID {key} in event {event}")
        if sensor_columns[0] == "dom_x" and key[:3] == (-1, -1, -1):
            raise ValueError(f"Padding coordinates in event {event}")
        if charge < 0 or (labeled and flag not in (0, 1)):
            raise ValueError(f"Invalid charge or binary pulse-origin flag in event {event}")
        sensors.setdefault(key, []).append((time, charge, flag))
    n_raw = sum(map(len, sensors.values()))
    q_raw = math.fsum(q for pulses in sensors.values() for _, q, _ in pulses)
    n_flag = sum(flag for pulses in sensors.values() for _, _, flag in pulses) if labeled else None
    for key, pulses in sensors.items():
        dom = "/".join(map(str, key))
        for p, later in zip(pulses, pulses[1:]):
            pairs.append((database, event, dom, later[0]-p[0], p[1], later[1], later[2]))
        # First in time/rowid order wins a brightest-charge tie.
        main = max(range(len(pulses)), key=lambda i: pulses[i][1])
        t0, q0, _ = pulses[main]
        anchors[brightness_band(q0)] += 1
        for i, (t, q, flag) in enumerate(pulses):
            if i != main:
                brightest.append((database, event, dom, t-t0, q0, q, flag))

    outcomes = []
    for width in windows:
        kept = 0
        shifted_charge = 0.0
        smaller_than_anchor = 0
        absorbed_flags = 0
        group_totals = []
        for pulses in sensors.values():
            anchor = None
            anchor_charge = 0.0
            group_charge = 0.0
            for t, q, flag in pulses:
                # Width zero is the identity, including exact timestamp ties.
                if anchor is None or width == 0 or t-anchor > width:
                    if anchor is not None:
                        group_totals.append(group_charge)
                    anchor, anchor_charge, group_charge = t, q, q
                    kept += 1
                else:
                    group_charge += q
                    shifted_charge += q
                    smaller_than_anchor += int(q < anchor_charge)
                    if labeled:
                        absorbed_flags += flag
            group_totals.append(group_charge)
        q_new = math.fsum(group_totals)
        if not math.isclose(q_new, q_raw, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Charge conservation failed for {database}, event {event}")
        outcomes.append(dict(database=database, event_no=event, window_ns=width, sensors=len(sensors),
                             raw_pulses=n_raw, merged_pulses=kept, absorbed_rows=n_raw-kept,
                             raw_charge_pe=q_raw, merged_charge_pe=q_new, shifted_charge_pe=shifted_charge,
                             absorbed_smaller_than_anchor=smaller_than_anchor,
                             above_cap_before=int(n_raw > 256), above_cap_after=int(kept > 256),
                             below_two_after=int(kept < 2), labeled_aftereffects=n_flag,
                             absorbed_aftereffects=absorbed_flags if labeled else None))
    return outcomes


def summarize(outcomes, windows, labeled):
    result = []
    for width in windows:
        rows = [r for r in outcomes if r["window_ns"] == width]
        def total(name):
            return sum(r[name] for r in rows)

        absorbed = total("absorbed_rows")
        summary = dict(window_ns=width, events=len(rows), changed_events=sum(r["absorbed_rows"] > 0 for r in rows),
                       raw_pulses=total("raw_pulses"), absorbed_rows=absorbed,
                       absorbed_fraction=absorbed/total("raw_pulses"), shifted_charge_pe=total("shifted_charge_pe"),
                       raw_charge_pe=total("raw_charge_pe"),
                       absorbed_smaller_than_anchor=total("absorbed_smaller_than_anchor"),
                       crossed_to_at_or_below_cap=total("above_cap_before")-total("above_cap_after"),
                       below_two_after=total("below_two_after"))
        if labeled:
            caught, positives = total("absorbed_aftereffects"), total("labeled_aftereffects")
            summary.update(labeled_aftereffects=positives, absorbed_aftereffects=caught,
                           absorbed_other_pulses=absorbed-caught,
                           precision=caught/absorbed if absorbed else None,
                           recall=caught/positives if positives else None)
        result.append(summary)
    return result


def timing_profile(brightest, anchors):
    """Count smaller companions per 100 sensor/event anchors, including quiet anchors."""
    counts = Counter()
    for _, _, _, delay, primary, other, _ in brightest:
        if delay != 0 and other < primary:
            index = int(np.searchsorted(GAPS, abs(delay), side="right") - 1)
            counts[brightness_band(primary), index, "after" if delay > 0 else "before"] += 1
    rows = []
    for band, n in sorted(anchors.items()):
        for i, (lo, hi) in enumerate(zip(GAPS, GAPS[1:])):
            before, after = counts[band, i, "before"], counts[band, i, "after"]
            rows.append(dict(brightest_charge_band_pe=band, sensor_event_anchors=n,
                             delay_min_ns=lo, delay_max_ns=hi, smaller_before=before, smaller_after=after,
                             before_per_100_anchors=100*before/n, after_per_100_anchors=100*after/n))
    return rows


def table_bytes(header, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode()


def dict_table(rows):
    columns = list(rows[0])
    return table_bytes(columns, ([r[c] for c in columns] for r in rows))


def diagnostic_figure(pairs, brightest, summary):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), layout="constrained")
    ticks = ["<1", "1–2", "2–5", "5–10", "10–25", "25–50", "50–100",
             "100–300", "300–1k", "1k–3k", "3k–11k", "≥11k"]
    counts, _, _ = np.histogram2d([p[3] for p in pairs], [p[5] for p in pairs], bins=(GAPS, CHARGES))
    image = axes[0].imshow(counts.T, origin="lower", aspect="auto", cmap="Blues",
                           norm=LogNorm(vmin=1, vmax=max(2, counts.max())))
    axes[0].set(yticks=range(len(CHARGES)-1), yticklabels=["<0.1", "0.1–0.25", "0.25–0.5", "0.5–1", "1–2", "2–5", "≥5"],
                ylabel="Later charge (PE)", title="Consecutive pulses on the same sensor")
    fig.colorbar(image, ax=axes[0], label="Pairs per bin")
    for side, label in [(1, "After brightest"), (-1, "Before brightest")]:
        counts, _ = np.histogram([side*p[3] for p in brightest if side*p[3] > 0], bins=GAPS)
        axes[1].plot(range(len(ticks)), counts, marker="o", label=label)
    axes[1].set(title="Other pulses relative to the brightest on each sensor (ties excluded)",
                ylabel="Pulses per interval")
    axes[1].legend(frameon=False)
    for ax in axes[:2]:
        ax.set(xticks=range(len(ticks)), xticklabels=ticks, xlabel="Time separation (ns, unequal bin widths)")
    axes[2].plot([r["window_ns"] for r in summary], [100*r["absorbed_fraction"] for r in summary], "o-")
    axes[2].set(xlabel="First-pulse merge window (ns)", ylabel="Pulse rows combined (%)",
                title="Simulated charge-preserving merge")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135)
    plt.close(fig)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CORE / "pilot_contract.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-database", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows-ns", nargs="+", type=float, default=[5, 10, 25, 50, 100])
    parser.add_argument("--aftereffect-column", help="Verified binary pulse-level origin label, 1=aftereffect, 0=other")
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix.lower() != ".zip":
        parser.error("Choose a new .zip output path")
    if args.per_database < 1 or any(not math.isfinite(w) or w < 0 for w in args.windows_ns):
        parser.error("Positive sample size and finite nonnegative window widths required")
    windows = sorted(set([0.0] + args.windows_ns))
    selections = json.loads(args.contract.read_text(encoding="utf-8-sig"))["selections"]
    outcomes, pairs, brightest, records = [], [], [], []
    anchors = Counter()
    for database in DATABASES:
        ids = selected_training_events(args.selection_root, selections, database, args.per_database, args.seed)
        path = (args.data_root / database).resolve()
        name = f"{path.stem}_train.csv"
        record = dict(database=database, bytes=path.stat().st_size, sha256=sha256(path),
                      training_selection=selections[name], sampled_events=len(ids))
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            rows = pulse_rows(conn, ids, args.aftereffect_column, record)
            seen = set()
            for event, group in itertools.groupby(rows, lambda r: r[0]):
                outcomes.extend(analyze_event(database, event, group, windows,
                                              bool(args.aftereffect_column), pairs, brightest, anchors, record["sensor_key"]))
                seen.add(event)
            if seen != set(ids):
                raise ValueError(f"{database}: sampled events missing pulse rows: {sorted(set(ids)-seen)[:10]}")
        records.append(record)
    summary = summarize(outcomes, windows, bool(args.aftereffect_column))
    manifest = dict(format_version=1, purpose="Aftereffect diagnostic on cleaned P14 training pulses",
                    slurm_job=os.getenv("SLURM_JOB_ID"), seed=args.seed, windows_ns=windows,
                    sample="Equal count per DB; lowest stable SHA256 ranks among checked training IDs",
                    label_column=args.aftereffect_column, databases=records, pulse_table=TABLE,
                    contract_sha256=sha256(args.contract), script_sha256=sha256(Path(__file__)),
                    selection_code_sha256=sha256(CORE / "contract.py"),
                    units="ns and PE", cap=256, anchor="earliest time; inclusive window; ties by SQLite rowid",
                    interpretation="Pulse rows consolidated, charge conserved. False-pulse rates require verified pulse-origin labels. Before/after counts depend on event readout and cleaning; they are descriptive.",
                    software=dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__,
                                  sqlite=sqlite3.sqlite_version))
    files = {"window_summary.csv": dict_table(summary), "event_windows.csv": dict_table(outcomes),
             "brightest_timing_profile.csv": dict_table(timing_profile(brightest, anchors)),
             "adjacent_pairs.csv": table_bytes(["database","event_no","sensor","gap_ns","leading_charge_pe",
                                                "later_charge_pe","later_aftereffect_label"], pairs),
             "brightest_neighbors.csv": table_bytes(["database","event_no","sensor","signed_delay_ns",
                                                     "brightest_charge_pe","other_charge_pe","other_aftereffect_label"], brightest),
             "timing_diagnostic.png": diagnostic_figure(pairs, brightest, summary)}
    manifest["members_sha256"] = {k: hashlib.sha256(v).hexdigest() for k,v in files.items()}
    files["manifest.json"] = json.dumps(manifest, indent=2, allow_nan=False).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise IOError("Result ZIP failed its CRC check")
    print(f"Return this ZIP: {args.output.resolve()}  (SHA256 {sha256(args.output)})")


if __name__ == "__main__":
    main()
