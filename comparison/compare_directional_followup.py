"""Three compact questions: count feature, joint head, vertex supervision.

All evaluation uses matched validation events. No test loader, extra seeds,
sampling convergence sweep or slide generation is performed.
"""
import argparse
import html
import json
import math
from pathlib import Path
import tarfile

from contract import environment_identity, sha256, write_json
from directional_followup import COUNTS, TASK_NAMES, read, settings, verify_plan


def load_export(entry):
    import numpy as np
    import pandas as pd
    folder = Path(entry["folder"])
    manifest_path = folder / "validation" / "prediction_manifest.json"
    manifest = read(manifest_path)
    csv_path = folder / "validation" / "predictions.csv"
    if (sha256(manifest_path) != entry["manifest_sha256"]
            or sha256(csv_path) != entry["prediction_sha256"]
            or manifest["sha256"] != entry["prediction_sha256"]
            or not manifest.get("complete") or manifest.get("rows") != COUNTS["validation"]
            or manifest.get("split") != "validation" or manifest.get("test_used") is not False
            or manifest.get("density_measure") != "per steradian"):
        raise ValueError("Invalid/changed validation export: " + str(folder))
    for kind, filename in (("settings", "resolved_settings.json"), ("provenance", "provenance.json"),
                           ("database_manifest", "database_manifest.json")):
        if manifest[kind+"_sha256"] != sha256(folder / filename):
            raise ValueError("Changed exported receipt: " + str(folder / filename))
    selected = entry["selected_checkpoint"]
    recorded = manifest["selected_checkpoint"]
    if selected["sha256"] != recorded["sha256"]:
        if not (entry.get("reused_same_epoch_export") and entry.get("weights_sha256")
                and selected["epoch"] == recorded["epoch"]):
            raise ValueError("Selected weights differ from inference receipt")
    frame = pd.read_csv(csv_path)
    if len(frame) != COUNTS["validation"] or frame.duplicated(["source_database", "event_no"]).any():
        raise ValueError("Incomplete/duplicate validation keys")
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite validation values")
    return frame.set_index(["source_database", "event_no"]).sort_index()


def require_same_inputs(reference, frame):
    fields = ["input_sha256", "n_pulses", "position_x", "position_y", "position_z",
              "azimuth", "zenith", "energy", "pid", "interaction_type", "oneweight"]
    if not reference[fields].equals(frame[fields]):
        raise ValueError("Evaluation truth, event keys, parent counts or actual pulse inputs differ")


def historical_frame(path, reference):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(path).set_index(["source_database", "event_no"]).sort_index()
    if not frame.index.is_unique:
        raise ValueError("Duplicate historical event")
    require_same_inputs(reference, frame)
    truth = reference[["direction_"+a for a in "xyz"]].to_numpy(copy=True)
    truth /= np.linalg.norm(truth, axis=1, keepdims=True)
    direction = frame[["dir_"+a+"_pred" for a in "xyz"]].to_numpy(copy=True)
    natural = direction*frame.direction_kappa.to_numpy()[:, None]
    k = np.linalg.norm(natural, axis=1)
    dot = (natural*truth).sum(axis=1)
    safe = np.maximum(k, 1e-4)
    regular = np.log(safe)-math.log(2*math.pi)-np.log(-np.expm1(-2*safe))+dot-k
    small = -math.log(4*math.pi)-k*k/6+k**4/180+dot
    frame["direction_nll"] = -np.where(k < 1e-4, small, regular)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    angle = np.arctan2(np.linalg.norm(np.cross(direction, truth), axis=1), (direction*truth).sum(axis=1))
    frame["angle_deg"] = np.degrees(angle)
    deficit = 1-np.cos(angle)
    frame["hpd_mass_at_truth"] = np.where(k < 1e-6, deficit/2,
        -np.expm1(-k*deficit)/np.maximum(-np.expm1(-2*k), 1e-300))
    frame["vertex_error_m"] = np.linalg.norm(
        frame[["position_"+a+"_pred" for a in "xyz"]].to_numpy()
        -frame[["position_"+a for a in "xyz"]].to_numpy(), axis=1)
    return frame


def summarize(frame, **identity):
    track = (abs(frame.pid) == 14) & (frame.interaction_type == 1)
    rows = []
    for topology, values in (("all", frame), ("track", frame[track]), ("cascade", frame[~track])):
        row = dict(identity, topology=topology, events=len(values), direction_nll_mean=values.direction_nll.mean(),
                   median_angle_deg=values.angle_deg.median(), p90_angle_deg=values.angle_deg.quantile(.9),
                   coverage68=(values.hpd_mass_at_truth <= .68).mean())
        if "vertex_error_m" in values:
            row["median_vertex_m"] = values.vertex_error_m.median()
        rows.append(row)
    return rows


def compare(plan_path, study_root):
    import numpy as np
    import pandas as pd
    plan_path, root = Path(plan_path), Path(study_root)
    plan = read(plan_path)
    verify_plan(plan)
    output = root / "summary"
    if output.exists():
        raise FileExistsError("Refusing to overwrite summary: " + str(output))
    models, frames, entries = {}, {}, {}
    reference = None
    for name in TASK_NAMES:
        folder = root / "runs" / name / "model"
        complete, cfg = read(folder / "complete.json"), read(folder / "resolved_settings.json")
        provenance = read(folder / "provenance.json")
        if (cfg != plan["runs"][name] or complete.get("run_name") != name
                or complete.get("protocol") != plan["protocol"]
                or complete.get("plan_sha256") != sha256(plan_path)
                or complete.get("settings_sha256") != sha256(folder / "resolved_settings.json")
                or complete.get("provenance_sha256") != sha256(folder / "provenance.json")
                or complete.get("test_exported") is not False or complete.get("test_rows") != 0
                or not complete.get("validation_exported") or not 10 <= complete.get("epochs", 0) <= 50
                or environment_identity(provenance) != plan["parent_environment"]
                or provenance["source_files"] != plan["source_files"]
                or read(folder / "initial_shared_state.json") != plan["common_initialization"]
                or read(folder / "input_probe.json") != plan["parent_input_probe"]
                or {r["database"]: r["sha256"] for r in read(folder / "database_manifest.json")} != plan["parent_databases"]):
            raise ValueError("Incompatible/incomplete follow-up run: " + name)
        if cfg["task"] == "joint" and name != "joint_king_no_count":
            required = ("epoch10/objective", "epoch10/direction", "final/objective", "final/direction")
        else:
            criterion = "objective" if cfg["task"] == "joint" else "direction"
            required = ("epoch10/"+criterion, "final/"+criterion)
        if not set(required) <= complete["exports"].keys():
            raise ValueError("Missing prespecified comparison export: " + name)
        models[name] = complete
        for tag in required:
            entry = complete["exports"][tag]
            frame = load_export(entry)
            if reference is None:
                reference = frame
            else:
                require_same_inputs(reference, frame)
            has_vertex = "vertex_error_m" in frame
            if has_vertex != (cfg["task"] == "joint"):
                raise ValueError("Mislabelled vertex outputs: " + name)
            frames[name, tag], entries[name, tag] = frame, entry
    king = plan["parents"]["king"]
    manifest_path = Path(king["folder"]) / "validation" / "prediction_manifest.json"
    old_manifest = read(manifest_path)
    count_on = load_export({"folder": king["folder"], "manifest_sha256": sha256(manifest_path),
                           "prediction_sha256": old_manifest["sha256"], "selected_checkpoint": king["selected"]})
    require_same_inputs(reference, count_on)
    old_csv = Path(plan["historical_root"]) / "validation" / "predictions.csv"
    for filename, expected in plan.get("historical_receipts", {}).items():
        if sha256(Path(plan["historical_root"]) / filename) != expected:
            raise ValueError("Historical completion/checkpoint receipt changed")
    if sha256(old_csv) != plan["historical_predictions_sha256"]:
        raise ValueError("Historical vMF validation export changed")
    historical = historical_frame(old_csv, reference)
    rows = summarize(count_on, question="count_epoch10", model="King with count", budget_epochs=10,
                     selected_epoch=king["selected"]["epoch"]+1)
    frame = frames["joint_king_no_count", "epoch10/objective"]
    rows += summarize(frame, question="count_epoch10", model="King without count", budget_epochs=10,
                      selected_epoch=entries["joint_king_no_count", "epoch10/objective"]["selected_checkpoint"]["epoch"]+1)
    for name in ("joint_vmf", "joint_king_no_count", "joint_flow"):
        entry = entries[name, "final/objective"]
        rows += summarize(frames[name, "final/objective"], question="joint_heads_final", model=name,
                          budget_epochs=entry["budget_epochs"], selected_epoch=entry["selected_checkpoint"]["epoch"]+1)
    rows += summarize(historical, question="historical_context", model="Historical vMF",
                      budget_epochs=plan["historical_epochs"], selected_epoch=plan["historical_selected_epoch"])
    for tag in ("epoch10/direction", "final/direction"):
        for name in ("joint_vmf", "direction_vmf", "joint_flow", "direction_flow"):
            entry = entries[name, tag]
            rows += summarize(frames[name, tag], question="vertex_"+tag.split("/")[0], model=name,
                              budget_epochs=entry["budget_epochs"], selected_epoch=entry["selected_checkpoint"]["epoch"]+1)
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes(include="number").drop(columns=["median_vertex_m"], errors="ignore")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Invalid numerical summary")
    output.mkdir()
    table.to_csv(output / "metrics.csv", index=False)
    overall = table[table.topology == "all"].copy()
    overall.to_csv(output / "headline.csv", index=False)
    limitations = ["One training seed; validation used for checkpoint selection and development.",
        "Same 50-epoch cap/stopping rule can yield different stopping epochs; display exposure explicitly.",
        "Ten-epoch vertex comparison is the matched fixed-budget diagnostic; final comparison also reflects stopping/scheduler trajectories.",
        "The vMF retains its original training approximation; all reported directional NLLs are exact per steradian.",
        "Flow mode/HPD/area use the unchanged 128-draw estimator; no sampling sweep was run.",
        "Truth-defined tracks mean abs(pid)==14 and interaction_type==1; remaining events are grouped as cascades.",
        "Historical epoch-31 vMF is descriptive context with a different source/training history.",
        plan["continuation_rng_limit"]]
    document = ["# Directional follow-up: three questions", "", "Validation only; 72,264 identical events; unweighted; seed 42.", ""]
    sections = [("1. Does the explicit count feature help King?", ["count_epoch10"]),
                ("2. Which joint head works best under the longer protocol?", ["joint_heads_final", "historical_context"]),
                ("3. Does removing vertex supervision help direction?", ["vertex_final"])]
    html_tables = []
    for title, questions in sections:
        selected = overall[overall.question.isin(questions)].copy()
        selected["coverage68"] *= 100
        cols = ["model", "budget_epochs", "selected_epoch", "direction_nll_mean", "median_angle_deg", "p90_angle_deg", "median_vertex_m", "coverage68"]
        selected = selected[cols]
        selected.columns = ["Model", "Trained epochs", "Selected epoch", "Direction NLL", "Median angle (deg)", "Angular p90 (deg)", "Median vertex (m)", "68% coverage (%)"]
        document += ["## "+title, "", selected.to_csv(index=False).strip(), ""]
        html_tables.append("<h2>"+html.escape(title)+"</h2>"+selected.to_html(index=False, na_rep="N/A", float_format=lambda x: "%.4f" % x))
    document += ["The complete CSV also preserves the fixed ten-epoch vertex comparison and track/cascade calibration.",
                 "Thirty-epoch checkpoints are retained without extra sampling exports.", "", "## Limits", ""] + ["- "+x for x in limitations]
    (output / "results.md").write_text("\n".join(document)+"\n", encoding="utf-8")
    page = "<!doctype html><meta charset='utf-8'><title>Directional follow-up</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#203245}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #d9e1e7}h2{margin-top:35px}th{background:#e7eef5}</style><h1>Directional follow-up: three questions</h1><p>Validation only; 72,264 identical events; unweighted; seed 42. Lower NLL/error is better. Vertex N/A means no vertex supervision/output.</p>"
    page += "".join(html_tables)+"<p>The fixed ten-epoch vertex comparison and topology breakdown are in headline.csv and metrics.csv.</p><h2>Limits</h2><ul>"+"".join("<li>"+html.escape(x)+"</li>" for x in limitations)+"</ul>"
    (output / "results.html").write_text(page, encoding="utf-8")
    write_json(output / "verification.json", {"matched_validation_events": COUNTS["validation"],
               "input_truth_and_count_match": True, "plan_sha256": sha256(plan_path),
               "test_used": False, "limits": limitations})
    # One compact figure; no deck or proliferation of per-bin figures.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    selectors = [("count_epoch10", "Count feature: 10 epochs"),
                 ("joint_heads_final", "Joint directional heads"),
                 ("vertex_final", "Vertex supervision: direction-NLL selection")]
    for ax, (question, title) in zip(axes, selectors):
        g = overall[overall.question == question]
        names = g.model.str.replace("joint_", "J: ", regex=False).str.replace("direction_", "D: ", regex=False).str.replace("king_no_count", "King", regex=False)
        ax.bar(range(len(g)), g.direction_nll_mean, color=["#2864a5", "#c77b25", "#1c8974", "#885a9d"][:len(g)])
        ax.set(xticks=range(len(g)), xticklabels=names, ylabel="Mean directional NLL", title=title)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.suptitle("One seed; validation-selected checkpoints; unchanged 128-draw flow estimator", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "three_questions.png", dpi=160)
    plt.close(fig)
    # Small analysis export: tables, receipts, configs and logs, not checkpoints
    # or the large repeated prediction tables already retained on PACE.
    archive = root / "directional-followup-results.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(plan_path, arcname="plan.json")
        bundle.add(output, arcname="summary")
        for name in TASK_NAMES:
            folder = root / "runs" / name / "model"
            for filename in ("complete.json", "resolved_settings.json", "provenance.json", "validation_history.json", "timing.csv"):
                bundle.add(folder / filename, arcname="runs/"+name+"/"+filename)
        for path in (root / "logs").glob("*.out"):
            # Slurm progress bars can dominate export size; users can inspect
            # the complete original log in place. Include terminal tails only.
            with path.open(errors="replace") as handle:
                tail = "".join(handle.readlines()[-100:])
            destination = output / (path.stem+"-tail.txt")
            destination.write_text(tail, encoding="utf-8")
            bundle.add(destination, arcname="logs/"+destination.name)
    print(overall.to_string(index=False))
    print("Analysis export:", archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    compare(args.plan, args.study_root)


if __name__ == "__main__":
    main()
