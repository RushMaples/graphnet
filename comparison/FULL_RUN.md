# Early-stopped full run on the frozen 10% sample

Continue the **combined augmentation** model from the completed 30-epoch run.
The maximum is **80 total epochs**, not 80 additional epochs. At most 50 epochs
are added. The training data remain the exact frozen 10% selection: 578,121
training and 72,264 validation events. No test predictions are produced.

## Submit on Phoenix

From the RushMaples/graphnet repository root:

```bash
git pull --ff-only origin main
bash comparison/submit_full.sh comparison_runs/augmentation/continue-14388200 --dry-run
bash comparison/submit_full.sh comparison_runs/augmentation/continue-14388200
```

The helper submits **one ordinary job eligible for any L40S node** in
`gpu-l40s`. It requests no specific host, excluded hosts, feature constraint,
reservation, dependency or user hold. It makes one `sbatch` call; there is no
queued preflight job, test-only scheduling request, hold/release sequence or
automatic follow-on job. The dry run prints the request without calling Slurm.

The temporary ECC-host exclusion and host-name rejection have been removed at
the user's request. This includes `atl1-1-01-004-31-0`, where earlier jobs
encountered ECC errors. The early CUDA health check below remains: a node is
accepted regardless of its name, but an unusable assigned GPU still stops the
job before database staging or training.

The helper clears inherited `SBATCH_*` settings after resolving its supported
account/QOS overrides, and clears the `SLURM_CLUSTERS`, `SLURM_HINT` and legacy
`SLURM_MEM_BIND` input aliases. Old shell settings cannot silently reintroduce a hold, exclusion,
host list, reservation or conflicting resource request. The obsolete
`EXCLUDE_NODES` setting is ignored. The partition and GPU type are fixed to
`gpu-l40s` and `gpu:l40s:1`; account, QOS and wall-time overrides remain
available. Use the helper rather than a bare `sbatch` call to get this cleanup.

After submission, the helper saves the job ID and a read-only controller
snapshot in `submission-JOBID.txt` beside the run directories. Failure to read
that snapshot does not cancel or hold an accepted job. If the controller adds
exclusions or other restrictions, those are site-side settings: the helper
reports them without changing them. Consult PACE about unexpected controller
restrictions instead of submitting duplicates. Scheduler access rules,
availability and priority still determine when the job starts.

The batch script records its assigned node and captures the controller record
at allocation start in `slurm_job_at_start.txt`. It has no host-based guard.

An initial CUDA allocation/kernel/synchronization check runs **before** the
large database copy and index scans. Its result is saved as `gpu_health.json`;
failure ends the allocation before model/data setup. The later real-model
preflight is retained. This check does not train or modify any checkpoint.

To replace the previously blocked job 14516841, cancel only that job before
the submission command above:

```bash
scancel 14516841
sacct -j 14516841 --format=JobID,State,ExitCode,Elapsed
```

Accounting may take a moment to reflect cancellation. Check that the old job
is cancelled or otherwise terminal before submitting its replacement. Cancelling
the pending allocation does not remove the completed 30-epoch checkpoint.

Defaults are one L40S, four CPUs, 64 GB, account `gts-itaboada3`, partition
`gpu-l40s`, QOS `inferno`, and an **eight-hour wall-time limit**. At the measured
epoch rate, fifty additional epochs would take about 5.2 GPU-hours;
startup, export and the added diagnostics require overhead. Plan roughly
5.5-6 hours if the cap is reached. Earlier stopping reduces usage. These are
usage estimates, not prices or a guarantee of improvement.

The parent `both/checkpoints/last.ckpt` and selected best checkpoint must remain
on Phoenix. The original `screen-14290300/both` metadata must also remain
readable: its hash-bound timing history seeds the early-stopping counter.
The driver accepts the recorded ancestor location or a relocated sibling run
tree. Downloaded analysis archives intentionally omit weights and cannot
replace a resumable checkpoint.

Use the existing environment and model code. The launcher does not install or
upgrade packages. All previously recorded source files must match; the new
full-run files are additions. Existing path/account overrides remain available.

## Exact stopping rule

- Monitor the unaugmented validation **joint loss**, `val_loss`, in `min` mode.
- A significant improvement is a decrease of **more than 0.0001 absolute**
  relative to the last significant best score.
- Stop after **12 consecutive validation epochs** without such improvement,
  or at epoch 80, whichever happens first.
- Keep the existing learning-rate scheduler: patience six, factor 0.5. The
  longer stopping patience leaves time to respond to a learning-rate reduction.
- Replay the existing validation history; do not reset the stopping counter
  merely because a new allocation begins. For this parent the initial best is
  1.5240609891749666 and the initial wait count is one. With no new significant
  improvement, the earliest stop is at **41 total epochs**.
- Checkpoint selection still keeps the **lowest raw validation loss** over
  parent and new epochs. `min_delta` controls stopping, not which weights win.

An early stop is a successful outcome. `complete.json` records the actual
completed horizon separately from the maximum, with a stop reason of
`early_stopping` or `max_epochs`. An interrupted, partial or nonfinite run is
not silently called complete. The parent best remains eligible if none of the
new epochs beats it.

The model, optimizer, scheduler, loop progress and mixed-precision scaler are
restored from the parent's **last** checkpoint. Loader workers restart; this
is not a claim of bitwise identity with an uninterrupted run. Fitted weights
are reconstructed and reseeded after a separate numerical/diagnostic probe.

## What is tracked

Results are under `comparison_runs/augmentation/full-JOBID/`, with the model's
artifacts in `both/`. The console log is `icemix-augmentation-full-JOBID.out`.

| File | Purpose |
|---|---|
| `summary.csv`, `full_summary.json` | Selected model, actual/max epochs, stop reason, pooled validation metrics and timing |
| `both/timing.csv` | Each added epoch: training/validation exposure, updates, LR, loss, train/epoch seconds, throughput, peak allocated GPU memory |
| `both/training_curve.csv` | All ancestor and added epoch timing/loss records in one file |
| `both/logs/training/version_0/metrics.csv` | Existing train/validation direction, vertex, kappa and scaled-residual summaries |
| `both/epoch_diagnostics.csv` | Per-epoch validation topology/energy/azimuth errors, p95/p99, coverage, confidence and region-size diagnostics |
| `both/validation/predictions.csv` | Final selected-checkpoint FP32 predictions with event IDs, truth and actual input hashes |
| `both/validation/diagnostics.csv`, `validation_diagnostics.csv` | Detailed final validation diagnostics |
| `both/validation/prediction_manifest.json` | Selected checkpoint, export hash and loading/transfer/CSV-inclusive inference throughput |
| `both/early_stopping.json` | Policy, carried initial state, final state and reason |
| `both/wall_timing.json` | Driver startup, setup/probes, fitting, checkpoint selection, export and final diagnostics |
| `phase_timing.csv`, `slurm_status.txt` | Allocation wrapper phases, database staging, exit code and wall duration |
| `allocation.txt`, `gpu_health.json` | Assigned node, unrestricted host policy and initial CUDA health check before database staging |
| `slurm_job_at_start.txt` | Controller's job record at allocation time, for diagnosing scheduler/exclusion problems |
| `gpu_telemetry.csv` | Assigned-GPU utilization, memory, power and temperature sampled every 30 seconds when available |
| `both/checkpoints/` | Resumable last and selected/best weights |
| Settings, provenance, input/database and checkpoint receipts | Reproducibility and integrity checks |

Epoch diagnostics reuse the loss recorders' predictions; they do not perform a
second full validation inference. A small separate real-data probe checks the
new callback before fitting. Epoch timing includes diagnostic CPU reductions,
so compare it with earlier runtimes with that overhead in mind. Training
throughput and final inference throughput are different measurements.

Epoch diagnostics use mixed-precision validation; final exports use FP32.
Legacy reported-kappa coverage is retained for comparison with previous runs.
Additional **effective-concentration** fields account for the actual norm of
the direction vector in the vMF loss; they are labeled separately. Raw
recorder values are preserved for loss reconstruction. Region radii/solid
angles accompany coverage so that better coverage is not confused with merely
wider regions.

Fields named `cone_radius_...` and `cone_solid_angle_...` use the reported
kappa. Their `effective_cone_...` counterparts use the concentration of the
actual natural-parameter vector. Epoch reconstructed loss is labeled
`recorder_prediction_joint_loss`; the final FP32 export retains
`fp32_prediction_joint_loss`.

GPU telemetry is optional. If the assigned GPU cannot be queried, inspect
`gpu_telemetry.stderr`; per-epoch allocated memory and timing still remain.
The successful driver completion, rather than the presence of telemetry,
determines whether training succeeded.

## Inspect and download

While running, replace JOBID with the submitted number:

```bash
tail -n 8 comparison_runs/augmentation/full-JOBID/both/timing.csv
tail -n 30 icemix-augmentation-full-JOBID.out
```

After completion:

```bash
column -s, -t < comparison_runs/augmentation/full-JOBID/summary.csv
cat comparison_runs/augmentation/full-JOBID/both/early_stopping.json
cat comparison_runs/augmentation/full-JOBID/slurm_status.txt
tar -czf augmentation-full-JOBID.tar.gz \
  --exclude='*.ckpt' --exclude='length_cache' \
  comparison_runs/augmentation/full-JOBID \
  icemix-augmentation-full-JOBID.out
```

Keep the weights on Phoenix; the smaller archive retains predictions,
diagnostics, timing and provenance for review. Freeze model/training choices
before a separate final held-out test evaluation. No test-based stopping or
selection is added by this job.

## Optional overrides

`FULL_EPOCHS` sets a smaller total cap, `EARLY_STOP_PATIENCE` and
`EARLY_STOP_MIN_DELTA` change the stopping policy, and `FULL_TIME` changes the
Slurm wall limit. Any policy changes are recorded. Defaults implement the
80-epoch ceiling and 12-epoch patience described above; do not change model,
batching, environment or input selections during this continuation.

## Local verification limit

CPU protocol/math tests, archive-history replay and shell dry-run checks can be
performed locally. A full Lightning/GPU resume cannot be tested from the
downloaded archive because it excludes weights and the local runtime lacks
the compiled GraphNeT dependencies. The Phoenix job performs real-data probes
before fitting and refuses changed sources, inputs, environment or incomplete
resume state.
