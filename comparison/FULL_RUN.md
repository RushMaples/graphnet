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

The helper validates the Slurm request without allocating resources, then
submits **one job on hold**. It queries the controller and releases that same
job only after confirming the job ID, pending/held state, zero priority and
accepted excluded hosts. Import, source, input and numerical checks then run
inside the allocation, followed by training and export. There is no separate
queued preflight job and no automatic follow-on job.

After two ECC failures on `atl1-1-01-004-31-0` (jobs 14421409 and 14503832),
the launcher **temporarily excludes that node**. The exclusion is present in
both the batch file's `#SBATCH` header and the helper's explicit Slurm options.
Job 14510941 was stopped by the runtime guard after allocation to that same
host; it did not initialize CUDA or start training. A runtime setting alone is
not proof that the scheduler accepted the exclusion, so the helper now checks
the controller's actual `ExcNodeList` before releasing a held submission.

If the controller response is unavailable, lacks the required exclusions or
does not confirm a pending held job, the helper **does not release it**. It
prints the job ID and an inspection message. Inspect that existing job instead
of submitting duplicates. Its submission/controller receipt is saved beside
the run directories as `submission-JOBID.txt`. An unexpected controller state
must be checked; the helper cannot assert that a scheduler which ignored the
hold has kept the job held.

The batch script also checks its assigned host before initializing CUDA or
staging databases. It captures the controller record at allocation start in
`slurm_job_at_start.txt`, including on a guard failure.

An initial CUDA allocation/kernel/synchronization check runs **before** the
large database copy and index scans. Its result is saved as `gpu_health.json`;
failure ends the allocation before model/data setup. The later real-model
preflight is retained. This check does not train or modify any checkpoint.

`EXCLUDE_NODES` can add other hosts to the blocked list; the known faulty node
is always retained. Empty lists and `EXCLUDE_NODES=none` are rejected during
this incident. After PACE confirms repair, remove the temporary block in a
reviewed launcher update, including the batch header and runtime guard. The
helper clears inherited `SBATCH_EXCLUDE` so it cannot silently contradict the
displayed selection. Excluding a host may increase queue time; it does not
request an additional GPU.

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
| `allocation.txt`, `gpu_health.json` | Assigned/excluded nodes and initial CUDA health check before database staging |
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
