# Frozen raw-model benchmark evaluation

Use the selected **epoch 31** checkpoint from `full-14534898/both`:
`best-epoch=30-val_loss=1.51333322.ckpt`, SHA256
`0d77854d1c72c0272bba9f8e01b8b4fb19265aa27f35d35ee276a2464cd2650c`.
The primary result uses the **raw model**, with no kappa scaling. Freeze this
checkpoint and protocol before running; do not tune, select, or calibrate on
the resulting test metrics.

This reuses the original **72,268-event test benchmark** from the frozen 10%
sample (29,000 numu, 20,305 nue, 22,963 nutau). Earlier Carlos/Ajai comparison
job `13992488` already evaluated these same test keys. Report this as a reused
benchmark, not an untouched independent test. No new selection is created.

## Submit on Phoenix

From the RushMaples/graphnet checkout after the new workflow files are available:

```bash
cd /storage/project/r-itaboada3-0/rmaples3/software/graphnet
git pull --ff-only origin main
bash comparison/submit_test.sh --dry-run
bash comparison/submit_test.sh
```

The second command submits one ordinary allocation: account `gts-itaboada3`,
QOS `inferno`, any L40S in `gpu-l40s`, one GPU, four CPUs, 64 GB, one-hour limit.
There are no node exclusions, holds, dependencies, or separate queued preflight.
The helper clears inherited scheduler options before submitting.
Record the printed job ID; do not submit duplicates while the job is pending.

The job uses the existing GraphNeT environment. Before staging data, a small
CUDA arithmetic, finite-result and synchronization check verifies the assigned
GPU and records `gpu_health.json`; failure stops before copying any databases.
It then stages the three original P14 databases to node-local temporary storage
and verifies the audited
completion/configuration/source/environment receipts, selected checkpoint,
all nine frozen split CSV hashes and split disjointness. Each staged database
is hashed once against the original database manifest.

Before constructing the test loader, it restores weights strictly and reruns
the first four original validation batches (512 events), preserving batch
composition. All seven predictions must agree with the archived validation
CSV, and all actual six-feature input hashes must match exactly. Fixed FP32
tolerances are `rtol=2e-5`, with absolute tolerances 0.002 m for position,
`2e-6` for direction components, and `2e-4` for kappa. These allow small FP32
kernel rounding differences in the same recorded software/GPU environment;
they are not test-derived. Any mismatch stops the job before test inference.

Inference uses `eval()`, `inference()` and `torch.inference_mode()` in FP32,
the same six features, 256-pulse cap, batch size 128, and deterministic
`p14-eval-v1` composite-key input policy. Training augmentation is inactive;
no training loader, optimizer, scheduler or fitting is run.

## Inspect and export

Replace `JOBID` with the submitted number:

```bash
squeue -j JOBID
tail -n 30 icemix-test-JOBID.out
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES
cat comparison_runs/test/test-JOBID/slurm_status.txt
cat comparison_runs/test/test-JOBID/results/complete.json
```

Success requires exit code zero **and** `results/complete.json` with
`complete: true`, `rows: 72268`, `unique_keys: 72268`, exact frozen keys and a
passed validation parity check. A partial predictions file alone is not success.

Outputs include `predictions.csv` (truth, seven raw predictions, event keys and
actual input hashes), checkpoint/input/output manifests, parity residuals,
`metrics.csv`, and `diagnostics.csv`. Diagnostics reuse the validation report's
pooled/track/cascade and energy/azimuth reporting: angle/vertex quantiles and
tails, direction NLL, joint loss, vMF coverage errors and cone sizes. Direction
uncertainty uses raw kappa; the validation-only scalar 0.88834 is not applied.
Driver and wrapper phase times, inference throughput, GPU allocated-memory
peak and process/worker RSS peaks are recorded; `sacct` complements the RSS
figures. The one-hour allocation limit is a budget, not an expected runtime.

Archive the results and log on Phoenix for download through the usual transfer
method; no weights or databases are copied into this archive:

```bash
tar -czf icemix-test-JOBID.tar.gz \
  comparison_runs/test/test-JOBID icemix-test-JOBID.out
```

Local checks cover protocol/math and launcher syntax. The actual GPU restore,
parity and benchmark inference run on Phoenix because the downloaded audit
bundle contains no weights or P14 databases. No benchmark job was submitted
while preparing this workflow.
