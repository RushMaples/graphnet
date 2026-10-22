# P14 augmentation comparison and full training

Use Ajai's IceMix for all arms. Keep length grouping factor 16, the working
pilot environment, and the **same frozen 10% data selection**. Never generate
a new split or select 10% of the already reduced lists. Counts remain
578,121 training / 72,264 validation / 72,268 test events.

## Experiments

| Arm | Pulse dropout | Online azimuth rotation |
| --- | --- | --- |
| baseline | off | off |
| pulse-dropout | Carlos: token_drop=0.05, drop_chance=0.05 | off |
| rotation | off | on |
| both | Carlos: token_drop=0.05, drop_chance=0.05 | on |

The two probabilities mean select 5% of events, then independently remove
each pulse with probability 5% within selected events (~0.25% overall).
Retain at least one pulse per event. Dropout occurs after the existing
stochastic 256-pulse cap and before Fourier/relative-position encoding.
The count embedding uses the retained count. Inputs and targets are not
mutated by the backbone's pulse dropout. Evaluation always disables it.

Rotation uses Ajai's existing callback and DeepCore pivot (46.3, -34.9)m,
with node coordinates scaled by /500 and physical vertex labels in metres.
Direction vectors and azimuth labels rotate together with pulse/vertex x/y.
This differs from Carlos's origin-centred rotation. It tests robustness to
azimuthal orientation, not an exact detector/ice symmetry. No precomputed
augmented database or expanded event-ID lists are used. See
[source audit](augmentation_audit.md) for the distinction between supported
Carlos configurations and augmentations actually enabled in the old pilots.

## Training and selection

- Four **10-epoch** screen runs, each from scratch with seed 42. Same AdamW,
  batch 128, pulse cap 256, regularization, precision and group factor 16.
- Each arm runs an inline real-data FP16 backward and one-batch Trainer smoke
  test before training; no separately queued preflight is needed.
- Smoke model, loaders, optimizer state and RNG exposure are discarded before
  fitting. Augmentation RNG is independently seeded by epoch and batch.
- Full unaugmented validation selects the minimum joint-loss checkpoint per
  arm, then minimum validation joint loss selects the arm. Exact ties favour
  baseline, pulse-dropout, rotation, both in that order.
- Screening exports validation predictions/metrics only. All arms must have
  identical validation event keys, truth, retained pulse fingerprints, source
  hashes, environments, input probes, and database hashes. Summaries also show
  angular/vertex median errors, direction coverage and energy/topology bins.
- The selected arm trains **80 epochs from scratch**, on that same 10% sample,
  with the same settings and a fresh optimizer. It is not 80 additional epochs
  after the screen. The validation-selected checkpoint is reloaded and exported
  on validation and the frozen test sample at the end. No early stopping.
- One short seed is a screening result, not proof of small performance gains.
  Prefer the simpler arm when differences are inconclusive; for manual review,
  submit `screen` only rather than the automatic two-job helper.

## Submit on Phoenix

From the RushMaples/graphnet checkout, after pulling these changes:

```bash
git pull --ff-only origin main
bash comparison/submit_augmentation.sh --dry-run
bash comparison/submit_augmentation.sh
```

The helper defaults to the previously working research allocation:
`gts-itaboada3`, `gpu-l40s`, `inferno`, one `gpu:l40s:1`, 4 CPUs and 64 GB.
It first calls Slurm `--test-only`, then submits the screen and an
`afterok` dependent full job. The dependent job is cancelled if the dependency
becomes invalid. Both allocations request 12 hours by default. The old
IceMix measurements suggest roughly one hour per ten epochs, but rotation,
longer-run validation and cluster conditions may change this estimate.
There are two scheduler waits, with preflight included in each allocation.

For a manual decision before the 80-epoch job:

```bash
sbatch --account=gts-itaboada3 --partition=gpu-l40s --qos=inferno comparison/pace_augment.sbatch screen
# Read screen-JOBID/summary.csv and selected_arm.json, then:
sbatch --account=gts-itaboada3 --partition=gpu-l40s --qos=inferno comparison/pace_augment.sbatch full "$PWD/comparison_runs/augmentation/screen-JOBID"
```

Overrides: `ENV_PREFIX`, `SOURCE_DATA_ROOT`, `PILOT_SELECTION_ROOT`,
`ICEMIX_CHECKOUT`, `AUGMENTATION_RUN_ROOT`, `SBATCH_ACCOUNT`,
`SBATCH_PARTITION`, `SBATCH_QOS`, `SBATCH_GRES`; plus `SCREEN_EPOCHS`,
`FULL_EPOCHS`, `SCREEN_TIME`, `FULL_TIME`, `TOKEN_DROP`, `DROP_CHANCE`.
Do not change these during a screen/full pair. The full run rejects source,
data, environment, GPU model and common-setting changes since screening.
The Carlos checkout is needed only as the default home of the frozen CSVs;
no Carlos model imports are used. Set PILOT_SELECTION_ROOT if those CSVs live
elsewhere. No packages are installed or upgraded on Phoenix by these scripts.

## Inspect and collect

Logs: `icemix-augmentation-JOBID.out` in the submission directory.
Use `sacct -j JOBID --format=JobID,State,ExitCode,MaxRSS,Elapsed`.

Screen directory: `comparison_runs/augmentation/screen-JOBID/` with
`summary.csv`, `selected_arm.json`, and four arm directories. Each arm saves
resolved settings, source/environment/data provenance, preflight result,
model configuration, best/last checkpoints, epoch timing, training logs,
and `validation/predictions.csv`, its manifest, and binned `metrics.csv`.

Full directory: `comparison_runs/augmentation/full-JOBID/selected/` with
the same artifacts, plus `predictions.csv`, `prediction_manifest.json`,
`test_metrics.csv`, and `screen_selection.json`. `complete.json` is written
only after all expected epochs and prediction exports finish. `last.ckpt`
is preserved if interrupted; this driver refuses to overwrite or silently
restart partial runs. Resume support is not provided by this launcher.

```bash
tar -czf icemix-augmentation-results.tar.gz \
  --exclude='*.ckpt' --exclude='length_cache' \
  comparison_runs/augmentation/screen-SCREEN_JOBID \
  comparison_runs/augmentation/full-FULL_JOBID \
  icemix-augmentation-SCREEN_JOBID.out icemix-augmentation-FULL_JOBID.out
```
