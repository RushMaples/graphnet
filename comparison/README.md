# Ten-epoch P14 comparison on PACE Phoenix

This compares Carlos' original joint model with Ajai's vertex-normalized model
using the successful September 25 pilot's data and hyperparameters. Each new
run trains **10 epochs from scratch**, not five additional epochs from the old
checkpoint. The old five-epoch result remains a historical reference.

The fork's three development branches were cumulative:
`main ddc54b10 -> port-icemix d1875055 -> perf-length-grouping 822fcd37 ->
fix-vertex-bias e4349812`. Main was fast-forwarded to the final commit; the
redundant branch names were removed with all commits retained in main.

## Preserved pilot contract

`pilot_reference.yaml` is the archived **training** configuration from job
13567223, not its inference-construction configuration. `pilot_contract.json`
records all nine frozen CSV SHA256 values and counts. No event lists or
predictions are committed here.

| Setting | Both new runs |
| --- | --- |
| Data | non-NuGen P14 part 1: numu, nue, nutau, in that order |
| Selection | exact original 10% lists; 578,121 / 72,264 / 72,268 train/validation/test |
| Horizon | 10 full epochs, no early stopping or resume |
| Seed / batch / accumulation | 42 / 128 / 1 |
| Workers / prefetch | 3 / 4, persistent workers, spawn |
| Hardware / precision | one L40S / 16-mixed training; FP32 prediction |
| Optimizer | AdamW, LR 6.25e-5, epsilon 1e-5, decay 0.01 |
| Scheduler | ReduceLROnPlateau val_loss, patience 6, factor 0.5 |
| Backbone | width 384, 4 relative + 12 ordinary blocks, Fourier base width 128 |
| Dropout / attention / projection / drop path | 0.1 / 0.05 / 0.1 / 0.2 |
| Pulse cap | 256 |
| Rotation / token dropping / EMA / multistage / WandB | disabled |
| Physical objective | 0.026 * vertex distance in metres + direction vMF NLL |
| Ajai vertex coefficient | 2.6 with target /100: physically equivalent to 0.026 |
| Checkpoint | minimum full-validation joint loss, explicitly reloaded |

Carlos retains random training batches; Ajai uses length grouping with factor
16. Grouping changes batch composition as well as cost, so this is a comparison
of the two implementations under a common training contract, not an isolated
proof of a normalization effect. Both keep the remainder: **4,517 steps/epoch**.
Validation/test use sequential batches. An optional later Ajai run with
`--group-factor 0` isolates grouping without changing the frozen inputs.

The harness calls the original model/loss classes from each checkout and uses
one shared training/export driver. It does not execute the old Hydra launcher.
It preserves Carlos' seven raw graph features (including pmt_area), whose
backbone encodes six; Ajai uses those six directly. Per-event fingerprints of
the six actual held-out model inputs must match across variants.

Two deliberate corrections apply to the new comparison: held-out pulse
sampling uses a stable composite `(database,event_no,seed)` CPU RNG seed, and
silent dropping of events with fewer than two pulses becomes a fatal check.
The prior pilot did not save pulse-row identities, so its exact capped inputs
cannot be recovered. Training pulse sampling stays stochastic. Ajai also has
a minimal small-kappa forward correction to make its existing vMF loss finite
at zero; this is recorded in Git and checked by the preflight.

## Environment and sources

The successful pilot used:

- `module load anaconda3/2022.05.0.1`
- `/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet`
- Carlos checkout `/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet`
- data `/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74`
- account `gts-pli77-ideas_l40s`, QoS `inferno`, one L40S, 4 CPUs, 64 GB, 12 hours.

The launcher initializes Conda explicitly and temporarily disables nounset for
activation. It captures Conda explicit packages, environment YAML and pip
freeze **without installing/upgrading anything**. Each variant runs in a fresh
process, overrides PYTHONPATH and asserts that GraphNeT imports from its own
source checkout. Do not install Ajai over the working Carlos environment.

The local pilot archive lacks the complete executed Carlos source snapshot
and package lock. The original PACE checkout/environment must still be present;
preflight will fail if they cannot supply the required classes. Source hashes
(including dirty/untracked Python files), revisions, dependency versions and
database hashes are saved. This is a controlled new comparison, not a claim
of byte-identical reproduction of the old job.

## On Phoenix: preflight first

Clone/pull `RushMaples/graphnet` into a **separate** directory from the old
Carlos checkout. From the new fork's root, run each command separately:

```bash
git pull --ff-only origin main
sbatch comparison/pace_compare.sbatch check
```

`check` stages the three databases to Slurm TMPDIR once, verifies exact split
hashes, disjointness, membership and schema, hashes the databases, exercises
PyG extensions plus real-data FP16 forward/backward, one actual Trainer step
with validation/callbacks/checkpoint reload and a six-event export, and matching input
fingerprints across variants (including capped validation events).

Inspect `icemix-comparison-JOBID.out` and
`comparison_runs/job-JOBID/{carlos,ajai}/preflight.json`. Require the final
`Matched ... validation input probes` message and successful Slurm exit status.
The preflight reads validation inputs; it does not choose a model on test data.

The successful check writes `paired_preflight.json`. Training requires this
receipt and rejects changes in sources, settings, environment or databases
since the check. Set the actual check job directory, then submit:

```bash
export COMPARISON_PREFLIGHT_ROOT="$PWD/comparison_runs/job-CHECK_JOBID"
sbatch comparison/pace_compare.sbatch carlos
sbatch comparison/pace_compare.sbatch ajai
```

Each job repeats its own checks, trains ten epochs and automatically exports
the validation-selected checkpoint on all frozen test events. No job has been
submitted by the local preparation work. The 12-hour resource request is kept
from the working pilot; actual ten-epoch duration is not yet measured.

Override paths by exporting `CARLOS_ROOT`, `ENV_PREFIX`, `SOURCE_DATA_ROOT`,
`PILOT_SELECTION_ROOT`, `ICEMIX_CHECKOUT` or `COMPARISON_RUN_ROOT` before sbatch.
If the archived selections are missing on PACE, copy the nine CSVs from the
local `pilot_review_2026-09-30/bundle/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections`
directory and point `PILOT_SELECTION_ROOT` there. **Do not regenerate them.**

## Compare completed jobs

Replace the two job directories with the actual successful run paths:

```bash
python comparison/evaluate.py comparison_runs/job-CARLOS_JOBID/carlos comparison_runs/job-AJAI_JOBID/ajai --output-dir comparison_summary
```

This requires matching settings, database content, composite test keys, truth
and actual pulse-feature fingerprints. Outputs include median/p68 angular and
3D vertex errors, signed coordinate biases, exact vMF cone coverage at several
levels, and track/cascade energy bins. Statistics are unweighted, as in the
pilot; oneweight is preserved but is not treated as a complete flux prescription.
Paired bootstrap intervals describe finite-event uncertainty, not seed or
detector-systematic uncertainty. Tune on validation; do not repeatedly select
architecture changes from this test set.

`timing.csv` records exposure, train/epoch seconds, throughput and peak GPU
memory. Startup staging/hash/probe costs are excluded from epoch timings.
Prediction timing explicitly includes loading, transfer and CSV export after
kernel warm-up; it is not an isolated model-latency benchmark. Compare quality
versus event exposure and GPU-hours; ten epochs still need not imply convergence.

## Local checks

```bash
python -m unittest discover -s comparison -p 'test_*.py'
python -m compileall -q comparison
bash -n comparison/pace_compare.sbatch
```

The vMF tensor test requires the GraphNeT Torch environment. The GPU preflight
is authoritative for CUDA/PyG compatibility; local CPU/static checks cannot
establish that Phoenix will run successfully.
