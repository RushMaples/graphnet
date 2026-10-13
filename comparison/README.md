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

## Recovering a deleted Carlos checkout

The `check` and Carlos training jobs need Carlos' source. The default frozen
selection directory also lived under that checkout. Ajai's fork cannot stand
in for Carlos' baseline. The folder name itself is not important; alternative
locations work through `CARLOS_ROOT` and `PILOT_SELECTION_ROOT`.

The pinned public revision `1d83975e21b5b1c7f02efb797d5fc39f5f5735c6`
contains all source classes imported by this comparison harness. A clone at
that revision is sufficient for a new Carlos baseline run. It does not
restore every uncommitted modification or checkpoint from the September job.

Upload the locally recovered `p14_pilot_frozen_selections.tar.gz` to Phoenix,
for example to your home directory. From the **new fork's root**, run:

```bash
git pull --ff-only origin main
/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet/bin/python comparison/recover_carlos.py "$HOME/p14_pilot_frozen_selections.tar.gz"
```

Then configure a current Slurm allocation and submit the preflight below.

The recovery command validates the archive, clones Carlos' `cursor` branch
into the original default location, checks out the pinned revision, and
restores all nine CSVs with their original checksums and membership. It refuses
to overwrite different files or an existing modified checkout. It does not
install packages or require a GPU. An ordinary Python 3 interpreter can run it
if the environment's interpreter is unavailable; the GPU jobs still require
the environment to be restored separately.

Deleting only `software/cursor` would not normally delete the separate
`software/envs/graphnet` environment or the source databases under Jiyuan's
directory. Confirm those paths still exist. If the old environment was also
deleted, recover an environment export/backup before rebuilding; do not guess
Torch/CUDA/PyG versions from current defaults.

The nine original CSVs are preserved locally under
`pilot_review_2026-09-30/bundle/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections`.
The recovery archive is a local transfer artifact, not committed to this
public repository. Do not regenerate the original splits using seed 42.

## On Phoenix: preflight first

### Select a current Slurm allocation

The archived pilot's account and QoS are historical records, not proof of
current access. The launcher no longer hardcodes `gts-pli77-ideas_l40s` or
`inferno`. An `Invalid account or account/partition combination` submission
error occurs before the script starts; restoring sources or reinstalling the
environment will not resolve it.

Inspect your associations and the cluster's available GPU types:

```bash
sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS
sinfo -h -o "%P %a %G"
```

On October 10, 2026, `rmaples3` reported associations for `gts-itaboada3`
and `gts-itaboada3-cygnus`, both with `embers,inferno`, and an active
`gpu-l40s` partition advertising `gpu:l40s`. Start with the general research
account and the pilot's QoS below. Association and inventory listings alone
do not establish partition access; require the test-only command to succeed.
Other users should substitute their own permitted settings.

```bash
export SBATCH_ACCOUNT="gts-itaboada3"
export SBATCH_PARTITION="gpu-l40s"
export SBATCH_QOS="inferno"
export SBATCH_GRES="gpu:l40s:1"
sbatch --test-only comparison/pace_compare.sbatch check
```

`--test-only` validates the resource request without submitting a job; it does
not run the Python preflight. Slurm accepts these environment variables for
all subsequent `sbatch` commands in the same shell. Command-line options take
precedence. Review any existing `SBATCH_*` variables if an unexpected setting
persists. See the [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html).

The script still requests the pilot's one L40S GPU. If your permitted partition
has a different GPU type, explicitly override `--gres=gpu:TYPE:1` for the check
and both training jobs, using the exact type reported by `sinfo`. Keep all
three jobs on the same GPU type; rerun preflight after changing hardware, and
do not attribute hardware differences from the historical pilot to model speed.

### Run the paired preflight

Clone/pull `RushMaples/graphnet` into a **separate** directory from the old
Carlos checkout. First check imports in the existing pilot environment on the
login node, without database reads, model construction, or GPU allocation:

```bash
git pull --ff-only origin main
/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet/bin/python comparison/check_imports.py
```

After both import checks pass, run the real-data GPU preflight:

```bash
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

## Diagnosing a failed preflight

The launcher writes `icemix-comparison-JOBID.out` in the submission directory,
not the default `slurm-JOBID.out`. For a `check` job, step `.0` runs Carlos and
step `.1` runs Ajai. Read the traceback before changing packages or resources:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,MaxRSS,Elapsed
tail -n 150 icemix-comparison-JOBID.out
```

Job 13945784 completed Carlos but failed while importing Ajai: the transformer
package eagerly imported the unrelated ISeeCube model, which requires
`torchscale`. The package now loads ISeeCube only when requested; IceMix does
not need that dependency. Pull the fix and rerun the paired `check` job in the
existing environment. No package installation is needed for this import fix.
Use the new successful job's receipt for training, since the source changed.

Job 13965604 passed Carlos' one-batch preflight, then failed on an Ajai import
of `lmdb`. The fork declares `lmdb>=1.4.1` in `setup.py`; adding this missing
dependency is appropriate. Avoid a broad environment upgrade. From the fork
root, capture the package versions and install only the missing package:

```bash
PILOT_PYTHON=/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet/bin/python
mkdir -p comparison_runs/environment-repair
"$PILOT_PYTHON" -m pip freeze > comparison_runs/environment-repair/before-lmdb.txt
"$PILOT_PYTHON" -m pip install --no-deps --only-binary=:all: "lmdb>=1.4.1"
"$PILOT_PYTHON" -m pip freeze > comparison_runs/environment-repair/after-lmdb.txt
"$PILOT_PYTHON" comparison/check_imports.py
```

The wheel-only installation fails rather than compiling if a compatible wheel
is unavailable. Both variants must use this same repaired environment and a
new successful paired preflight receipt. The old job did not produce a
successful paired receipt or ten-epoch comparison results. Import checks catch
Python dependency errors before queueing; they do not certify CUDA execution.

## Local checks

```bash
python -m unittest discover -s comparison -p 'test_*.py'
python -m compileall -q comparison
bash -n comparison/pace_compare.sbatch
```

The vMF tensor test requires the GraphNeT Torch environment. The GPU preflight
is authoritative for CUDA/PyG compatibility; local CPU/static checks cannot
establish that Phoenix will run successfully.
