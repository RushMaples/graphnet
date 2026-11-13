# Directional uncertainty comparison on the frozen 10% P14 subset

Train **three new models from scratch for ten complete epochs**, seed 42:
current IceMix with its original vMF task, IceMix with a learned pulse-aware
King density, and IceMix with a four-layer spherical normalizing flow.
The epoch-31 model is a historical reference, not a fourth matched training arm.

The exact existing frozen CSV selections contain **578,121 training, 72,264
validation and 72,268 test events**. Training uses the training list only.
Validation selects each arm's checkpoint and supplies this initial screen's
diagnostics. The launcher never constructs a test loader or exports test
predictions. The existing test benchmark has already been used in earlier
comparisons; any later evaluation must describe it as a reused benchmark.

## Model and comparison contract

| Setting | All three arms |
|---|---|
| Inputs | Six existing xyz/time/charge/RDE features, SRTInIcePulses, cap 256 |
| Encoder | Current width-384 IceMix, 4 relative + 12 ordinary blocks |
| Initial state | Seed-42 identical backbone and vertex-head weights; SHA256 recorded |
| Augmentation | Existing z rotation about (46.3,-34.9) m; 5% of events receive independent 5% pulse dropout after cap |
| Vertex | LayerNorm CLS; original task predicts position/100 with loss factor 2.6 |
| Direction context | Raw CLS; King additionally sees log(cleaned parent pulse count /32) |
| Optimizer | AdamW LR6.25e-5, eps1e-5, decay.01 |
| Scheduler | ReduceLROnPlateau val_loss, patience6, factor.5 |
| Exposure | Ten full epochs, batch128, grouping16, no discarded remainder, no early stopping |
| Loading | Existing workers3, prefetch4, persistent spawn workers; deterministic held-out pulse sampling |
| Precision | Existing 16-mixed encoder; King integrals and flow density in FP64 outside AMP |
| Selection | Minimum full-validation family-specific joint loss; selected checkpoint strictly reloaded |
| Hardware | One L40S/task, 4 CPUs, 64 GB; any host in gpu-l40s |

The common physical objective is `0.026 * vertex_distance_metres + direction_loss`.
The original vMF training task and its high-concentration approximation remain
unchanged. A separate **exact per-steradian vMF log density** is recorded as
`direction_nll` for comparison with King and flow densities. Use that field
for cross-family likelihood comparisons, rather than assuming every recorded
training loss equals an exact spherical NLL.

All arms instantiate the original model first and then replace the directional
head. This keeps initial shared weights identical; `initial_shared_state.json`
records hashes. The global RNG is reseeded after head construction and after
preflight, and loaders are recreated. Head sizes and operations still differ,
so this is not a claim that all later dropout masks are bitwise identical.

### Learned King density

For angular separation theta in radians, the density per steradian is

```
q(theta | X) = [1 + theta^2 / (2 beta alpha^2)]^(-beta) / Z(alpha,beta)
Z = 2 pi integral_0^pi [1+t^2/(2 beta alpha^2)]^(-beta) sin(t) dt
```

A two-layer MLP receives raw CLS and `log(N_parent/32)` and predicts direction,
core scale `alpha=1e-4+softplus(a)` and tail `beta=.05+softplus(b)`. Parent
count means the cleaned pulse count **before** the cap and training dropout;
this agrees with the existing `data.n_pulses` convention and earlier King
calibration experiments. This arm is now trained end to end. It is not the
older post-hoc three-parameter calibration with frozen directions.

The full spherical integral uses differentiable Gauss-Legendre quadrature in
`s=log1p(theta/alpha)` to resolve narrow cores and broad tails. Refinement
checks compare 128 and 256 nodes during preflight and validation export.
The earlier untracked `experiments/king` scripts remain separate.

### Spherical flow prototype

The flow follows [IceCube's 2026 paper](https://arxiv.org/abs/2604.19846) and
its [official implementation](https://github.com/icecube/tnf_paper), using
`jammy_flows` version 1.1.0 from its official source at commit
`51f70764c12272715c3f47c593cccd1701d978c7`, the `s2` manifold, smooth spline/rotation blocks and a
final pure-vMF block. The prototype has **four** blocks; `FLOW_LAYERS=16`
selects the official head depth. The encoder, data and truth labels remain
those of this IceMix experiment, so neither depth is a reproduction of the
paper's full model or results. The paper uses a different encoder and lepton
targets; this comparison preserves existing neutrino-direction labels.

`force_embedding_coordinates=True` evaluates unit xyz with density measured
against solid angle. The head runs in double precision and gradients propagate
through the CLS cast. The existing GraphNeT `NormalizingFlow` wrapper is not
used: its Euclidean targets and scaling are unsuitable for this spherical task.

## Prepare the environment on Phoenix

For the ordered transfer/setup/check/submission commands, see
[PACE_QUICKSTART.md](PACE_QUICKSTART.md). Use the revised v2 transfer kit.

Use the same compatible environment for all arms. The launcher installs or
upgrades no packages. If jammy_flows is absent, first create a separate clone
of the working environment and install the pinned dependency there:

```bash
module load anaconda3/2022.05.0.1
conda create --yes --prefix /storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet-directional \
  --clone /storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet
export ENV_PREFIX=/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet-directional
"$ENV_PREFIX/bin/python" -m pip freeze > directional-before.txt
"$ENV_PREFIX/bin/python" -m pip install --no-deps --force-reinstall -r comparison/requirements-directional.txt
"$ENV_PREFIX/bin/python" -m pip freeze > directional-after.txt
"$ENV_PREFIX/bin/python" comparison/directional_run.py check-environment
```

Version 1.1.0 was not available from the inspected package index; the dependency
file therefore pins the official Git repository to the full commit above.
Its setup metadata declares version 1.1.0. The environment check requires both
that version and pip's recorded Git commit in `direct_url.json`, and saves the
source identity in the run provenance. An unrelated installation with the same
version label must be reinstalled from the pinned specification.

Start a Bash shell with `bash` before these commands if your PACE login uses
tcsh. Clone only once; if `graphnet-directional` already exists, reuse its
interpreter rather than cloning over it. The requirements also pin the tested
`torchdiffeq==0.2.5`. `--no-deps` preserves the working Torch/NumPy/PyG stack.
Jammy's optional astronomy/contour packages are omitted; the tested spherical
density, sampling and gradients do not need them. Do not use a generic
`pip install graphnet`, a broad upgrade, or a bare jammy install that resolves
all dependencies. The CPU check imports IceMix, lmdb-backed datasets and the
compiled PyG modules as well as the actual flow API.

Inspect the before/after dependency difference and check Torch/CUDA/PyG
compatibility. Package availability and the installed environment are not
verified by local preparation. The check exercises the actual spherical API
on CPU before any GPU job stages the databases; the allocation then exercises
the actual model, data, mixed-precision backward and checkpoint restore.
The original working environment remains available separately.

From the RushMaples/graphnet checkout, place these new files under `comparison/`
alongside the existing harness. The development base was commit
`6cfc70eea87a8ad74373fddd16deb21e471cf29b`. Keep the existing nine frozen CSVs;
do not regenerate the selections from seed alone.

## Submit the screen

From the repository root after copying the implementation:

```bash
"$ENV_PREFIX/bin/python" comparison/directional_run.py check-environment
bash comparison/submit_directional.sh --dry-run
bash comparison/submit_directional.sh --test-only
bash comparison/submit_directional.sh
```

This makes one Slurm submission with three array tasks: `0=vmf`, `1=king`,
`2=flow`. Each task receives its own ordinary allocation and independently
stages the three P14 databases to node-local temporary storage. The default
account is `gts-itaboada3`, QOS `inferno`, partition `gpu-l40s`, wall limit
12 hours. This limit is a budget, not a measured runtime guarantee. There are
no host exclusions, holds, reservations or dependencies. The helper clears
inherited `SBATCH_*` settings and old scheduler aliases before submission.
`--dry-run` prints the command. `--test-only` asks Slurm to validate the exact
resource/account request without submitting a job. Normal submission first
runs CPU import checks, verifies all nine frozen CSV hashes and split counts,
and checks that source database paths exist. This catches missing `lmdb`, PyG,
or deleted selection paths before waiting for GPUs. Full database content
audits and actual GPU execution still occur inside each allocation.

Supported overrides include `ENV_PREFIX`, `ICEMIX_CHECKOUT`, `SOURCE_DATA_ROOT`,
`PILOT_SELECTION_ROOT`, `DIRECTIONAL_RUN_ROOT`, `DIRECTIONAL_ACCOUNT`,
`DIRECTIONAL_QOS`, `DIRECTIONAL_TIME`, `FLOW_LAYERS` and `POSTERIOR_SAMPLES`.
Use the same settings for the entire array. Paths and account access must
still be valid on Phoenix when you submit. No jobs were submitted during
this preparation.

Each allocation checks CUDA before staging data, records environment exports,
audits all frozen selections and data membership/schema/checksums, and runs
eight real training batches and one validation batch with a saved/restored
checkpoint. It requires a successful optimizer update and changed parameter
hashes, then
reconstructs fresh weights and loaders before the ten-epoch fit. The numerical
probe is excluded from the actual training exposure.
Normal AMP overflows are handled by GradScaler skipping the update and
reducing its scale; successful updates and overflow batches are recorded.
Nonfinite losses, nonfinite probe parameters or an epoch with no successful
updates are fatal. The default environment path is the cloned
`software/envs/graphnet-directional` environment.

## Outputs, inspection and later TANGO-style plots

Results live under `comparison_runs/directional/screen-ARRAYID/ARM/model/`.
Logs are `icemix-directional-ARRAYID-TASKID.out`. Inspect:

```bash
squeue -j ARRAYID
sacct -j ARRAYID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 10 comparison_runs/directional/screen-ARRAYID/vmf/model/timing.csv
cat comparison_runs/directional/screen-ARRAYID/vmf/model/complete.json
```

| Artifact | Contents |
|---|---|
| `resolved_settings.json` | Model family, frozen training settings and count policy |
| `initial_shared_state.json` | Starting backbone/vertex hashes, checked again after preflight |
| `provenance.json`, `database_manifest.json`, `input_probe.json` | Executed source hashes, runtime packages, complete database hashes and actual validation input fingerprints |
| `preflight.json`, `preflight/` | Real Trainer/backward/checkpoint probe |
| `timing.csv` | Per-epoch counts, steps, LR, time, throughput, peak allocated GPU memory |
| `logs/training/version_0/metrics.csv` | Train/validation total loss, direction training loss, exact directional NLL and mean vertex error |
| `selected_checkpoint.json`, `checkpoints/` | Validation-selected and resumable last weights |
| `validation/predictions.csv` | Keys, truth, input fingerprints, parent counts, predicted position/direction, exact NLL and uncertainty diagnostics |
| `validation/prediction_manifest.json` | Full validation membership, output hash, selected checkpoint and provenance bindings |
| `complete.json` | Ten completed epochs and whether the requested validation export also succeeded |

The launcher enables validation export with 128 posterior samples/event for
flow. Its point prediction is the maximum-density posterior sample, a
**sampled MAP approximation**. HPD coverage uses the fraction of posterior
draws with density above the truth density; HPD areas use an importance
estimate under those draws. Both have Monte Carlo error. A later precision
study can increase `POSTERIOR_SAMPLES`; do not interpret sample-count changes
as an architecture gain. King/vMF region masses and areas use spherical cones.

`cone_radius_0.5_deg` is an exact 50%-mass cone for the isotropic heads. For
flow it is the sampled median opening angle about the sampled mode, which
differs from its HPD region. The export records this distinction.

After all three tasks complete, validate their receipts and make the plots:

```bash
python comparison/compare_directional.py comparison_runs/directional/screen-ARRAYID \
  --output-dir comparison_runs/directional/summary-ARRAYID
python comparison/plot_directional.py comparison_runs/directional/screen-ARRAYID \
  --output-dir comparison_runs/directional/summary-ARRAYID/figures
```

The analysis scripts require pandas and matplotlib in the analysis environment;
they do not require a GPU. Copy the run outputs to your analysis computer if
preferred. Keep all three arms together, with their provenance and selected
checkpoint receipts, when transferring outputs. The source transfer kit
contains only the new comparison implementation and analysis files; extract
it into the existing repository root so its `comparison/` paths merge with
the tracked harness. It does not contain data, frozen CSVs, dependencies or
model weights.

Future plots can use the exported energy, topology, parent pulse count, xyz
truth/predictions, angular errors, density ranks and region areas. Match
Jiyuan's TANGO technical report with median error versus energy in the upper
panel and the ratio to the vMF baseline in the lower panel, split by tracks
and cascades; include z/rho vertex errors, angular errors, coverage and size.
Training curves should distinguish the exact NLL diagnostic from the original
vMF training objective. Confirm all arms' source/data/input and initialization
receipts match before interpreting the comparisons. Validation is a development
sample, so its uncertainty summaries are not an independent calibration test.
The shared `timing.csv` field `optimizer_steps` counts attempted updates
(processed batches). `train_successful_optimizer_steps` and
`train_amp_overflow_batches` in the training logger, plus allocation totals
in `complete.json`, distinguish actual updates from AMP skips.

## Verification boundary

Preparation on 9 November 2026 passed seven CPU density tests, including the
actual spherical flow API from the pinned official source, two protocol tests
and the CPU GradScaler recovery/update-counter test,
all new Python syntax and CLI checks, and Bash syntax-tree parsing. Plotting
and the three-arm summary were exercised with synthetic fixtures; mismatched
truth and shared initialization were rejected. These fixtures are disposable
test inputs, not research results. The optional full-GraphNeT parity test was
skipped locally because the compiled training environment is unavailable.
Git Bash execution is blocked in the local sandbox; run `bash -n` on Phoenix
before submission as shown below.

```bash
bash -n comparison/pace_directional.sbatch
bash -n comparison/submit_directional.sh
python -m unittest discover -s comparison -p 'test_directional_*.py'
```

Local checks cover Python syntax, argument parsing, the independent spherical
density tests and any shell syntax checks supported by the local shell.
Full GraphNeT/PyG/Lightning training, P14 membership scans and GPU compatibility
require the Phoenix environment and actual data. Require both zero Slurm exit
status and `model/complete.json` for each arm. A partial CSV is not completion.
