# PACE Phoenix: directional comparison setup

Use the revised `directional-screen-kit-v2.zip`, not the previous kit. Upload
it and `directional-frozen-selections.zip` to your PACE home directory using
your usual transfer connection. The selections archive contains the original
nine CSVs, checked against `pilot_contract.json`; it contains no new split.
No training has been submitted by this preparation.

## 1. Shell, checkout, and transfer

After logging into Phoenix, enter Bash (your usual login may be tcsh):

```bash
bash
cd /storage/project/r-itaboada3-0/rmaples3/software/graphnet
git status --short
git rev-parse HEAD
```

The kit extends the existing RushMaples/graphnet checkout based on revision
`6cfc70eea87a8ad74373fddd16deb21e471cf29b`. It is an overlay, not a complete
GraphNeT checkout. `git pull` alone does not transfer these new local files.
The following overwrites only the new directional comparison files listed in
the kit. If you edited those files on PACE since receiving the first kit, save
your edits before extracting.

```bash
unzip -o ~/directional-screen-kit-v2.zip
mkdir -p /storage/project/r-itaboada3-0/rmaples3/inputs
unzip -n ~/directional-frozen-selections.zip \
  -d /storage/project/r-itaboada3-0/rmaples3/inputs
export PILOT_SELECTION_ROOT=/storage/project/r-itaboada3-0/rmaples3/inputs/p14_frozen_10pct
export SOURCE_DATA_ROOT=/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74
export ICEMIX_CHECKOUT="$PWD"
```

An existing alternative directory with the same nine original CSVs also works:
set `PILOT_SELECTION_ROOT` to it. Carlos source/checkpoints are unnecessary for
these three from-scratch models. Do not regenerate CSVs with seed42.

## 2. Clone the working environment once

```bash
module load anaconda3/2022.05.0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
export ENV_PREFIX=/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet-directional
```

If `graphnet-directional/bin/python` does not already exist, create it once:

```bash
conda create --yes --prefix "$ENV_PREFIX" \
  --clone /storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet
```

Then activate the clone and install the two pinned additions:

```bash
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/comparison"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/icemix-directional-${USER}/matplotlib"
"$ENV_PREFIX/bin/python" -m pip freeze > directional-before.txt
"$ENV_PREFIX/bin/python" -m pip install --no-deps --force-reinstall -r comparison/requirements-directional.txt
"$ENV_PREFIX/bin/python" -m pip freeze > directional-after.txt
diff -u directional-before.txt directional-after.txt || true
```

Expected changes are the pinned jammy_flows source and torchdiffeq0.2.5.
`--no-deps` prevents the resolver from changing Torch, NumPy or the compiled
PyG packages. It also avoids unused astronomy/contour dependencies. Jammy may
print warnings about absent healpy/astropy/mhealpy; those contour utilities
are not used here. Actual density/API errors must still be resolved.

Do not rebuild the environment with current Torch/CUDA defaults, install
GraphNeT again, or run a broad package upgrade. If the original working
environment is missing, recover its recorded package export first. If the
historical Anaconda module is unavailable, inspect `module avail anaconda3`
before substituting a module; a module change alone does not recreate the
working Torch/PyG environment.

## 3. Check before waiting for GPUs

```bash
bash -n comparison/pace_directional.sbatch
bash -n comparison/submit_directional.sh
"$ENV_PREFIX/bin/python" comparison/directional_run.py check-inputs \
  --selection-root "$PILOT_SELECTION_ROOT" --data-root "$SOURCE_DATA_ROOT"
"$ENV_PREFIX/bin/python" comparison/directional_run.py check-environment
"$ENV_PREFIX/bin/python" -m unittest discover -s comparison -p 'test_directional_workflow.py' -v
```

Require successful import/API/input checks and the original-vMF parity test
to pass. The parity test should execute rather than skip in the PACE training
environment. These checks do not train or scan full databases on the login node.
If the old `lmdb` import error recurs, repair only the cloned environment:

```bash
"$ENV_PREFIX/bin/python" -m pip install --no-deps --only-binary=:all: lmdb==1.4.1
```

Then rerun the checks. A `torchscale` error while importing IceMix indicates
old/wrong source is being imported: verify this fork's lazy transformer import
fix and the printed GraphNeT path before adding unrelated dependencies.

## 4. Validate scheduling, then submit once

```bash
sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS
sinfo -h -o "%P %a %G"
bash comparison/submit_directional.sh --dry-run
bash comparison/submit_directional.sh --test-only
```

Defaults are account `gts-itaboada3`, QOS `inferno`, partition `gpu-l40s`,
one L40S/task, four CPUs,64GB and12hours. The dry run only prints; test-only
asks Slurm to validate this resource request without submitting jobs. If the
account request fails, use an association you actually have via
`export DIRECTIONAL_ACCOUNT=...` and/or `export DIRECTIONAL_QOS=...`, then
repeat the test. Installing packages cannot fix an invalid account.

After the test succeeds:

```bash
bash comparison/submit_directional.sh
```

Record the returned array ID. It submits three tasks:0=vMF,1=King,2=flow.
The helper clears old `SBATCH_*` inputs; it requests no host exclusions,
reservation, dependency or hold. Each allocation performs a CUDA health test
before database copying, then a real eight-batch mixed-precision probe plus
validation/checkpoint reload. It verifies at least one actual optimizer update,
then rebuilds fresh weights/loaders and trains ten full epochs.

## 5. Watch the accepted array

Replace ARRAYID with the printed number:

```bash
squeue -j ARRAYID
sacct -j ARRAYID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 80 icemix-directional-ARRAYID-0.out
tail -n 80 icemix-directional-ARRAYID-1.out
tail -n 80 icemix-directional-ARRAYID-2.out
```

Results are `comparison_runs/directional/screen-ARRAYID/{vmf,king,flow}/model/`.
Success requires zero Slurm exit status and each `model/complete.json`, which
also confirms validation export. A pending job or a partial CSV is not a
reason to submit duplicates. ECC/CUDA failures are recorded in the allocation
log before data staging; inspect the assigned GPU failure without changing
model settings or reviving earlier host restrictions.

After all three finish:

```bash
python comparison/compare_directional.py comparison_runs/directional/screen-ARRAYID \
  --output-dir comparison_runs/directional/summary-ARRAYID
python comparison/plot_directional.py comparison_runs/directional/screen-ARRAYID \
  --output-dir comparison_runs/directional/summary-ARRAYID/figures
```

These are validation comparisons. The frozen test set remains separate; no
test inference or test-based model selection is submitted here.
