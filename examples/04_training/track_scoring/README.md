# Track-scoring learner

This directory is a guided workflow for training the TANGO track/cascade
classifier on approximately 10% of the classification data used in the TANGO
technical report. It favors explicit, inspectable steps over heavy automation.

The learning problem is binary classification:

- `track = 1`: a muon-neutrino charged-current (`nu_mu CC`) event;
- `track = 0`: an electron-neutrino event or a muon-neutrino neutral-current
  (`nu_mu NC`) event;
- model output: one value in `[0, 1]` called `track_score`;
- training loss: binary cross-entropy.

The workflow components are:

1. `01_make_splits.py`: inspect SQLite inputs, select the intended event
   classes, draw a reproducible 10% sample, and save a CSV manifest.
2. `02_train_track_scorer.py`: turn pulses into graphs, build DynEdgeTITO,
   train the classifier, save checkpoints, and write validation predictions.
3. `03_evaluate_track_scorer.py`: calculate metrics and diagnostic plots.
4. `pace_prepare_track_scorer.sbatch`: create the manifest on a PACE CPU node.
5. `pace_train_track_scorer.sbatch`: run a GPU preflight, smoke test, or full
   training job according to `RUN_MODE`.

## What comes from GraphNeT

The training script follows `examples/04_training/02_train_tito_model.py`:

```text
data loaders -> graph definition -> DynEdgeTITO backbone
             -> task and loss -> StandardModel -> fit/predict/save
```

Direction reconstruction is replaced by `BinaryClassificationTask` and
`BinaryCrossEntropyLoss`. The graph definition follows the IceCube-86
track/cascade inference workflow. Training and inference must retain the same
feature order and detector preprocessing.

## Exact PACE inputs

The supplied path notation expands to these six files:

```text
/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_numu_database_part_1.db
/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nue_database_part_1.db
/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nutau_database_part_1.db
/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_numu_database_part_1.db
/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nue_database_part_1.db
/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nutau_database_part_1.db
```

The parentheses in `P14_(nugen_)nu(mu/e/tau)_database_part_1.db` describe
filename alternatives; they are not literal path characters. Underscores do
not need escaping when the complete path is quoted.

The default track-scoring manifest deliberately uses only:

```text
P14_numu_database_part_1.db
P14_nue_database_part_1.db
```

This follows the TANGO report's classification sample: GRECO electron- and
muon-neutrino events. NuGen augmentation and tau-neutrino samples belong to the
broader reconstruction corpus and are not automatically added. If the dataset
owner confirms that NuGen was part of the classification training, add
`nugen_nue` and `nugen_numu` to `TRACK_DATABASES` in the preparation job. Do
not add tau events without first defining how their physical topologies should
be labeled.

## 0. Prepare your PACE checkout

Make sure this directory exists in your GraphNeT checkout on PACE, then enter
the repository root—the directory containing `setup.py`. Both Slurm
scripts derive the checkout from `SLURM_SUBMIT_DIR`, so submitting elsewhere
stops early instead of importing the wrong GraphNeT installation.

Check the existing PACE environment and your account association:

```bash
module load anaconda3/2022.05.0.1
conda activate graphnet
python -c "import graphnet, torch; print(graphnet.__file__); print(torch.__version__); print(torch.cuda.is_available())"
sacctmgr show assoc user=$USER format=Account%30,User%20
```

It is normal for `torch.cuda.is_available()` to be false on a login node. The
tracked jobs use account `gts-itaboada3`, QoS `inferno`, one RTX 6000 GPU, and
the `graphnet` Conda environment because those settings appear in this
repository's PACE-tested reconstruction jobs. If the account is not associated
with your user, ask the project owner/PACE rather than using another account.

You can inspect the command-line interfaces without touching the data:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python examples/04_training/track_scoring/01_make_splits.py --help
python examples/04_training/track_scoring/02_train_track_scorer.py --help
python examples/04_training/track_scoring/03_evaluate_track_scorer.py --help
```

## 1. Create the reproducible 10% manifest

Submit from the repository root:

```bash
sbatch examples/04_training/track_scoring/pace_prepare_track_scorer.sbatch
squeue -u $USER
```

The job inventories all six shared paths and requires the two classification
databases to be readable. It then validates their schema, reads only `nu_e`
and `nu_mu` truth, samples 10% inside each class/energy group, and creates:

```text
outputs/track_scoring/splits_10pct_seed42.csv
```

Watch the log using the job ID printed by `sbatch`:

```bash
tail -f track-data-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

If a table or column differs, this job fails before any GPU time is used. The
likely settings are `truth`, `SplitRTCleanedInIcePulses`, `event_no`, `pid`,
`interaction_type`, and `energy`; change the corresponding arguments only
after reading the reported schema error.

The underlying command can also be run directly on a compute node:

```bash
python examples/04_training/track_scoring/01_make_splits.py \
  --databases \
    /storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_numu_database_part_1.db \
    /storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nue_database_part_1.db \
  --output outputs/track_scoring/splits_10pct_seed42.csv \
  --pulsemap SplitRTCleanedInIcePulses \
  --sample-fraction 0.10 \
  --validation-fraction 0.10
```

If these databases are already a special 10% subset, use
`--sample-fraction 1.0`; otherwise you would accidentally use only 1% of the
original data. For the full corpus sizes quoted in the report, a fresh 10%
sample should be approximately:

| Split | Tracks | Cascades | Total |
| --- | ---: | ---: | ---: |
| Training | 270,000 | 182,700 | 452,700 |
| Validation | 30,000 | 20,300 | 50,300 |

Inspect the printed counts. Both classes must be nonzero, the total should be
approximately 10% of the selected sources, and only the two intended paths
should occur in the manifest. Event keys include both database path and event
number, so event numbers need not be globally unique.

## 2. Run the GPU preflight and smoke test

The training job defaults to `RUN_MODE=preflight`. It selects 256 training and
128 validation events, loads one small batch, runs forward/loss checks on the
GPU, and stops without fitting:

```bash
sbatch examples/04_training/track_scoring/pace_train_track_scorer.sbatch
```

After it succeeds, run two epochs on 10,000 training and 2,000 validation
events:

```bash
sbatch --export=ALL,RUN_MODE=smoke \
  examples/04_training/track_scoring/pace_train_track_scorer.sbatch
```

Check `track-scoring-JOBID.out` for the `nvidia-smi` output, seven node
features, both label values, one prediction per event, and finite loss. The
smoke run should additionally create checkpoints and validation predictions.

## 3. Train the full reduced sample

Only after the smoke run succeeds:

```bash
sbatch --export=ALL,RUN_MODE=full,RUN_TAG=full_seed42 \
  examples/04_training/track_scoring/pace_train_track_scorer.sbatch
```

Monitor it without keeping the SSH connection open:

```bash
squeue -u $USER
tail -f track-scoring-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The initial full settings are batch size 32, six data workers, at most 40
epochs, early-stopping patience 6, and learning rate `1e-3`. DynEdgeTITO pads a
batch to its largest graph, so a long event can cause a sudden memory increase.
If the GPU runs out of memory, change only the full-mode batch size to 16 and
rerun before considering architecture changes.

PACE's 20-hour wall time may end before 40 epochs. A `last.ckpt` is written at
completed epochs. Resume with exactly the same settings:

```bash
sbatch --export=ALL,RUN_MODE=full,RUN_TAG=full_seed42,RESUME_FROM=$PWD/outputs/track_scoring/full_seed42/checkpoints/last.ckpt \
  examples/04_training/track_scoring/pace_train_track_scorer.sbatch
```

Each run directory contains:

```text
checkpoints/                  best and last Lightning checkpoints
lightning/                    per-epoch CSV logs
validation_predictions.csv   score plus validation truth
model.pth                     complete, version-sensitive model
model_config.yml              GraphNeT construction config
state_dict.pth                learned parameters
run_arguments.json            command-line settings
selected_manifest.csv         exact events used by this run
```

## 4. Evaluate the track score

Once full training has written predictions:

```bash
python examples/04_training/track_scoring/03_evaluate_track_scorer.py \
  --predictions outputs/track_scoring/full_seed42/validation_predictions.csv \
  --output-dir outputs/track_scoring/full_seed42/evaluation
```

Evaluation produces ROC AUC, average precision, binary cross-entropy, Brier
score, score distributions, calibration, energy-binned AUC, and the report's
working points:

- `track_score > 0.8`: track acceptance and cascade leakage;
- `track_score < 0.4`: cascade acceptance and track leakage.

This reads only a CSV and can normally run in the login environment. If PACE
policy or runtime makes that inappropriate, run it in a small CPU batch job.
Do not train the GNN on a login node.

## Recommended experimental order

1. Verify the manifest counts and manually inspect about ten truth labels.
2. Run `preflight`.
3. Run `smoke` and inspect loss, predictions, and checkpoints.
4. Run one full 10% experiment with seed 42.
5. Evaluate AUC and efficiencies versus energy.
6. Repeat with other seeds or model variants only after the first run is
   understood.

Keeping the manifest, configuration, and seed separates model changes from
data-selection changes and makes each result reproducible.
