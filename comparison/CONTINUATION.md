# Review at 30 epochs before committing to 80

Continue **baseline** and **both** from the completed ten-epoch screen to
**30 total epochs**. Each arm adds 20 epochs. All training stays on the same
frozen **10%** selection: 578,121 training events. No additional data are used.

The screen's small pooled differences did not establish an augmentation win.
Its validation predictions show direction-dependent tradeoffs and opposite
track/cascade coverage errors. Training loss was still falling and the
learning-rate scheduler had never reduced the rate. A checkpointed 30-epoch
comparison is a bounded next experiment; it does not guarantee improvement.

## Submission

On Phoenix, from the RushMaples/graphnet root:

```bash
git pull --ff-only origin main
bash comparison/submit_continuation.sh comparison_runs/augmentation/screen-14290300 --dry-run
bash comparison/submit_continuation.sh comparison_runs/augmentation/screen-14290300
```

One allocation runs the two arms sequentially, stages databases once, and
performs checks inside the allocation. Defaults: one L40S, 4 CPUs, 64 GB,
`gts-itaboada3`, `gpu-l40s`, `inferno`, and an **8-hour wall-time limit**.
The measured typical epoch cost suggests **4.3–4.4 additional GPU-hours** for
40 added epochs, plus startup/audits/export. Allow roughly **5–6 hours** for
planning; these are usage estimates, not prices or a guarantee of duration.
There is no automatic 80-epoch submission.

`CONTINUE_EPOCHS` overrides the total horizon; `CONTINUE_TIME` overrides the
Slurm limit. Existing environment/path/account overrides match the screen
launcher. Keep hardware, environment, source, and training settings fixed.

## What is restored

The parent `checkpoints/last.ckpt` must still exist on Phoenix for each arm.
It was deliberately excluded from the downloaded analysis archive. The
driver restores model weights, optimizer, scheduler, epoch/global-step state,
and mixed-precision scaler through Lightning's checkpoint resume path.
It does not warm-start weights with a fresh optimizer or restart at epoch 0.

The previous validation-best checkpoint must also remain readable. At the
end, the best validation checkpoint is chosen across **the parent run and
the continuation**, so the old best is retained if the added epochs do not
improve it. Training always continues from `last.ckpt`, not the selected best.

The driver verifies every original source/harness file against the recorded
hashes, while allowing the new continuation files. It verifies frozen split
checksums, source data hashes, runtime and GPU identity, and held-out input
fingerprints. A mismatch fails rather than silently changing the experiment.
Do not edit old source files to bypass these checks.

Augmentation masks/angles are seeded by absolute epoch and batch. Loader
workers are newly spawned during resume; this is not a claim of bitwise
equivalence with uninterrupted training. Both arms use the same resume
procedure. Validation is unaugmented, and no test loader is constructed.

## Results

Logs: `icemix-augmentation-continue-JOBID.out`.
Run root: `comparison_runs/augmentation/continue-JOBID/`, with `baseline/`
and `both/`, full continuation provenance, parent checkpoint hashes,
best/last checkpoints, epoch logs/timing, and validation predictions/metrics.
`summary.csv` compares the two arms. Send that summary and the curves for
review before extending either arm to 80 epochs.

`validation_diagnostics.csv` adds topology/energy subdivisions, error tails,
coverage and predicted region size. Its `fp32_prediction_joint_loss` is a CPU
reconstruction from the unaugmented FP32 inference export. Checkpoint selection
continues to use the original logged mixed-precision `val_loss`; these two
numbers can differ slightly and are labeled separately.

```bash
column -s, -t < comparison_runs/augmentation/continue-JOBID/summary.csv
tar -czf augmentation-continue-JOBID.tar.gz \
  --exclude='*.ckpt' --exclude='length_cache' \
  comparison_runs/augmentation/continue-JOBID \
  icemix-augmentation-continue-JOBID.out
```

## Local validation limits

The CPU continuation tests cover frozen settings, exposure, ancestry, full-state
checkpoint requirements, parent-best retention, source/export tampering, paired
inputs and diagnostic loss units. The submission script has been checked in
dry-run mode. Actual baseline/both screen metadata and prediction hashes pass
the continuation checks. A complete GPU resume cannot be tested from the
downloaded bundle because it excludes the full checkpoints; the existing
Phoenix environment performs the real-data forward/backward checks inside the
training allocation before fitting.
