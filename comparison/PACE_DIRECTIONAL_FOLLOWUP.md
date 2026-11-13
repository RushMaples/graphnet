# Five-run directional follow-up

Agreed scope, 10 November 2026: one parallel Slurm array with five members,
a 50-total-epoch cap, patience 12, and absolute minimum improvement 0.0001.
The launcher also submits a CPU comparison job with an `afterok` dependency.

| Array index | Name | Initial state | Vertex supervision | Explicit parent count |
|---|---|---|---|---|
| 0 | joint_vmf | screen-15407679 vMF last checkpoint, ten epochs completed | Yes | No |
| 1 | joint_king_no_count | Fresh, seed 42 | Yes | No |
| 2 | joint_flow | screen-15407679 flow last checkpoint, ten epochs completed | Yes | No |
| 3 | direction_vmf | Fresh, seed 42 | No | No |
| 4 | direction_flow | Fresh, seed 42 | No | No |

No new training seeds, sampling-convergence studies, deeper flows, test
evaluation, or slide generation are included. The flow remains four layers
with the existing 128-draw estimator. The old pulse-aware King is reused only
as the ten-epoch count-conditioning control.

## Upload and launch

Upload directional-followup-kit.zip to your PACE home directory. From your
existing GraphNeT checkout, run these commands, one line at a time:

```bash
cd /storage/project/r-itaboada3-0/rmaples3/software/graphnet
unzip -o ~/directional-followup-kit.zip
bash comparison/start_directional_followup.sh --test-only
bash comparison/start_directional_followup.sh
```

The first launcher command verifies the original prediction/receipt hashes,
the two full resume checkpoints, frozen selections, source compatibility,
existing environment, and BOTH scheduler resource requests. No jobs are
submitted by `--test-only`. Run the final command once after that succeeds.
The final command repeats its checks, freezes the source code, registers the
array on hold, registers the dependent comparison job, then releases the array.
If comparison submission fails, it cancels only the newly held array.

No environment installation or package upgrade is performed. The profile uses
the existing graphnet-directional-v2 environment, Anaconda module
anaconda3/2022.05.0.1, account gts-itaboada3, QOS inferno, and gpu-l40s partition.
Each GPU run requests one L40S, four CPUs, 64G memory, and a 16-hour wall limit
to accommodate training and the retained sampling exports. Early stopping
can finish much sooner. The CPU comparison uses cpu-small, two CPUs, 16G,
and 30 minutes. Scheduler acceptance is checked before real submission.

`start_directional_followup.sh` records the paths already used in this project.
Existing environment variables override them. An incorrect inherited ENV_PREFIX
or selection path will fail the checks rather than silently using a different
environment. To change one known path explicitly, use for example:

```bash
ENV_PREFIX=/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet-directional-v2 bash comparison/start_directional_followup.sh --test-only
```

Other explicit overrides: FOLLOWUP_SCREEN_ROOT, FOLLOWUP_HISTORICAL_ROOT,
FOLLOWUP_RUN_ROOT, FOLLOWUP_TIME, and FOLLOWUP_SUMMARY_PARTITION. Changing a
resource request does not alter the declared scientific run matrix.

## Required existing artifacts

The original screen is expected under
`comparison_runs/directional/screen-15407679/{vmf,king,flow}/model/`.
All three arms need their original settings, completion/provenance/database
receipts, initial state/input probe, metrics log, selected-checkpoint receipt,
and validation predictions/manifest. vMF and flow additionally need their
selected checkpoint file and full `checkpoints/last.ckpt`.

The historical vMF validation CSV, completion receipt and selected-checkpoint
receipt are expected under
`comparison_runs/augmentation/full-14534898/both/`. Its model weights are not
needed for the historical table. Any missing or changed artifact stops the
launcher before GPU submission. Original screen outputs are never overwritten.

## The three scientific comparisons

1. **Count conditioning:** compare existing pulse-aware King at a ten-epoch
   budget to the saved ten-epoch count-free King result. Its architecture,
   initial parameters and loss are unchanged; only log(parent count / 32)
   is replaced by a constant zero. True counts stay in data and exports.
2. **Joint density heads:** compare the three count-free joint models under
   the common 50-epoch cap and stopping rule. Checkpoints are selected by
   joint validation loss. Display actual trained and selected epochs.
3. **Vertex supervision:** compare joint and direction-only vMF, and joint
   and direction-only flow, selecting both by exact directional validation
   NLL. Fixed ten-epoch comparisons are retained, along with final results.
   Direction-only output tables omit vertex predictions and errors.

The native vMF training approximation is preserved in both objectives, so
removing vertex supervision is the intended loss change. Evaluation NLL is
always exact per steradian. The plateau scheduler retains the original
val_loss monitor, which contains vertex loss only for the joint models.
Final task comparisons therefore also reflect their stopping/scheduler
trajectories; the fixed ten-epoch contrast is the simpler diagnostic.

The old vMF epoch-31 model appears as historical performance context, labelled
with its actual 43-epoch training history. It is not substituted for a matched
control. Source, input identities, settings, environment and completion records
are audited before comparisons are published.

## What is saved

Each task writes to a new timestamped folder beneath
`comparison_runs/directional_followup/`. The launcher prints the exact path,
array ID, comparison-job ID and monitoring command. It freezes the submission's
source code in that folder; later working-checkout edits do not mix queued runs.

Each model retains full last checkpoints, best joint/direction checkpoints as
applicable, training logs, timing, preflight results and prediction manifests.
Snapshots at 10 and 30 completed epochs are retained when reached. Thirty-epoch
snapshots do not trigger extra sampling exports. Ten-epoch parent exports are
reused when their selected weights are identical. Identical final selections
also share an export after their model-state hashes are checked.

The last checkpoint is sealed after the final scheduler update. On legacy
resume, a pending final validation scheduler check is replayed exactly once
when checkpoint progress proves it was not yet applied. Optimizer, scheduler,
Lightning loop and AMP states are restored. The old checkpoints did not retain
all RNG states, so exact uninterrupted stochastic-trajectory equivalence is not
claimed. Lightning may warn that the legacy ModelCheckpoint callback key is
not present: this workflow deliberately carries parent-best checkpoints into
new callback paths, rather than letting a callback delete an original file.

The dependent comparison job produces:

- `summary/results.html`: three compact result tables, readable in a browser.
- `summary/headline.csv`: overall results, including ten-epoch controls.
- `summary/metrics.csv`: the same questions broken down into tracks/cascades.
- `summary/three_questions.png`: one compact figure.
- `summary/verification.json`: matching/protocol checks and limitations.
- `directional-followup-results.tar.gz`: compact report, logs and receipts
  suitable for upload. Large prediction tables and checkpoints stay on PACE.

The report does not choose a model automatically or fit calibration on these
results. One seed and validation-selected results remain provisional. Failure
of any array member prevents the comparison job from publishing partial results.

## Checks included in the kit

Local tests exercise constant-count invariance, common initialization, removal
of vertex gradients, strict stopping arithmetic, actual Lightning 2.4 CPU
checkpoint continuation, retention of parent-best weights, scheduler completion,
post-fit checkpoint resume, and a complete five-run comparison using fixtures.
The GPU job additionally executes an eight-batch real CUDA/AMP smoke fit and
checkpoint restore before production training. Continued models check restored
selected weights against archived validation predictions before fitting.

Windows cannot execute this environment's MSYS Bash process, so local shell
verification uses a Bash syntax parser. The PACE launcher additionally executes
`bash -n` on every submitted shell script. Full GraphNeT CUDA training and Slurm
allocation checks execute on PACE; they are not claimed as local test results.
