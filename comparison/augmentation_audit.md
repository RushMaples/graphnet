# Carlos augmentation evidence and controlled port

Audit date: 2026-10-22. Carlos source is pinned to `1d83975e21b5b1c7f02efb797d5fc39f5f5735c6`, the revision recorded in the archived comparison provenance. This audit distinguishes source/configuration support from evidence that a run actually used it.

## Configured variants and exact rates

| Variant | Rotation | Conditional pulse removal | Event selection probability | Evidence |
|---|---:|---:|---:|---|
| Carlos standard | off | 0 | 1.0 default | [standard.yaml](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/conf/data/standard.yaml#L20), [baseline.yaml](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/conf/attention/baseline.yaml#L20) |
| Carlos drop | off | `token_drop=0.05` | `drop_chance=0.05` | [drop.yaml](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/conf/data/drop.yaml#L5) |
| Carlos rotation | on | 0 | 1.0 default | [augmented_rotation.yaml](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/conf/data/augmented_rotation.yaml#L4) |
| Carlos combined | on | `token_drop=0.05` | `drop_chance=0.05` | [drop_aug_rot.yaml](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/conf/data/drop_aug_rot.yaml#L1) |

The pinned [drop launcher](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/run_drop.sbatch), [rotation launcher](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/run_augmented.sbatch), and [combined launcher](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/run_drop_aug_rot.sbatch) select these configurations. Their presence establishes configured workflows, not successful executions or measured benefits. No local resolved nonzero augmentation run was found in the audited pilot/comparison bundles.

## Methods and faithful semantics

**Pulse thinning:** [Carlos transformer.py:270–315](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/src/models/transformer.py#L270) first selects events with probability `drop_chance`, then independently removes each pulse in a selected event with probability `token_drop`. If an event would become empty, its first pulse is restored. Rows are removed before sequence padding, Fourier encoding, and relative-position construction. Event labels and event count are preserved. It is active during training; the explicit `force_token_drop` hook also permits separate robustness evaluation. Ordinary validation/test inference does not thin pulses.

At the configured `0.05/0.05`, 5% of events are eligible for thinning and each eligible pulse has a 5% removal probability. Expected removal across all pulses is approximately **0.25%**, reduced slightly by the nonempty-event safeguard. This is not 5% removal across the whole dataset. The standalone `reproduce_token_drops.py` synthetic example uses `0.5`; that is a diagnostic setting, not the configured training rate.

**Online rotation:** [Carlos utils.py:177–265](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/src/utils.py#L177) draws one independent angle uniformly in `[0,2*pi)` for every training event on each visit. It rotates pulse x/y, vertex x/y, direction x/y, and azimuth consistently. z, zenith, charge, and time are unchanged. Carlos rotates about x=y=0. Its standard `rotation_seed: null` uses private entropy, so global run seeding alone does not reproduce rotation angles.

Ajai's current [callback](../src/graphnet/training/callbacks.py) already implements optional train-only rotation and handles list batches. The [training example](../examples/04_training/09_train_icemix_joint_reconstruction.py) chooses the DeepCore string-36 pivot `(46.3,-34.9)` metres for vertex targets and that pivot divided by the 500 m input coordinate scale for node features. Direction vectors rotate about zero. This is a consistent physical pivot, but differs from Carlos's origin. A controlled port should preserve the current coordinate treatment and record that difference explicitly. Rotation is an augmentation hypothesis: the discrete detector geometry and directional detector response do not have exact continuous azimuthal symmetry, so validation must establish its effect.

## Evidence of what the archived runs actually enabled

All paths below are relative to the workspace root, three directories above this file.

- Sep30 frozen 10% pilot: `pilot_review_2026-09-30/bundle/resolved-config.yaml:66,89–90` records `token_drop: 0.0`, `augment_rotation: false`, `rotation_seed: null`; `bundle/predictions/model_config.yml:5,10` records `drop_chance: 1.0`, `token_drop: 0.0`.
- Oct13 Carlos comparison job `13992486`: `output/icemix_comparison_2026-10-13/bundle/comparison_runs/job-13992486/carlos/model_config.yml:5,10` records `drop_chance: 1.0`, `token_drop: 0.0`; its `resolved_settings.json:18` records `rotation_augmentation: false`.
- Oct13 Ajai comparison job `13992488`: the corresponding `ajai/resolved_settings.json:18` also records `rotation_augmentation: false`. Optional rotation support already existed; the previous matched comparison disabled it.

These are actual archived run settings. They establish an augmentation-off baseline, not evidence that Carlos's nonzero variants improved performance.

## Frozen-split experiment boundary

Use the same frozen parent-event train/validation/test selections in both arms. Apply online augmentation only to the training arm: existing Ajai rotation plus Carlos pulse-thinning semantics at `token_drop=0.05`, `drop_chance=0.05`. Use explicit seeds and persist resolved settings. Keep validation/test inputs unaugmented and use validation to select any subsequent full-training configuration; a combined on/off comparison measures the package and cannot attribute effects separately to rotation and pulse thinning.

Exclude the older precomputed augmented-database path. [Carlos train.py:140–208](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/train.py#L140) detects `augmented` in database paths, infers replica count from `MAX(event_no)//10**10`, and expands each training ID to `id + k*10**10`. That path assumes a particular complete replica/ID layout and changes training exposures per epoch. The [old config](https://github.com/carlosm-silva/graphnet/blob/1d83975e21b5b1c7f02efb797d5fc39f5f5735c6/ice_mix/conf/data/augmented_rotation_OLD.yaml) points to precomputed augmented databases. None of this expansion is needed for the frozen 10% online comparison. If such databases are ever used, every replica must stay in its parent's partition and replica existence must be verified explicitly.
