#!/bin/bash
# Queue the screen and a dependent full run; no separate preflight allocation.
set -euo pipefail
cd "${ICEMIX_CHECKOUT:-$(pwd)}"
test -f comparison/pace_augment.sbatch
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-gts-itaboada3}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-l40s}"
export SBATCH_QOS="${SBATCH_QOS:-inferno}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:l40s:1}"
RUN_ROOT="${AUGMENTATION_RUN_ROOT:-${PWD}/comparison_runs/augmentation}"
if [[ "${1:-}" == --dry-run ]]; then
    printf 'Screen: 4 arms x %s epochs; full: %s epochs; frozen P14 10%% selections\n' "${SCREEN_EPOCHS:-10}" "${FULL_EPOCHS:-80}"
    printf 'Allocation: %s / %s / %s / %s\n' "$SBATCH_ACCOUNT" "$SBATCH_PARTITION" "$SBATCH_QOS" "$SBATCH_GRES"
    printf 'Run root: %s\n' "$RUN_ROOT"
    exit 0
fi
sbatch --test-only --time="${SCREEN_TIME:-12:00:00}" comparison/pace_augment.sbatch screen
screen_job=$(sbatch --parsable --time="${SCREEN_TIME:-12:00:00}" comparison/pace_augment.sbatch screen)
screen_job="${screen_job%%;*}"
if [[ ! "$screen_job" =~ ^[0-9]+$ ]]; then
    echo "Could not parse screen job ID: $screen_job" >&2; exit 3
fi
echo "Screen submitted: $screen_job"
full_job=$(sbatch --parsable --dependency="afterok:${screen_job}" --kill-on-invalid-dep=yes \
    --time="${FULL_TIME:-12:00:00}" comparison/pace_augment.sbatch full "${RUN_ROOT}/screen-${screen_job}")
echo "Full run submitted (after successful screen): ${full_job%%;*}"
echo "Screen results: ${RUN_ROOT}/screen-${screen_job}/summary.csv"
