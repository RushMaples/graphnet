#!/bin/bash
# Queue only the screen; review results before submitting full training.
set -euo pipefail
cd "${ICEMIX_CHECKOUT:-$(pwd)}"
test -f comparison/pace_augment.sbatch
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-gts-itaboada3}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-l40s}"
export SBATCH_QOS="${SBATCH_QOS:-inferno}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:l40s:1}"
RUN_ROOT="${AUGMENTATION_RUN_ROOT:-${PWD}/comparison_runs/augmentation}"
case "${1:-}" in
    ""|--dry-run) ;;
    -h|--help) echo "Usage: bash comparison/submit_augmentation.sh [--dry-run] (screen only)"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if [[ "${1:-}" == --dry-run ]]; then
    printf 'Screen only: 4 arms x %s epochs; frozen P14 10%% selections\n' "${SCREEN_EPOCHS:-10}"
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
echo "Screen results: ${RUN_ROOT}/screen-${screen_job}/summary.csv"
echo "Review the results before submitting the 80-epoch full run."
