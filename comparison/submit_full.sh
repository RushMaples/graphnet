#!/bin/bash
# Continue the selected combined model; one allocation, maximum 80 total epochs.
set -euo pipefail
cd "${ICEMIX_CHECKOUT:-$(pwd)}"
test -f comparison/pace_full.sbatch
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash comparison/submit_full.sh CONTINUATION_ROOT [--dry-run]" >&2
    exit 2
fi
if [[ -n "${2:-}" && "$2" != --dry-run ]]; then
    echo "Unknown argument: $2" >&2
    exit 2
fi
PARENT_ROOT="$(cd "$1" && pwd)"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-gts-itaboada3}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-l40s}"
export SBATCH_QOS="${SBATCH_QOS:-inferno}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:l40s:1}"
if [[ "${2:-}" == --dry-run ]]; then
    printf 'Resume ONLY combined augmentation (both) to at most %s TOTAL epochs, frozen P14 10%% selection.\n' "${FULL_EPOCHS:-80}"
    printf 'Parent: %s/both\n' "$PARENT_ROOT"
    printf 'Early stopping: val_loss, min, patience=%s validation epochs, absolute min_delta=%s.\n' \
        "${EARLY_STOP_PATIENCE:-12}" "${EARLY_STOP_MIN_DELTA:-0.0001}"
    printf 'Allocation: %s / %s / %s / %s; limit %s\n' \
        "$SBATCH_ACCOUNT" "$SBATCH_PARTITION" "$SBATCH_QOS" "$SBATCH_GRES" "${FULL_TIME:-08:00:00}"
    echo 'Requires parent last/best checkpoints and recorded screen ancestry on Phoenix.'
    echo 'Tracks epoch/phase timing, GPU telemetry, early-stop state and validation diagnostics; no test evaluation.'
    echo 'Dry run: no job submitted.'
    exit 0
fi
for name in complete.json selected_checkpoint.json checkpoint_source.json timing.csv; do
    test -f "${PARENT_ROOT}/both/${name}"
done
if [[ ! -s "${PARENT_ROOT}/both/checkpoints/last.ckpt" ]]; then
    echo "Missing resumable checkpoint: ${PARENT_ROOT}/both/checkpoints/last.ckpt" >&2
    exit 3
fi
sbatch --test-only --time="${FULL_TIME:-08:00:00}" comparison/pace_full.sbatch "$PARENT_ROOT"
sbatch --time="${FULL_TIME:-08:00:00}" comparison/pace_full.sbatch "$PARENT_ROOT"
