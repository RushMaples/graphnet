#!/bin/bash
# Submit one baseline/both continuation allocation. No automatic 80-epoch run.
set -euo pipefail
cd "${ICEMIX_CHECKOUT:-$(pwd)}"
test -f comparison/pace_continue.sbatch
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash comparison/submit_continuation.sh SCREEN_ROOT [--dry-run]" >&2; exit 2
fi
if [[ -n "${2:-}" && "$2" != --dry-run ]]; then
    echo "Unknown argument: $2" >&2; exit 2
fi
PARENT_ROOT="$(cd "$1" && pwd)"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-gts-itaboada3}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-l40s}"
export SBATCH_QOS="${SBATCH_QOS:-inferno}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:l40s:1}"
if [[ "${2:-}" == --dry-run ]]; then
    printf 'Continue baseline and both to %s TOTAL epochs on frozen P14 10%% selection.\n' "${CONTINUE_EPOCHS:-30}"
    printf 'Parent: %s\n' "$PARENT_ROOT"
    printf 'Allocation: %s / %s / %s / %s; limit %s\n' "$SBATCH_ACCOUNT" "$SBATCH_PARTITION" "$SBATCH_QOS" "$SBATCH_GRES" "${CONTINUE_TIME:-08:00:00}"
    echo 'Requires parent last.ckpt and previous best checkpoints on Phoenix; submits no full run.'
    exit 0
fi
for arm in baseline both; do
    if [[ ! -s "${PARENT_ROOT}/${arm}/checkpoints/last.ckpt" ]]; then
        echo "Missing resumable checkpoint: ${PARENT_ROOT}/${arm}/checkpoints/last.ckpt" >&2; exit 3
    fi
done
sbatch --test-only --time="${CONTINUE_TIME:-08:00:00}" comparison/pace_continue.sbatch "$PARENT_ROOT"
sbatch --time="${CONTINUE_TIME:-08:00:00}" comparison/pace_continue.sbatch "$PARENT_ROOT"
