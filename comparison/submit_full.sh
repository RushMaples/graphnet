#!/bin/bash
# Continue the selected combined model; one allocation, maximum 80 total epochs.
set -euo pipefail
HELPER_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
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
# Preserve the two supported account overrides before clearing sbatch inputs.
ACCOUNT="${SBATCH_ACCOUNT:-gts-itaboada3}"
QOS="${SBATCH_QOS:-inferno}"
# sbatch reads these inputs before submission. Stale shell settings must not add
# holds, host lists, constraints, dependencies or conflicting resource requests.
for variable in "${!SBATCH_@}"; do unset "$variable"; done
# Documented sbatch aliases (including older Slurm memory-binding aliases).
# Preserve SLURM_CONF, diagnostics and allocation/output metadata.
unset SLURM_CLUSTERS SLURM_HINT SLURM_MEM_BIND EXCLUDE_NODES
SBATCH_OPTIONS=(--parsable --account="$ACCOUNT" --partition=gpu-l40s --qos="$QOS"
    --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:l40s:1 --mem=64G
    --time="${FULL_TIME:-08:00:00}")
COMMAND=(sbatch "${SBATCH_OPTIONS[@]}" comparison/pace_full.sbatch "$PARENT_ROOT")
printf 'Checkout: %s\n' "$(pwd -P)"
printf 'Helper: %s\n' "$HELPER_PATH"
REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
printf 'Revision: %s\n' "${REVISION:-unavailable}"
printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
if [[ "${2:-}" == --dry-run ]]; then
    printf 'Resume ONLY combined augmentation (both) to at most %s TOTAL epochs, frozen P14 10%% selection.\n' "${FULL_EPOCHS:-80}"
    printf 'Parent: %s/both\n' "$PARENT_ROOT"
    printf 'Early stopping: val_loss, min, patience=%s validation epochs, absolute min_delta=%s.\n' \
        "${EARLY_STOP_PATIENCE:-12}" "${EARLY_STOP_MIN_DELTA:-0.0001}"
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
SUBMISSION="$("${COMMAND[@]}")"
JOBID="${SUBMISSION%%;*}"
if [[ ! "$SUBMISSION" =~ ^[0-9]+(\;[^[:space:]\;]+)?$ ]]; then
    echo "Unexpected submission response (do not resubmit blindly): $SUBMISSION" >&2
    exit 4
fi
printf 'Submitted batch job %s. No further submission will be attempted.\n' "$JOBID"
RUN_ROOT="${AUGMENTATION_RUN_ROOT:-$(pwd)/comparison_runs/augmentation}"
RECEIPT="${RUN_ROOT}/submission-${JOBID}.txt"
# The batch script creates full-JOBID and refuses an existing allocation directory.
if mkdir -p "$RUN_ROOT" && {
    printf 'job_id=%s\ncheckout=%s\nhelper=%s\nrevision=%s\nhost_policy=any_l40s\nrequested_exclusions=none\n' \
        "$JOBID" "$(pwd -P)" "$HELPER_PATH" "${REVISION:-unavailable}"
    printf 'command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
} > "$RECEIPT"; then
    printf 'Submission receipt: %s\n' "$RECEIPT"
else
    echo 'Warning: could not write submission receipt; job was already submitted.' >&2
    RECEIPT=""
fi
record() {
    printf '%s\n' "$*"
    if [[ -n "$RECEIPT" ]]; then printf '%s\n' "$*" >> "$RECEIPT" || true; fi
}
# Read-only evidence: controller/site policy may add restrictions. Record them
# exactly as returned; never hold, release or rewrite the accepted job.
if DETAILS="$(scontrol --oneliner show job "$JOBID" 2>&1)"; then
    record "Controller snapshot: $DETAILS"
    record 'Host policy: any L40S; any controller-added restrictions are recorded above and were not requested by this helper.'
else
    record "Warning: controller inspection failed for submitted job $JOBID: $DETAILS" >&2
    record "Job $JOBID was submitted successfully; inspect that job before considering another submission." >&2
fi
