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
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-gts-itaboada3}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-l40s}"
export SBATCH_QOS="${SBATCH_QOS:-inferno}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:l40s:1}"
# Temporary mandatory exclusion until PACE repair has been reviewed.
# Custom exclusions extend this block; they cannot disable the known bad host.
BAD_NODE=atl1-1-01-004-31-0
if [[ "${EXCLUDE_NODES+x}" == x && ( -z "$EXCLUDE_NODES" || "$EXCLUDE_NODES" == none ) ]]; then
    echo 'EXCLUDE_NODES cannot be empty or none until the PACE repair is reviewed.' >&2
    exit 2
fi
export EXCLUDE_NODES="${EXCLUDE_NODES-$BAD_NODE}"
if [[ ",$EXCLUDE_NODES," != *",$BAD_NODE,"* ]]; then
    export EXCLUDE_NODES="$BAD_NODE,$EXCLUDE_NODES"
fi
unset SBATCH_EXCLUDE
SBATCH_OPTIONS=(--time="${FULL_TIME:-08:00:00}" --exclude="$EXCLUDE_NODES")
printf 'Checkout: %s\n' "$(pwd -P)"
printf 'Helper: %s\n' "$HELPER_PATH"
REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
printf 'Revision: %s\n' "${REVISION:-unavailable}"
printf 'Excluded nodes: %s\n' "$EXCLUDE_NODES"
if [[ "${2:-}" == --dry-run ]]; then
    printf 'Resume ONLY combined augmentation (both) to at most %s TOTAL epochs, frozen P14 10%% selection.\n' "${FULL_EPOCHS:-80}"
    printf 'Parent: %s/both\n' "$PARENT_ROOT"
    printf 'Early stopping: val_loss, min, patience=%s validation epochs, absolute min_delta=%s.\n' \
        "${EARLY_STOP_PATIENCE:-12}" "${EARLY_STOP_MIN_DELTA:-0.0001}"
    printf 'Allocation: %s / %s / %s / %s; limit %s\n' \
        "$SBATCH_ACCOUNT" "$SBATCH_PARTITION" "$SBATCH_QOS" "$SBATCH_GRES" "${FULL_TIME:-08:00:00}"
    echo 'Requires parent last/best checkpoints and recorded screen ancestry on Phoenix.'
    echo 'Tracks epoch/phase timing, GPU telemetry, early-stop state and validation diagnostics; no test evaluation.'
    echo 'Submission is held until the controller confirms the hold and every excluded host; only then released.'
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
sbatch --test-only "${SBATCH_OPTIONS[@]}" comparison/pace_full.sbatch "$PARENT_ROOT" "$EXCLUDE_NODES"
SUBMISSION="$(sbatch --hold --parsable "${SBATCH_OPTIONS[@]}" comparison/pace_full.sbatch "$PARENT_ROOT" "$EXCLUDE_NODES")"
JOBID="${SUBMISSION%%;*}"
if [[ ! "$JOBID" =~ ^[0-9]+$ ]]; then
    echo "Unexpected submission response (do not resubmit blindly): $SUBMISSION" >&2
    exit 4
fi
printf 'Submitted batch job %s with --hold; verifying controller before release.\n' "$JOBID"
RUN_ROOT="${AUGMENTATION_RUN_ROOT:-$(pwd)/comparison_runs/augmentation}"
RECEIPT="${RUN_ROOT}/submission-${JOBID}.txt"
# A receipt lives beside allocation directories: the batch script must create
# full-JOBID itself and refuses a pre-existing allocation directory.
if mkdir -p "$RUN_ROOT" && printf 'job_id=%s\ncheckout=%s\nhelper=%s\nrevision=%s\nrequested_exclusions=%s\n' \
    "$JOBID" "$(pwd -P)" "$HELPER_PATH" "${REVISION:-unavailable}" "$EXCLUDE_NODES" > "$RECEIPT"; then
    printf 'Submission receipt: %s\n' "$RECEIPT"
else
    echo 'Warning: could not write submission receipt; controller output follows.' >&2
    RECEIPT=""
fi
record() {
    printf '%s\n' "$*"
    if [[ -n "$RECEIPT" ]]; then printf '%s\n' "$*" >> "$RECEIPT" || true; fi
}
stop_held() {
    record "STOP: job $JOBID was NOT released by this helper: $*" >&2
    record "Keep job $JOBID held and inspect it with scontrol; do not resubmit, retry, or cancel blindly." >&2
    exit 5
}
if ! DETAILS="$(scontrol --oneliner show job "$JOBID" 2>&1)"; then
    stop_held "controller inspection failed: $DETAILS"
fi
record "Controller accepted: $DETAILS"
read -r -a JOB_FIELDS <<< "${DETAILS//$'\n'/ }"
ACCEPTED_ID="" ACCEPTED_STATE="" ACCEPTED_PRIORITY="" ACCEPTED_EXCLUSIONS=""
ID_COUNT=0 STATE_COUNT=0 PRIORITY_COUNT=0 EXCLUSION_COUNT=0
for field in "${JOB_FIELDS[@]}"; do
    case "$field" in
        JobId=*) ACCEPTED_ID="${field#*=}"; ID_COUNT=$((ID_COUNT + 1)) ;;
        JobState=*) ACCEPTED_STATE="${field#*=}"; STATE_COUNT=$((STATE_COUNT + 1)) ;;
        Priority=*) ACCEPTED_PRIORITY="${field#*=}"; PRIORITY_COUNT=$((PRIORITY_COUNT + 1)) ;;
        ExcNodeList=*) ACCEPTED_EXCLUSIONS="${field#*=}"; EXCLUSION_COUNT=$((EXCLUSION_COUNT + 1)) ;;
    esac
done
[[ "$ID_COUNT" == 1 && "$ACCEPTED_ID" == "$JOBID" ]] || stop_held 'missing, duplicate or mismatched JobId'
[[ "$STATE_COUNT" == 1 && "$ACCEPTED_STATE" == PENDING ]] || stop_held 'job is not confirmed PENDING'
[[ "$PRIORITY_COUNT" == 1 && "$ACCEPTED_PRIORITY" == 0 ]] || stop_held 'hold is not confirmed by Priority=0'
[[ "$EXCLUSION_COUNT" == 1 && -n "$ACCEPTED_EXCLUSIONS" && "$ACCEPTED_EXCLUSIONS" != '(null)' && "$ACCEPTED_EXCLUSIONS" != none ]] || stop_held 'missing or null ExcNodeList'
if ! REQUESTED_HOSTS="$(scontrol show hostnames "$EXCLUDE_NODES" 2>&1)"; then
    stop_held "requested exclusion expansion failed: $REQUESTED_HOSTS"
fi
if ! ACCEPTED_HOSTS="$(scontrol show hostnames "$ACCEPTED_EXCLUSIONS" 2>&1)"; then
    stop_held "accepted exclusion expansion failed: $ACCEPTED_HOSTS"
fi
[[ -n "$REQUESTED_HOSTS" && -n "$ACCEPTED_HOSTS" ]] || stop_held 'exclusion expansion returned no hosts'
while IFS= read -r host; do
    [[ -n "$host" && "$host" != *[[:space:]]* ]] || stop_held 'invalid requested host expansion'
    found=false
    while IFS= read -r accepted_host; do
        [[ -n "$accepted_host" && "$accepted_host" != *[[:space:]]* ]] || stop_held 'invalid accepted host expansion'
        if [[ "$host" == "$accepted_host" ]]; then found=true; fi
    done <<< "$ACCEPTED_HOSTS"
    [[ "$found" == true ]] || stop_held "controller exclusion is missing requested host $host"
done <<< "$REQUESTED_HOSTS"
record 'Verified: matching JobId, PENDING, Priority=0, and all requested excluded hosts.'
if ! RELEASE_OUTPUT="$(scontrol release "$JOBID" 2>&1)"; then
    record "Release command failed for job $JOBID: $RELEASE_OUTPUT" >&2
    record "Inspect job $JOBID and its hold before any action; do not resubmit." >&2
    exit 6
fi
record "Released verified job $JOBID."
