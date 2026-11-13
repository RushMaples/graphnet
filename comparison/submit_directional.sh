#!/bin/bash
# Three array tasks; each trains one arm for ten full epochs on any L40S.
set -euo pipefail
if [[ $# -gt 1 || ( -n "${1:-}" && "$1" != --dry-run && "$1" != --test-only ) ]]; then
    echo 'Usage: bash comparison/submit_directional.sh [--dry-run|--test-only]' >&2; exit 2
fi
cd "${ICEMIX_CHECKOUT:-$(pwd)}"
export ICEMIX_CHECKOUT="$(pwd -P)"
test -f comparison/pace_directional.sbatch
ACCOUNT="${DIRECTIONAL_ACCOUNT:-gts-itaboada3}"
QOS="${DIRECTIONAL_QOS:-inferno}"
WALL_TIME="${DIRECTIONAL_TIME:-12:00:00}"
for variable in "${!SBATCH_@}"; do unset "$variable"; done
unset SLURM_CLUSTERS SLURM_HINT SLURM_MEM_BIND EXCLUDE_NODES
OPTIONS=(--parsable --export=ALL --account="$ACCOUNT" --qos="$QOS" --partition=gpu-l40s
    --array=0-2 --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:l40s:1
    --mem=64G --time="$WALL_TIME")
if [[ "${1:-}" == --test-only ]]; then OPTIONS+=(--test-only); fi
COMMAND=(sbatch "${OPTIONS[@]}" comparison/pace_directional.sbatch)
printf 'Checkout: %s\nCommand:' "$ICEMIX_CHECKOUT"; printf ' %q' "${COMMAND[@]}"; printf '\n'
if [[ "${1:-}" == --dry-run ]]; then echo 'Dry run: no jobs submitted.'; exit 0; fi
if [[ "${1:-}" == --test-only ]]; then
    "${COMMAND[@]}"
    echo 'Scheduler validation passed; no jobs submitted. This does not run the GPU preflight.'
    exit 0
fi
export ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet-directional}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
CARLOS_ROOT="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
SELECTION_ROOT="${PILOT_SELECTION_ROOT:-${CARLOS_ROOT}/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "Missing interpreter: ${ENV_PREFIX}/bin/python; prepare the cloned environment first." >&2; exit 3
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ICEMIX_CHECKOUT}/src:${ICEMIX_CHECKOUT}/comparison"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/icemix-directional-${USER}/matplotlib"
# Read/import checks only on the login node; no model training or DB scans.
"${ENV_PREFIX}/bin/python" comparison/directional_run.py check-inputs \
    --selection-root "$SELECTION_ROOT" --data-root "$SOURCE_DATA_ROOT"
"${ENV_PREFIX}/bin/python" comparison/directional_run.py check-environment
SUBMISSION="$("${COMMAND[@]}")"
[[ "$SUBMISSION" =~ ^[0-9]+(\;[^[:space:]\;]+)?$ ]] || {
    echo "Unexpected response, inspect before resubmitting: $SUBMISSION" >&2; exit 4
}
JOBID="${SUBMISSION%%;*}"
printf 'Submitted array %s (0=vmf, 1=king, 2=flow).\n' "$JOBID"
RUN_ROOT="${DIRECTIONAL_RUN_ROOT:-${ICEMIX_CHECKOUT}/comparison_runs/directional}"
mkdir -p "$RUN_ROOT"
if ! scontrol show job --oneliner "$JOBID" > "${RUN_ROOT}/submission-${JOBID}.txt" 2>&1; then
    echo "Controller snapshot unavailable; array $JOBID was already submitted. Inspect before resubmitting." >&2
fi
