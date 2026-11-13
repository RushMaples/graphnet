#!/bin/bash
# One submission command: five one-GPU array members, then CPU comparison.
set -euo pipefail
if [[ $# -gt 1 || ( -n "${1:-}" && "$1" != --dry-run && "$1" != --test-only ) ]]; then
    echo 'Usage: bash comparison/start_directional_followup.sh [--dry-run|--test-only]' >&2; exit 2
fi
cd "${ICEMIX_CHECKOUT:?Use start_directional_followup.sh}"
for script in comparison/start_directional_followup.sh comparison/submit_directional_followup.sh comparison/pace_directional_followup.sbatch comparison/pace_directional_followup_summary.sbatch; do bash -n "$script"; done
export ICEMIX_CHECKOUT="$(pwd -P)"
: "${ENV_PREFIX:?}" "${PILOT_SELECTION_ROOT:?}" "${SOURCE_DATA_ROOT:?}" "${PACE_CONDA_MODULE:?}"
: "${FOLLOWUP_SCREEN_ROOT:?}" "${FOLLOWUP_HISTORICAL_ROOT:?}"
ACCOUNT="${DIRECTIONAL_ACCOUNT:?}"; QOS="${DIRECTIONAL_QOS:?}"
export FOLLOWUP_RUN_ROOT="${FOLLOWUP_RUN_ROOT:-${ICEMIX_CHECKOUT}/comparison_runs/directional_followup}"
for variable in "${!SBATCH_@}"; do unset "$variable"; done
unset SLURM_CLUSTERS SLURM_HINT SLURM_MEM_BIND EXCLUDE_NODES
GPU_OPTIONS=(--parsable --export=ALL --account="$ACCOUNT" --qos="$QOS" --partition=gpu-l40s
    --array=0-4 --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:l40s:1 --mem=64G
    --time="${FOLLOWUP_TIME:-16:00:00}")
CPU_OPTIONS=(--parsable --export=ALL --account="$ACCOUNT" --qos="$QOS"
    --partition="${FOLLOWUP_SUMMARY_PARTITION:-cpu-small}" --nodes=1 --ntasks=1
    --cpus-per-task=2 --mem=16G --time=00:30:00)
printf 'Five runs: joint_vmf, joint_king_no_count, joint_flow, direction_vmf, direction_flow\n'
printf 'Protocol: seed 42; cap 50 total epochs; patience 12; min delta 0.0001; 128 flow draws.\n'
printf 'Original screen: %s\nHistorical vMF: %s\n' "$FOLLOWUP_SCREEN_ROOT" "$FOLLOWUP_HISTORICAL_ROOT"
if [[ "${1:-}" == --dry-run ]]; then
    printf 'GPU command:'; printf ' %q' sbatch "${GPU_OPTIONS[@]}" comparison/pace_directional_followup.sbatch; printf '\n'
    printf 'Then CPU command:'; printf ' %q' sbatch "${CPU_OPTIONS[@]}" '--dependency=afterok:ARRAY_ID' comparison/pace_directional_followup_summary.sbatch; printf '\n'
    echo 'Dry run: no files or jobs created. Actual submission checks all parent artifacts first.'
    exit 0
fi
test -x "${ENV_PREFIX}/bin/python" || { echo "Missing interpreter: ${ENV_PREFIX}/bin/python" >&2; exit 3; }
module load "$PACE_CONDA_MODULE"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export PYTHONPATH="${ICEMIX_CHECKOUT}/src:${ICEMIX_CHECKOUT}/comparison"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/icemix-followup-${USER}/pycache"
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/icemix-followup-${USER}/pip"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/icemix-followup-${USER}/uv"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/icemix-followup-${USER}/matplotlib"
CHECK_ARGS=(--source-root "$ICEMIX_CHECKOUT" --screen-root "$FOLLOWUP_SCREEN_ROOT"
    --historical-root "$FOLLOWUP_HISTORICAL_ROOT" --selection-root "$PILOT_SELECTION_ROOT" --data-root "$SOURCE_DATA_ROOT")
"${ENV_PREFIX}/bin/python" comparison/directional_followup.py check "${CHECK_ARGS[@]}"
"${ENV_PREFIX}/bin/python" comparison/directional_run.py check-environment
# Validate BOTH resource requests before submitting any real work.
sbatch "${GPU_OPTIONS[@]}" --test-only comparison/pace_directional_followup.sbatch
sbatch "${CPU_OPTIONS[@]}" --test-only comparison/pace_directional_followup_summary.sbatch
if [[ "${1:-}" == --test-only ]]; then
    echo 'Input/checkpoint/environment checks and scheduler validation passed; no jobs submitted.'
    exit 0
fi
mkdir -p "$FOLLOWUP_RUN_ROOT"
export FOLLOWUP_STUDY_ROOT
FOLLOWUP_STUDY_ROOT="$(mktemp -d "${FOLLOWUP_RUN_ROOT}/followup-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
mkdir -p "$FOLLOWUP_STUDY_ROOT/logs"
export FOLLOWUP_PLAN="${FOLLOWUP_STUDY_ROOT}/plan.json"
"${ENV_PREFIX}/bin/python" comparison/directional_followup.py prepare "${CHECK_ARGS[@]}" --plan "$FOLLOWUP_PLAN"
export FOLLOWUP_SOURCE_ROOT="${FOLLOWUP_STUDY_ROOT}/source"
ARRAY_RESPONSE="$(sbatch "${GPU_OPTIONS[@]}" --hold --output="$FOLLOWUP_STUDY_ROOT/logs/%A_%a.out" comparison/pace_directional_followup.sbatch)"
[[ "$ARRAY_RESPONSE" =~ ^[0-9]+(\;[^[:space:]\;]+)?$ ]] || { echo "Unexpected submission response: $ARRAY_RESPONSE; inspect squeue before retrying." >&2; exit 4; }
ARRAY_ID="${ARRAY_RESPONSE%%;*}"
if ! SUMMARY_RESPONSE="$(sbatch "${CPU_OPTIONS[@]}" --dependency="afterok:${ARRAY_ID}" --kill-on-invalid-dep=yes --output="$FOLLOWUP_STUDY_ROOT/logs/summary-%j.out" comparison/pace_directional_followup_summary.sbatch)"; then
    echo "Summary submission failed; cancelling newly held array $ARRAY_ID." >&2
    scancel "$ARRAY_ID" || echo "Could not cancel held array $ARRAY_ID; inspect squeue." >&2
    exit 5
fi
[[ "$SUMMARY_RESPONSE" =~ ^[0-9]+(\;[^[:space:]\;]+)?$ ]] || { echo "Unexpected summary response; array $ARRAY_ID remains held. Inspect squeue." >&2; exit 6; }
SUMMARY_ID="${SUMMARY_RESPONSE%%;*}"
"${ENV_PREFIX}/bin/python" - "$FOLLOWUP_STUDY_ROOT" "$ARRAY_ID" "$SUMMARY_ID" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
(root/'submission.json').write_text(json.dumps({'array_job':sys.argv[2], 'comparison_job':sys.argv[3], 'study_root':str(root)},indent=2)+'\n')
PY
if ! scontrol release "$ARRAY_ID"; then
    echo "Array $ARRAY_ID remains held; comparison $SUMMARY_ID is registered. Run: scontrol release $ARRAY_ID" >&2; exit 7
fi
printf 'Submitted GPU array: %s_[0-4]\nAutomatic comparison job: %s\nResults: %s\n' "$ARRAY_ID" "$SUMMARY_ID" "$FOLLOWUP_STUDY_ROOT"
printf 'Monitor: squeue -j %s,%s\n' "$ARRAY_ID" "$SUMMARY_ID"
