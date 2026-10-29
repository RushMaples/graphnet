#!/bin/bash
# Submit one inference allocation on the previously used test benchmark.
set -euo pipefail
if [[ $# -gt 1 || ( -n "${1:-}" && "$1" != --dry-run ) ]]; then
    echo 'Usage: bash comparison/submit_test.sh [--dry-run]' >&2
    exit 2
fi
cd "${ICEMIX_CHECKOUT:-$(pwd)}"
test -f comparison/pace_test.sbatch
# Do not inherit a previous hold, excluded node, reservation or dependency.
for variable in "${!SBATCH_@}"; do unset "$variable"; done
unset SLURM_CLUSTERS SLURM_HINT SLURM_MEM_BIND EXCLUDE_NODES
COMMAND=(sbatch --parsable --account=gts-itaboada3 --qos=inferno --partition=gpu-l40s
    --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:l40s:1 --mem=64G --time=01:00:00
    comparison/pace_test.sbatch)
printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
if [[ "${1:-}" == --dry-run ]]; then echo 'Dry run: no job submitted.'; exit 0; fi
SUBMISSION="$("${COMMAND[@]}")"
[[ "$SUBMISSION" =~ ^[0-9]+(\;[^[:space:]\;]+)?$ ]] || {
    echo "Unexpected submission response; inspect before resubmitting: $SUBMISSION" >&2; exit 3;
}
printf 'Submitted batch job %s. Results: comparison_runs/test/test-%s/results\n' "${SUBMISSION%%;*}" "${SUBMISSION%%;*}"
