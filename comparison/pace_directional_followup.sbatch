#!/bin/bash
#SBATCH --job-name=IceMixFollowup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
set -euo pipefail
RUNS=(joint_vmf joint_king_no_count joint_flow direction_vmf direction_flow)
INDEX="${SLURM_ARRAY_TASK_ID:?Use start_directional_followup.sh}"
[[ "$INDEX" =~ ^[0-4]$ ]] || { echo 'Invalid follow-up task ID' >&2; exit 2; }
RUN_NAME="${RUNS[$INDEX]}"
cd "${FOLLOWUP_SOURCE_ROOT:?Frozen submission source required}"
: "${FOLLOWUP_STUDY_ROOT:?}" "${FOLLOWUP_PLAN:?}" "${ENV_PREFIX:?}" "${PACE_CONDA_MODULE:?}"
: "${SOURCE_DATA_ROOT:?}" "${PILOT_SELECTION_ROOT:?}" "${TMPDIR:?Node-local TMPDIR required}"
JOB_ROOT="${FOLLOWUP_STUDY_ROOT}/runs/${RUN_NAME}"
[[ ! -e "$JOB_ROOT" ]] || { echo "Refusing to overwrite $JOB_ROOT" >&2; exit 3; }
mkdir -p "$JOB_ROOT"
trap 'status=$?; printf "exit_code=%s\n" "$status" > "${JOB_ROOT}/slurm_status.txt"' EXIT
module load "$PACE_CONDA_MODULE"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${FOLLOWUP_SOURCE_ROOT}/src:${FOLLOWUP_SOURCE_ROOT}/comparison" SLURM_EXPORT_ENV=ALL
export PYTHONPYCACHEPREFIX="${TMPDIR}/icemix-followup-cache/pycache" PIP_CACHE_DIR="${TMPDIR}/icemix-followup-cache/pip"
export UV_CACHE_DIR="${TMPDIR}/icemix-followup-cache/uv" MPLCONFIGDIR="${TMPDIR}/icemix-followup-cache/matplotlib"
conda list --explicit > "$JOB_ROOT/conda-explicit.txt"
"${ENV_PREFIX}/bin/python" -m pip freeze > "$JOB_ROOT/pip-freeze.txt"
nvidia-smi > "$JOB_ROOT/nvidia-smi.txt"
scontrol show job --oneliner "$SLURM_JOB_ID" > "$JOB_ROOT/slurm_job_at_start.txt" || true
srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" - "$JOB_ROOT/gpu_health.json" <<'PY'
import json, sys, torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
x=torch.arange(32,device='cuda',dtype=torch.float32)
y=(x*x).sum()
torch.cuda.synchronize()
assert torch.isfinite(y) and y.item()==10416
with open(sys.argv[1],'w') as handle:
 json.dump({'passed':True,'gpu':torch.cuda.get_device_name(0)},handle)
PY
DATABASES=(P14_numu_database_part_1.db P14_nue_database_part_1.db P14_nutau_database_part_1.db)
for database in "${DATABASES[@]}"; do test -r "$SOURCE_DATA_ROOT/$database"; done
required_kb=$(du -ck -- "${DATABASES[@]/#/${SOURCE_DATA_ROOT}/}" | awk 'END {print $1}')
available_kb=$(df -Pk "$TMPDIR" | awk 'NR==2 {print $4}')
(( required_kb <= available_kb * 9 / 10 )) || { echo 'Insufficient node-local database space' >&2; exit 4; }
DATA_ROOT="$TMPDIR/icemix-followup-${SLURM_JOB_ID}-${INDEX}"
mkdir -p "$DATA_ROOT"
for database in "${DATABASES[@]}"; do
    cp -- "$SOURCE_DATA_ROOT/$database" "$DATA_ROOT/$database"
    test -s "$DATA_ROOT/$database"
done
sync -f "$DATA_ROOT"
srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/directional_followup.py train --plan "$FOLLOWUP_PLAN" --run-name "$RUN_NAME" --output-dir "$JOB_ROOT/model" --data-root "$DATA_ROOT" --selection-root "$PILOT_SELECTION_ROOT"
echo "Completed $RUN_NAME. Automatic comparison waits for every array member."
