#!/bin/bash
#SBATCH --job-name=IceMixDirection
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=icemix-directional-%A-%a.out
set -euo pipefail
ARMS=(vmf king flow)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:?Submit through submit_directional.sh}]}"
CHECKOUT="${ICEMIX_CHECKOUT:-${SLURM_SUBMIT_DIR:?Submit from fork root}}"
ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet-directional}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
CARLOS_ROOT="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
SELECTION_ROOT="${PILOT_SELECTION_ROOT:-${CARLOS_ROOT}/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
RUN_ROOT="${DIRECTIONAL_RUN_ROOT:-${CHECKOUT}/comparison_runs/directional}"
FLOW_LAYERS="${FLOW_LAYERS:-4}"
case "$FLOW_LAYERS" in 4|16) ;; *) echo 'FLOW_LAYERS must be 4 or 16' >&2; exit 2;; esac
cd "$CHECKOUT"
test -f comparison/directional_run.py
test -d "$SELECTION_ROOT"
test -x "${ENV_PREFIX}/bin/python"
test -d "${TMPDIR:?Slurm node-local TMPDIR required}"
JOB_ROOT="${RUN_ROOT}/screen-${SLURM_ARRAY_JOB_ID}/${ARM}"
if [[ -e "$JOB_ROOT" ]]; then echo "Refusing to overwrite $JOB_ROOT" >&2; exit 3; fi
mkdir -p "$JOB_ROOT"
trap 'status=$?; printf "exit_code=%s\n" "$status" > "${JOB_ROOT}/slurm_status.txt"' EXIT
module load anaconda3/2022.05.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${CHECKOUT}/src:${CHECKOUT}/comparison"
export SLURM_EXPORT_ENV=ALL
export PYTHONPYCACHEPREFIX="${TMPDIR}/icemix-cache/pycache" PIP_CACHE_DIR="${TMPDIR}/icemix-cache/pip"
export UV_CACHE_DIR="${TMPDIR}/icemix-cache/uv" MPLCONFIGDIR="${TMPDIR}/icemix-cache/matplotlib"
conda list --explicit > "${JOB_ROOT}/conda-explicit.txt"
conda env export > "${JOB_ROOT}/conda-environment.yml"
python -m pip freeze > "${JOB_ROOT}/pip-freeze.txt"
nvidia-smi > "${JOB_ROOT}/nvidia-smi.txt"
scontrol show job --oneliner "$SLURM_JOB_ID" > "${JOB_ROOT}/slurm_job_at_start.txt" || true
# Fail before data staging if CUDA hardware or flow API is unusable.
srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" - "${JOB_ROOT}/gpu_health.json" <<'PY'
import json, sys, torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.arange(32, device='cuda', dtype=torch.float32)
y = (x*x).sum()
torch.cuda.synchronize()
assert torch.isfinite(y) and y.item() == 10416
with open(sys.argv[1], 'w') as f:
    json.dump({'passed': True, 'gpu': torch.cuda.get_device_name(0)}, f)
PY
"${ENV_PREFIX}/bin/python" comparison/directional_run.py check-environment > "${JOB_ROOT}/environment_check.txt"
"${ENV_PREFIX}/bin/python" comparison/directional_run.py check-inputs \
    --selection-root "$SELECTION_ROOT" --data-root "$SOURCE_DATA_ROOT" > "${JOB_ROOT}/input_check.txt"
DATABASES=(P14_numu_database_part_1.db P14_nue_database_part_1.db P14_nutau_database_part_1.db)
for database in "${DATABASES[@]}"; do test -r "${SOURCE_DATA_ROOT}/${database}"; done
required_kb=$(du -ck -- "${DATABASES[@]/#/${SOURCE_DATA_ROOT}/}" | awk 'END {print $1}')
available_kb=$(df -Pk "$TMPDIR" | awk 'NR == 2 {print $4}')
if (( required_kb > available_kb * 9 / 10 )); then
    echo 'Insufficient local space for P14 databases' >&2; exit 4
fi
DATA_ROOT="${TMPDIR}/icemix-directional-${SLURM_JOB_ID}"
mkdir -p "$DATA_ROOT"
for database in "${DATABASES[@]}"; do
    cp -- "${SOURCE_DATA_ROOT}/${database}" "${DATA_ROOT}/${database}"
    test -s "${DATA_ROOT}/${database}"
done
sync -f "$DATA_ROOT"
# Wrapper metadata lives alongside model artifacts: driver uses model/ so its
# no-overwrite contract stays strict even after successful environment setup.
srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/directional_run.py train \
    --arm "$ARM" --source-root "$CHECKOUT" --data-root "$DATA_ROOT" \
    --selection-root "$SELECTION_ROOT" --output-dir "${JOB_ROOT}/model" \
    --flow-layers "$FLOW_LAYERS" --export-validation --samples "${POSTERIOR_SAMPLES:-128}"
echo "Completed ${ARM}: ${JOB_ROOT}/model"
