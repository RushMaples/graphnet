#!/bin/bash
#SBATCH --job-name=IceMixTest
#SBATCH --account=gts-itaboada3
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=icemix-test-%j.out

set -euo pipefail
[[ $# == 0 ]] || { echo 'Usage: pace_test.sbatch' >&2; exit 2; }
SAMPLE=reused-benchmark
CHECKOUT="${ICEMIX_CHECKOUT:-${SLURM_SUBMIT_DIR:?Submit from the fork root}}"
ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
CARLOS_ROOT="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
SELECTION_ROOT="${PILOT_SELECTION_ROOT:-${CARLOS_ROOT}/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
TRAINING_RUN="${CHECKOUT}/comparison_runs/augmentation/full-14534898/both"
JOB_ROOT="${CHECKOUT}/comparison_runs/test/test-${SLURM_JOB_ID:?}"
test ! -e "$JOB_ROOT"
test -d "${TMPDIR:?Slurm node-local TMPDIR is required}"
mkdir -p "$JOB_ROOT"
cd "$CHECKOUT"
printf 'phase,seconds\n' > "${JOB_ROOT}/phase_timing.csv"
STARTED=$SECONDS
PHASE=environment
PHASE_STARTED=$SECONDS
finish() {
    local status=$?
    printf '%s,%s\n' "$PHASE" "$((SECONDS - PHASE_STARTED))" >> "${JOB_ROOT}/phase_timing.csv"
    printf 'exit_code=%s\nsample=%s\njob_id=%s\nnode=%s\nwrapper_seconds=%s\n' \
        "$status" "$SAMPLE" "$SLURM_JOB_ID" "${SLURMD_NODENAME:-unknown}" "$((SECONDS - STARTED))" \
        > "${JOB_ROOT}/slurm_status.txt"
    exit "$status"
}
trap finish EXIT
module load anaconda3/2022.05.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${CHECKOUT}/src:${CHECKOUT}/comparison"
export PYTHONPYCACHEPREFIX="${TMPDIR}/icemix-cache/pycache" PIP_CACHE_DIR="${TMPDIR}/icemix-cache/pip"
export UV_CACHE_DIR="${TMPDIR}/icemix-cache/uv" MPLCONFIGDIR="${TMPDIR}/icemix-cache/matplotlib"
printf '%s,%s\n' "$PHASE" "$((SECONDS - PHASE_STARTED))" >> "${JOB_ROOT}/phase_timing.csv"
PHASE=gpu_health
PHASE_STARTED=$SECONDS
# Catch an unusable assigned device before copying the P14 databases.
srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" - "${JOB_ROOT}/gpu_health.json" <<'PY'
import json
import os
import sys
import time
from pathlib import Path
import torch

started = time.perf_counter()
report = {
    "passed": False,
    "node": os.environ.get("SLURMD_NODENAME"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
}
try:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU health check requires exactly one CUDA GPU")
    report["gpu"] = torch.cuda.get_device_name(0)
    count = 4096
    values = torch.arange(count, device="cuda", dtype=torch.float32)
    total = (values + 1).sum()
    torch.cuda.synchronize()
    if not torch.isfinite(total).item() or total.item() != count * (count + 1) / 2:
        raise RuntimeError("GPU health kernel returned an invalid result")
    report["passed"] = True
except Exception as error:
    report["error_type"] = type(error).__name__
    report["error"] = str(error)
    raise
finally:
    report["seconds"] = time.perf_counter() - started
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print("GPU health check passed:", report["gpu"])
PY
printf '%s,%s\n' "$PHASE" "$((SECONDS - PHASE_STARTED))" >> "${JOB_ROOT}/phase_timing.csv"
PHASE=database_staging
PHASE_STARTED=$SECONDS
DATABASES=(P14_numu_database_part_1.db P14_nue_database_part_1.db P14_nutau_database_part_1.db)
required_kb=$(du -ck -- "${DATABASES[@]/#/${SOURCE_DATA_ROOT}/}" | awk 'END {print $1}')
available_kb=$(df -Pk "$TMPDIR" | awk 'NR == 2 {print $4}')
(( required_kb <= available_kb * 9 / 10 )) || { echo 'Insufficient node-local space' >&2; exit 3; }
DATA_ROOT="${TMPDIR}/icemix-test-${SLURM_JOB_ID}"
mkdir -p "$DATA_ROOT"
for database in "${DATABASES[@]}"; do cp -- "${SOURCE_DATA_ROOT}/${database}" "${DATA_ROOT}/${database}"; done
sync -f "$DATA_ROOT"
printf '%s,%s\n' "$PHASE" "$((SECONDS - PHASE_STARTED))" >> "${JOB_ROOT}/phase_timing.csv"
PHASE=parity_and_test
PHASE_STARTED=$SECONDS
srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/frozen_test.py \
    --run-root "$TRAINING_RUN" --source-root "$CHECKOUT" --data-root "$DATA_ROOT" \
    --selection-root "$SELECTION_ROOT" --output-dir "${JOB_ROOT}/results"
echo "Completed raw-model $SAMPLE evaluation: ${JOB_ROOT}/results"
