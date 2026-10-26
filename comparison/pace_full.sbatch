#!/bin/bash
#SBATCH --job-name=IceMixFull
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=icemix-augmentation-full-%j.out

set -euo pipefail
PARENT_ROOT="${1:?Provide the completed 30-epoch continuation directory}"
PARENT_ROOT="$(cd "$PARENT_ROOT" && pwd)"
PARENT_RUN="${PARENT_ROOT}/both"
CHECKOUT="${ICEMIX_CHECKOUT:-${SLURM_SUBMIT_DIR:?Submit from the fork root}}"
ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
CARLOS_ROOT="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
SELECTION_ROOT="${PILOT_SELECTION_ROOT:-${CARLOS_ROOT}/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
RUN_ROOT="${AUGMENTATION_RUN_ROOT:-${CHECKOUT}/comparison_runs/augmentation}"
TARGET_EPOCHS="${FULL_EPOCHS:-80}"
PATIENCE="${EARLY_STOP_PATIENCE:-12}"
MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
cd "$CHECKOUT"
test -f comparison/full_augmentation.py
test -x "${ENV_PREFIX}/bin/python"
test -d "$SELECTION_ROOT"
test -d "${TMPDIR:?Slurm node-local TMPDIR is required}"
test -f "${PARENT_RUN}/complete.json"
test -s "${PARENT_RUN}/checkpoints/last.ckpt"
test -f "${PARENT_RUN}/selected_checkpoint.json"
JOB_ROOT="${RUN_ROOT}/full-${SLURM_JOB_ID}"
if [[ -e "$JOB_ROOT" ]]; then
    echo "Refusing to overwrite existing allocation results: $JOB_ROOT" >&2
    exit 3
fi
mkdir -p "$JOB_ROOT"
printf 'phase,start_utc,end_utc,seconds,exit_code\n' > "${JOB_ROOT}/phase_timing.csv"
JOB_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
JOB_STARTED_SECONDS="$(date +%s)"
PHASE=""
PHASE_STARTED_UTC=""
PHASE_STARTED_SECONDS=0
TELEMETRY_PID=""
end_phase() {
    local status="$1"
    if [[ -n "$PHASE" ]]; then
        printf '%s,%s,%s,%s,%s\n' "$PHASE" "$PHASE_STARTED_UTC" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$(( $(date +%s) - PHASE_STARTED_SECONDS ))" "$status" \
            >> "${JOB_ROOT}/phase_timing.csv"
    fi
}
begin_phase() {
    end_phase 0
    PHASE="$1"
    PHASE_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    PHASE_STARTED_SECONDS="$(date +%s)"
}
finish() {
    local status=$?
    end_phase "$status"
    if [[ -n "$TELEMETRY_PID" ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
    fi
    printf 'exit_code=%s\nlast_phase=%s\njob_id=%s\nstarted_utc=%s\nended_utc=%s\nwrapper_seconds=%s\n' \
        "$status" "$PHASE" "$SLURM_JOB_ID" "$JOB_STARTED_UTC" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(( $(date +%s) - JOB_STARTED_SECONDS ))" \
        > "${JOB_ROOT}/slurm_status.txt"
    exit "$status"
}
trap finish EXIT
begin_phase environment_and_imports
module load anaconda3/2022.05.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${CHECKOUT}/src:${CHECKOUT}/comparison"
python -c 'from graphnet.models.transformer.icemix import IceMixReconstruction; from full_augmentation import main; print("Full-run imports passed")'
conda list --explicit > "${JOB_ROOT}/conda-explicit.txt"
conda env export > "${JOB_ROOT}/conda-environment.yml"
python -m pip freeze > "${JOB_ROOT}/pip-freeze.txt"
nvidia-smi > "${JOB_ROOT}/nvidia-smi.txt"
# Query only the GPU assigned to this allocation. Telemetry is optional.
GPU_TARGET="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -n "$GPU_TARGET" && "$GPU_TARGET" != *,* ]]; then
    nvidia-smi -i "$GPU_TARGET" \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv -l "${GPU_TELEMETRY_SECONDS:-30}" \
        > "${JOB_ROOT}/gpu_telemetry.csv" 2> "${JOB_ROOT}/gpu_telemetry.stderr" &
    TELEMETRY_PID=$!
else
    echo "Assigned GPU ID unavailable; per-epoch allocated GPU memory is still recorded." \
        > "${JOB_ROOT}/gpu_telemetry.stderr"
fi
begin_phase database_staging
DATABASES=(P14_numu_database_part_1.db P14_nue_database_part_1.db P14_nutau_database_part_1.db)
for database in "${DATABASES[@]}"; do test -r "${SOURCE_DATA_ROOT}/${database}"; done
required_kb=$(du -ck -- "${DATABASES[@]/#/${SOURCE_DATA_ROOT}/}" | awk 'END {print $1}')
available_kb=$(df -Pk "$TMPDIR" | awk 'NR == 2 {print $4}')
if (( required_kb > available_kb * 9 / 10 )); then
    echo "Insufficient node-local space for P14 databases" >&2
    exit 4
fi
DATA_ROOT="${TMPDIR}/icemix-full-${SLURM_JOB_ID}"
mkdir -p "$DATA_ROOT"
for database in "${DATABASES[@]}"; do
    cp -- "${SOURCE_DATA_ROOT}/${database}" "${DATA_ROOT}/${database}"
    test -s "${DATA_ROOT}/${database}"
done
sync -f "$DATA_ROOT"
begin_phase training_and_validation_export
srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/full_augmentation.py train \
    --parent-run "$PARENT_RUN" --source-root "$CHECKOUT" \
    --data-root "$DATA_ROOT" --selection-root "$SELECTION_ROOT" \
    --output-dir "${JOB_ROOT}/both" --epochs "$TARGET_EPOCHS" \
    --patience "$PATIENCE" --min-delta "$MIN_DELTA"
begin_phase final_summary
"${ENV_PREFIX}/bin/python" comparison/full_augmentation.py summarize --run-root "$JOB_ROOT"
echo "Full run completed successfully (early stopping is a successful outcome): $JOB_ROOT"
echo "Inspect complete.json and summary.csv for actual epoch count and stop reason."
