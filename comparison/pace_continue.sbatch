#!/bin/bash
#SBATCH --job-name=IceMixContinue30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=icemix-augmentation-continue-%j.out

set -euo pipefail
PARENT_ROOT="${1:?Provide the completed screen directory}"
PARENT_ROOT="$(cd "$PARENT_ROOT" && pwd)"
CHECKOUT="${ICEMIX_CHECKOUT:-${SLURM_SUBMIT_DIR:?Submit from the fork root}}"
ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
CARLOS_ROOT="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
SELECTION_ROOT="${PILOT_SELECTION_ROOT:-${CARLOS_ROOT}/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
RUN_ROOT="${AUGMENTATION_RUN_ROOT:-${CHECKOUT}/comparison_runs/augmentation}"
TARGET_EPOCHS="${CONTINUE_EPOCHS:-30}"
cd "$CHECKOUT"
test -f comparison/continue_augmentation.py
test -x "${ENV_PREFIX}/bin/python"
test -d "$SELECTION_ROOT"
test -d "${TMPDIR:?Slurm node-local TMPDIR is required}"
for arm in baseline both; do
    test -f "${PARENT_ROOT}/${arm}/complete.json"
    test -s "${PARENT_ROOT}/${arm}/checkpoints/last.ckpt"
    # The previous validation-best checkpoint must remain available too.
    test -f "${PARENT_ROOT}/${arm}/selected_checkpoint.json"
done
module load anaconda3/2022.05.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${CHECKOUT}/src:${CHECKOUT}/comparison"
python -c 'from graphnet.models.transformer.icemix import IceMixReconstruction; from augmentation_callbacks import AugmentationCallback; print("Continuation imports passed")'
JOB_ROOT="${RUN_ROOT}/continue-${SLURM_JOB_ID}"
mkdir -p "$JOB_ROOT"
conda list --explicit > "${JOB_ROOT}/conda-explicit.txt"
conda env export > "${JOB_ROOT}/conda-environment.yml"
python -m pip freeze > "${JOB_ROOT}/pip-freeze.txt"
nvidia-smi > "${JOB_ROOT}/nvidia-smi.txt"
DATABASES=(P14_numu_database_part_1.db P14_nue_database_part_1.db P14_nutau_database_part_1.db)
for database in "${DATABASES[@]}"; do test -r "${SOURCE_DATA_ROOT}/${database}"; done
required_kb=$(du -ck -- "${DATABASES[@]/#/${SOURCE_DATA_ROOT}/}" | awk 'END {print $1}')
available_kb=$(df -Pk "$TMPDIR" | awk 'NR == 2 {print $4}')
if (( required_kb > available_kb * 9 / 10 )); then
    echo "Insufficient node-local space for P14 databases" >&2; exit 4
fi
DATA_ROOT="${TMPDIR}/icemix-continuation-${SLURM_JOB_ID}"
mkdir -p "$DATA_ROOT"
for database in "${DATABASES[@]}"; do
    cp -- "${SOURCE_DATA_ROOT}/${database}" "${DATA_ROOT}/${database}"
    test -s "${DATA_ROOT}/${database}"
done
sync -f "$DATA_ROOT"
for arm in baseline both; do
    srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/continue_augmentation.py train \
        --parent-run "${PARENT_ROOT}/${arm}" --source-root "$CHECKOUT" \
        --data-root "$DATA_ROOT" --selection-root "$SELECTION_ROOT" \
        --output-dir "${JOB_ROOT}/${arm}" --epochs "$TARGET_EPOCHS"
done
"${ENV_PREFIX}/bin/python" comparison/continue_augmentation.py summarize --run-root "$JOB_ROOT"
echo "Paired continuation complete: $JOB_ROOT"
echo "Review the comparison before submitting further training."
