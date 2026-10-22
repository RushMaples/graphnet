#!/bin/bash
#SBATCH --job-name=IceMixAugment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=icemix-augmentation-%j.out

set -euo pipefail
STAGE="${1:-screen}"
SCREEN_ROOT="${2:-}"
case "$STAGE" in
    screen|full) ;;
    *) echo "Usage: sbatch comparison/pace_augment.sbatch screen|full [SCREEN_ROOT]" >&2; exit 2 ;;
esac
CHECKOUT="${ICEMIX_CHECKOUT:-${SLURM_SUBMIT_DIR:?Submit from the fork root}}"
ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
CARLOS_ROOT="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
SELECTION_ROOT="${PILOT_SELECTION_ROOT:-${CARLOS_ROOT}/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
RUN_ROOT="${AUGMENTATION_RUN_ROOT:-${CHECKOUT}/comparison_runs/augmentation}"
SCREEN_EPOCHS="${SCREEN_EPOCHS:-10}"
FULL_EPOCHS="${FULL_EPOCHS:-80}"
TOKEN_DROP="${TOKEN_DROP:-0.05}"
DROP_CHANCE="${DROP_CHANCE:-0.05}"
if [[ "$STAGE" == full ]]; then
    test -f "${SCREEN_ROOT:?Full stage needs a completed screen directory}/selected_arm.json"
fi
cd "$CHECKOUT"
test -f comparison/augmentation.py
test -x "${ENV_PREFIX}/bin/python"
test -d "$SELECTION_ROOT"
test -d "${TMPDIR:?Slurm node-local TMPDIR is required}"

module load anaconda3/2022.05.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${CHECKOUT}/src:${CHECKOUT}/comparison"
JOB_ROOT="${RUN_ROOT}/${STAGE}-${SLURM_JOB_ID}"
mkdir -p "$JOB_ROOT"
conda list --explicit > "${JOB_ROOT}/conda-explicit.txt"
conda env export > "${JOB_ROOT}/conda-environment.yml"
python -m pip freeze > "${JOB_ROOT}/pip-freeze.txt"
nvidia-smi > "${JOB_ROOT}/nvidia-smi.txt"
# Catch imports/API differences before staging the large databases.
python -c 'from graphnet.models.transformer.icemix import IceMixReconstruction; from augmentation_callbacks import AugmentationCallback; m=IceMixReconstruction(token_drop=0.05, drop_chance=0.05); assert hasattr(m.backbone, "set_token_drop_seed"); print("Augmentation imports and constructor passed")'

DATABASES=(P14_numu_database_part_1.db P14_nue_database_part_1.db P14_nutau_database_part_1.db)
for database in "${DATABASES[@]}"; do test -r "${SOURCE_DATA_ROOT}/${database}"; done
required_kb=$(du -ck -- "${DATABASES[@]/#/${SOURCE_DATA_ROOT}/}" | awk 'END {print $1}')
available_kb=$(df -Pk "$TMPDIR" | awk 'NR == 2 {print $4}')
if (( required_kb > available_kb * 9 / 10 )); then
    echo "Insufficient node-local space for P14 databases" >&2; exit 4
fi
DATA_ROOT="${TMPDIR}/icemix-augmentation-${SLURM_JOB_ID}"
mkdir -p "$DATA_ROOT"
for database in "${DATABASES[@]}"; do
    cp -- "${SOURCE_DATA_ROOT}/${database}" "${DATA_ROOT}/${database}"
    test -s "${DATA_ROOT}/${database}"
done
sync -f "$DATA_ROOT"
COMMON=(--source-root "$CHECKOUT" --data-root "$DATA_ROOT" --selection-root "$SELECTION_ROOT")
if [[ "$STAGE" == screen ]]; then
    for arm in baseline pulse-dropout rotation both; do
        srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/augmentation.py train \
            --stage screen --arm "$arm" --epochs "$SCREEN_EPOCHS" \
            --token-drop "$TOKEN_DROP" --drop-chance "$DROP_CHANCE" \
            --output-dir "${JOB_ROOT}/${arm}" "${COMMON[@]}"
    done
    "${ENV_PREFIX}/bin/python" comparison/augmentation.py summarize --screen-root "$JOB_ROOT"
else
    srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/augmentation.py train \
        --stage full --epochs "$FULL_EPOCHS" --screen-root "$SCREEN_ROOT" \
        --output-dir "${JOB_ROOT}/selected" "${COMMON[@]}"
fi
echo "Completed ${STAGE}: ${JOB_ROOT}"
