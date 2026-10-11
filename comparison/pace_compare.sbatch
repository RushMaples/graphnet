#!/bin/bash
#SBATCH --job-name=IceMixCompare10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=icemix-comparison-%j.out

# Set account/partition/QoS using sbatch options or SBATCH_* environment variables.
# The archived pilot's allocation is not necessarily available to this user.
# See comparison/README.md for read-only allocation discovery and --test-only.
set -euo pipefail

# Submit from the new fork's root. Do not reinstall into the working pilot env.
CHECKOUT="${ICEMIX_CHECKOUT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the fork root}}"
CARLOS_ROOT="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
SELECTION_ROOT="${PILOT_SELECTION_ROOT:-${CARLOS_ROOT}/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
RUN_ROOT="${COMPARISON_RUN_ROOT:-${CHECKOUT}/comparison_runs}"
VARIANT="${1:-check}"
case "$VARIANT" in
    check) VARIANTS=(carlos ajai); MODE=(--preflight-only) ;;
    carlos|ajai)
        VARIANTS=("$VARIANT")
        : "${COMPARISON_PREFLIGHT_ROOT:?Set COMPARISON_PREFLIGHT_ROOT to the successful check job directory}"
        test -f "${COMPARISON_PREFLIGHT_ROOT}/paired_preflight.json"
        MODE=(--preflight-receipt "${COMPARISON_PREFLIGHT_ROOT}/paired_preflight.json") ;;
    *) echo "Usage: sbatch comparison/pace_compare.sbatch [check|carlos|ajai]" >&2; exit 2 ;;
esac

cd "$CHECKOUT"
test -f comparison/run.py
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "[FATAL] Pilot environment missing: ${ENV_PREFIX}/bin/python" >&2
    echo "Source recovery does not restore a deleted Conda environment." >&2
    exit 5
fi
if [[ ! -d "$SELECTION_ROOT" ]]; then
    echo "[FATAL] Frozen pilot selections missing: $SELECTION_ROOT" >&2
    echo "See comparison/README.md: Recovering a deleted Carlos checkout. Do not regenerate the splits." >&2
    exit 6
fi
for variant in "${VARIANTS[@]}"; do
    if [[ "$variant" == carlos && ! -f "${CARLOS_ROOT}/ice_mix/src/models/transformer.py" ]]; then
        echo "[FATAL] Carlos baseline source missing: $CARLOS_ROOT" >&2
        echo "Restore the pinned checkout with comparison/recover_carlos.py, or set CARLOS_ROOT to an existing baseline." >&2
        exit 7
    fi
done
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    echo "[FATAL] Slurm node-local TMPDIR is required" >&2; exit 3
fi

module load anaconda3/2022.05.0.1
set +u
# Explicit initialization avoids the previous 'conda activate' shell problem.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED=42
export PYTHONDONTWRITEBYTECODE=1
JOB_ROOT="${RUN_ROOT}/job-${SLURM_JOB_ID}"
mkdir -p "$JOB_ROOT"
conda list --explicit > "${JOB_ROOT}/conda-explicit.txt"
conda env export > "${JOB_ROOT}/conda-environment.yml"
python -m pip freeze > "${JOB_ROOT}/pip-freeze.txt"
python -c 'import sys; print(sys.executable); print(sys.version)' > "${JOB_ROOT}/python.txt"
nvidia-smi > "${JOB_ROOT}/nvidia-smi.txt"

DATABASES=(P14_numu_database_part_1.db P14_nue_database_part_1.db P14_nutau_database_part_1.db)
for database in "${DATABASES[@]}"; do test -r "${SOURCE_DATA_ROOT}/${database}"; done
required_kb=$(du -ck -- "${DATABASES[@]/#/${SOURCE_DATA_ROOT}/}" | awk 'END {print $1}')
available_kb=$(df -Pk "$TMPDIR" | awk 'NR == 2 {print $4}')
if (( required_kb > available_kb * 9 / 10 )); then
    echo "[FATAL] Insufficient node-local disk for P14 databases" >&2; exit 4
fi
export DATA_ROOT="${TMPDIR}/icemix-comparison-${SLURM_JOB_ID}"
mkdir -p "$DATA_ROOT"
for database in "${DATABASES[@]}"; do
    cp -- "${SOURCE_DATA_ROOT}/${database}" "${DATA_ROOT}/${database}"
    test -s "${DATA_ROOT}/${database}"
done
sync -f "$DATA_ROOT"

for variant in "${VARIANTS[@]}"; do
    if [[ "$variant" == carlos ]]; then SOURCE_ROOT="$CARLOS_ROOT"; else SOURCE_ROOT="$CHECKOUT"; fi
    test -f "${SOURCE_ROOT}/src/graphnet/__init__.py"
    # Never let an old editable installation choose the other model's package.
    export PYTHONPATH="${SOURCE_ROOT}/src:${CHECKOUT}/comparison"
    srun --ntasks=1 --cpu-bind=cores "${ENV_PREFIX}/bin/python" comparison/run.py --variant "$variant" --source-root "$SOURCE_ROOT" --data-root "$DATA_ROOT" --selection-root "$SELECTION_ROOT" --output-dir "${JOB_ROOT}/${variant}" "${MODE[@]}"
done
if [[ "$VARIANT" == check ]]; then
    "${ENV_PREFIX}/bin/python" comparison/evaluate.py --probe "${JOB_ROOT}/carlos" "${JOB_ROOT}/ajai"
fi
