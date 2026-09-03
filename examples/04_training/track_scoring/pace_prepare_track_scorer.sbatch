#!/bin/bash
# Build the reproducible 10% train/validation manifest on a PACE CPU node.
#
# Submit this file FROM THE GRAPHNET REPOSITORY ROOT:
#   sbatch examples/04_training/track_scoring/pace_prepare_track_scorer.sbatch

#SBATCH --job-name=tango-track-data
#SBATCH --account=gts-itaboada3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --qos=inferno
#SBATCH --time=02:00:00
#SBATCH --output=track-data-%j.out

set -euo pipefail

# SLURM_SUBMIT_DIR is the directory from which `sbatch` was run. Requiring the
# repository root avoids embedding one student's home-directory path here.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
if [[ ! -f "${REPO_ROOT}/setup.py" ]]; then
    echo "Submit from the graphnet repository root; setup.py was not found."
    exit 2
fi

DATA_ROOT="/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74"
TRACK_OUTPUT_ROOT="${TRACK_OUTPUT_ROOT:-${REPO_ROOT}/outputs/track_scoring}"
MANIFEST="${TRACK_OUTPUT_ROOT}/splits_10pct_seed42.csv"

# These are the six expansions of the path notation supplied for PACE. We
# check all six so spelling or permissions problems are caught together.
ALL_DATABASES=(
    "${DATA_ROOT}/P14_numu_database_part_1.db"
    "${DATA_ROOT}/P14_nue_database_part_1.db"
    "${DATA_ROOT}/P14_nutau_database_part_1.db"
    "${DATA_ROOT}/P14_nugen_numu_database_part_1.db"
    "${DATA_ROOT}/P14_nugen_nue_database_part_1.db"
    "${DATA_ROOT}/P14_nugen_nutau_database_part_1.db"
)

echo "Checking the six supplied PACE database paths:"
for database in "${ALL_DATABASES[@]}"; do
    if [[ ! -r "${database}" ]]; then
        echo "NOT READABLE (may be unused): ${database}"
    else
        echo "OK: ${database}"
    fi
done

# The TANGO report's track classifier uses GRECO nu_e and nu_mu events. It does
# not use the NuGen augmentation or nu_tau sample used by other reconstruction
# tasks. These two databases are therefore the deliberate default.
TRACK_DATABASES=(
    "${DATA_ROOT}/P14_numu_database_part_1.db"
    "${DATA_ROOT}/P14_nue_database_part_1.db"
)

for database in "${TRACK_DATABASES[@]}"; do
    if [[ ! -r "${database}" ]]; then
        echo "Required classification database is not readable: ${database}"
        exit 3
    fi
done

cd "${REPO_ROOT}"
module load anaconda3/2022.05.0.1
conda activate graphnet
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${TRACK_OUTPUT_ROOT}"

srun python examples/04_training/track_scoring/01_make_splits.py \
    --databases "${TRACK_DATABASES[@]}" \
    --output "${MANIFEST}" \
    --pulsemap SplitRTCleanedInIcePulses \
    --truth-table truth \
    --sample-fraction 0.10 \
    --validation-fraction 0.10 \
    --test-fraction 0.0 \
    --seed 42

echo "Manifest ready: ${MANIFEST}"
