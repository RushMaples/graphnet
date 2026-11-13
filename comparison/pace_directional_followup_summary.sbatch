#!/bin/bash
#SBATCH --job-name=IceMixFollowupSummary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${FOLLOWUP_SOURCE_ROOT:?Frozen submission source required}"
: "${ENV_PREFIX:?}" "${PACE_CONDA_MODULE:?}" "${FOLLOWUP_STUDY_ROOT:?}" "${FOLLOWUP_PLAN:?}"
module load "$PACE_CONDA_MODULE"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${FOLLOWUP_SOURCE_ROOT}/src:${FOLLOWUP_SOURCE_ROOT}/comparison"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/icemix-followup-summary/pycache"
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/icemix-followup-summary/pip"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/icemix-followup-summary/uv"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/icemix-followup-summary/matplotlib"
srun --ntasks=1 "${ENV_PREFIX}/bin/python" comparison/compare_directional_followup.py --plan "$FOLLOWUP_PLAN" --study-root "$FOLLOWUP_STUDY_ROOT"
echo "Ready: $FOLLOWUP_STUDY_ROOT/summary/results.html"
echo "Upload for review: $FOLLOWUP_STUDY_ROOT/directional-followup-results.tar.gz"
