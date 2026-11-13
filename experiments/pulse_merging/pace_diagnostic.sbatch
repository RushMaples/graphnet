#!/bin/bash
#SBATCH --job-name=P14PulseAudit
#SBATCH --account=gts-itaboada3
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=pulse-diagnostic-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the graphnet root}"
env_prefix="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet}"
data="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
carlos="${CARLOS_ROOT:-/storage/project/r-itaboada3-0/rmaples3/software/cursor/graphnet}"
selections="${PILOT_SELECTION_ROOT:-$carlos/ice_mix/outputs/pilot_10pct_p14_three_seed42/selections}"
result_root="${PULSE_RESULTS_ROOT:-$PWD/comparison_runs/pulse_diagnostic}"
result="$result_root/pulse-diagnostic-${SLURM_JOB_ID:?}.zip"
test -r experiments/pulse_merging/pulse_diagnostic.py
test -d "$data"
test -d "$selections"
module load anaconda3/2022.05.0.1
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_prefix"
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/p14-pulse-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"
srun --ntasks=1 --cpu-bind=cores "$env_prefix/bin/python" experiments/pulse_merging/pulse_diagnostic.py \
    --data-root "$data" --selection-root "$selections" --output "$result" "$@"
