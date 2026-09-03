#!/bin/bash
# Train the track scorer on a PACE RTX 6000 node.
#
# Submit FROM THE GRAPHNET REPOSITORY ROOT. The safe default is a preflight:
#   sbatch examples/04_training/track_scoring/pace_train_track_scorer.sbatch
#
# Select a longer run without editing this file:
#   sbatch --export=ALL,RUN_MODE=smoke \
#     examples/04_training/track_scoring/pace_train_track_scorer.sbatch
#   sbatch --export=ALL,RUN_MODE=full,RUN_TAG=full_seed42 \
#     examples/04_training/track_scoring/pace_train_track_scorer.sbatch

#SBATCH --job-name=tango-track-score
#SBATCH --account=gts-itaboada3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=inferno
#SBATCH --mem-per-gpu=48G
#SBATCH --time=20:00:00
#SBATCH --output=track-scoring-%j.out

set -euo pipefail

# Derive paths from the repository where `sbatch` was invoked. You may redirect
# all outputs by exporting TRACK_OUTPUT_ROOT when submitting.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
if [[ ! -f "${REPO_ROOT}/setup.py" ]]; then
    echo "Submit from the graphnet repository root; setup.py was not found."
    exit 2
fi

TRACK_OUTPUT_ROOT="${TRACK_OUTPUT_ROOT:-${REPO_ROOT}/outputs/track_scoring}"
MANIFEST="${MANIFEST:-${TRACK_OUTPUT_ROOT}/splits_10pct_seed42.csv}"
RUN_MODE="${RUN_MODE:-preflight}"
RUN_TAG="${RUN_TAG:-${RUN_MODE}_seed42}"
OUTPUT_DIR="${TRACK_OUTPUT_ROOT}/${RUN_TAG}"

cd "${REPO_ROOT}"

# These match the PACE-tested reconstruction jobs already in this repository.
module load anaconda3/2022.05.0.1
conda activate graphnet

# Import this checkout rather than a different GraphNeT installation.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable: ${MANIFEST}"
    echo "Run pace_prepare_track_scorer.sbatch first."
    exit 3
fi

case "${RUN_MODE}" in
    preflight)
        # Loads one small batch, runs forward/loss checks, then exits.
        RUN_ARGUMENTS=(
            --dry-run
            --limit-train 256
            --limit-validation 128
            --batch-size 8
            --max-epochs 1
        )
        ;;
    smoke)
        # Short end-to-end run that also writes predictions and checkpoints.
        RUN_ARGUMENTS=(
            --limit-train 10000
            --limit-validation 2000
            --batch-size 16
            --max-epochs 2
        )
        ;;
    full)
        RUN_ARGUMENTS=(
            --batch-size 32
            --max-epochs 40
        )
        ;;
    *)
        echo "RUN_MODE must be one of: preflight, smoke, full"
        exit 4
        ;;
esac

RESUME_ARGUMENTS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    RESUME_ARGUMENTS=(--resume-from "${RESUME_FROM}")
fi

echo "Run mode:  ${RUN_MODE}"
echo "Manifest:  ${MANIFEST}"
echo "Output:    ${OUTPUT_DIR}"
nvidia-smi

srun python examples/04_training/track_scoring/02_train_track_scorer.py \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --pulsemap SplitRTCleanedInIcePulses \
    --truth-table truth \
    --gpus 0 \
    --early-stopping-patience 6 \
    --scheduler-patience 2 \
    --learning-rate 0.001 \
    --num-workers 6 \
    --seed 42 \
    "${RUN_ARGUMENTS[@]}" \
    "${RESUME_ARGUMENTS[@]}"
