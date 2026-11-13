#!/bin/bash
# Known working PACE profile from screen 15407679; overrides remain explicit.
set -euo pipefail
CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export ICEMIX_CHECKOUT="$CHECKOUT"
export ENV_PREFIX="${ENV_PREFIX:-/storage/project/r-itaboada3-0/rmaples3/software/envs/graphnet-directional-v2}"
export PILOT_SELECTION_ROOT="${PILOT_SELECTION_ROOT:-/storage/project/r-itaboada3-0/rmaples3/inputs/p14_frozen_10pct}"
export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74}"
export PACE_CONDA_MODULE="${PACE_CONDA_MODULE:-anaconda3/2022.05.0.1}"
export DIRECTIONAL_ACCOUNT="${DIRECTIONAL_ACCOUNT:-gts-itaboada3}"
export DIRECTIONAL_QOS="${DIRECTIONAL_QOS:-inferno}"
export FOLLOWUP_SCREEN_ROOT="${FOLLOWUP_SCREEN_ROOT:-${CHECKOUT}/comparison_runs/directional/screen-15407679}"
export FOLLOWUP_HISTORICAL_ROOT="${FOLLOWUP_HISTORICAL_ROOT:-${CHECKOUT}/comparison_runs/augmentation/full-14534898/both}"
exec bash "$CHECKOUT/comparison/submit_directional_followup.sh" "$@"
