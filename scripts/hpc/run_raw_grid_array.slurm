#!/bin/bash
#SBATCH --job-name=lfp_raw_grid
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-lfp-raw-grid-%A_%a.out
#SBATCH --error=slurm-lfp-raw-grid-%A_%a.err

set -euo pipefail

# Run from the repository submitted to Slurm.
cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# Six tasks: three degrees crossed with two low-pass cutoffs.
DEGREES=(1 2 3)
LOWPASSES=(35 80)
DEGREE_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
LOWPASS_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
DEGREE="${DEGREES[$DEGREE_INDEX]}"
LOWPASS="${LOWPASSES[$LOWPASS_INDEX]}"
PART_NAME="part_lp${LOWPASS}_degree${DEGREE}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" scripts/pysindy/raw_grid_sweep.py \
  --trial-type fixation \
  --channel 0 \
  --test-fraction 0.25 \
  --seed 0 \
  --downsample 2 \
  --lowpass-list "$LOWPASS" \
  --degree-list "$DEGREE" \
  --n-delays-list 2,4,6,8 \
  --delay-list 1,2,5 \
  --smooth-window-list 0,5,9 \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
