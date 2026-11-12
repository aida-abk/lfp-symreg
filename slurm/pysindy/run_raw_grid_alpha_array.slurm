#!/bin/bash
#SBATCH --job-name=lfp_alpha_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-alpha-sweep-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-alpha-sweep-%A_%a.err

# STLSQ ridge (alpha) sweep: alpha in {0.0, 0.01, 0.05, 0.1, 0.5, 1.0}.
# 216 base configs × 6 alpha values = 1296 configs per part (degree × lowpass).
# threshold=0.1 isolates alpha's effect on the LS conditioning step.
# Merge after all 6 tasks complete:
#   merge_raw_grid.py --input-dir outputs/pysindy/raw_grid_alpha_sweep/parts
#                     --output-csv outputs/pysindy/raw_grid_alpha_sweep/raw_grid_merged.csv
#                     --expected 1296
# (216 per part × 6 parts)

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_alpha_sweep}"
STLSQ_THRESHOLD="${STLSQ_THRESHOLD:-0.1}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# Six tasks: three degrees (1, 2, 3) × two low-pass cutoffs (35, 80 Hz).
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
  --threshold "$STLSQ_THRESHOLD" \
  --normalize-columns \
  --optimizer stlsq \
  --alpha-list "0.0,0.01,0.05,0.1,0.5,1.0" \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
