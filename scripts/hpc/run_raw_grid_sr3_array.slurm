#!/bin/bash
#SBATCH --job-name=lfp_sr3_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-sr3-sweep-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-sr3-sweep-%A_%a.err

# SR3 optimizer sweep: nu in {0.01, 0.1, 1.0, 10.0}.
# 216 base configs × 4 nu values = 864 configs per part.
# threshold=0.1, thresholder=l0 (hard threshold, analogous to STLSQ).
# Merge after all 6 tasks complete:
#   merge_raw_grid.py --input-dir outputs/pysindy/raw_grid_sr3_sweep/parts
#                     --output-csv outputs/pysindy/raw_grid_sr3_sweep/raw_grid_merged.csv
#                     --expected 864
# (144 per part × 6 parts)

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_sr3_sweep}"
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
  --optimizer sr3 \
  --nu-list "0.01,0.1,1.0,10.0" \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
