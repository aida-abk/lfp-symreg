#!/bin/bash
#SBATCH --job-name=lfp_raw_grid_gsz_ncfalse_t20
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-raw-grid-gsz-ncfalse-t20-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-raw-grid-gsz-ncfalse-t20-%A_%a.err

# Fitting sweep: global z-score, normalize_columns=False, STLSQ threshold=20.
# Threshold chosen from the coefficient-magnitude probe (median |coef| ~8-18);
# 20 sits at the term-count knee (deg 2 ~47% of terms retained, deg 5 ~22%),
# a softer sparsity level than 50.
#
# Grid (18 configurations total):
#   degree        : 3, 5, 7   (one per array task)
#   n_delays      : 4, 6, 8
#   delay_samples : 1, 5
#   smooth_window : 5
#   lowpass       : 35 Hz
#
# One array task per degree so that degree=7, n_delays=8 (the largest library,
# ~6.4k features) gets its own resources. Merge after all three finish:
#   python scripts/pysindy/merge_raw_grid.py \
#     --input-dir outputs/pysindy/raw_grid_gsz_ncfalse_t20/parts \
#     --output-csv outputs/pysindy/raw_grid_gsz_ncfalse_t20/raw_grid_merged.csv \
#     --expected 18

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_gsz_ncfalse_t20}"
STLSQ_THRESHOLD="${STLSQ_THRESHOLD:-20}"
LOWPASS="${LOWPASS:-35}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# One degree per array task.
DEGREES=(3 5 7)
DEGREE="${DEGREES[$SLURM_ARRAY_TASK_ID]}"
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
  --n-delays-list 4,6,8 \
  --delay-list 1,5 \
  --smooth-window-list 5 \
  --threshold "$STLSQ_THRESHOLD" \
  --no-normalize-columns \
  --global-zscore \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
