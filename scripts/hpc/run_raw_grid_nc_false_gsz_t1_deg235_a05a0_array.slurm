#!/bin/bash
#SBATCH --job-name=lfp_ncf_gsz_t1_d235
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-ncf-gsz-t1-d235-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-ncf-gsz-t1-d235-%A_%a.err

# Small sweep: normalize_columns=False, global z-score, threshold=1.
# Three array tasks, one polynomial degree each (2, 3, 5), single low-pass 35 Hz.
# Per task: n_delays {4} x delay {5} x smooth {5} x alpha {0.05, 0.0} = 2 configs.
# Total across the array: 3 degrees x 2 = 6 configs.
# With the z-scored signal, state variables are O(1) so coefficients are naturally
# O(1); threshold=1 is the principled cutoff for unnormalized columns.
#
# Merge after all three tasks complete:
#   /oscar/data/tdesroch/dbuyukya/symreg_env/bin/python scripts/pysindy/merge_raw_grid.py \
#     --input-dir outputs/pysindy/raw_grid_nc_false_gsz_t1_deg235_a05a0/parts \
#     --output-csv outputs/pysindy/raw_grid_nc_false_gsz_t1_deg235_a05a0/raw_grid_merged.csv \
#     --expected 6
#
# Example submit:
#   sbatch scripts/hpc/run_raw_grid_nc_false_gsz_t1_deg235_a05a0_array.slurm

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_nc_false_gsz_t1_deg235_a05a0}"
STLSQ_THRESHOLD="${STLSQ_THRESHOLD:-1}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# One degree per task; single low-pass cutoff.
DEGREES=(2 3 5)
LOWPASS=35
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
  --n-delays-list 4 \
  --delay-list 5 \
  --smooth-window-list 5 \
  --threshold "$STLSQ_THRESHOLD" \
  --no-normalize-columns \
  --global-zscore \
  --optimizer stlsq \
  --alpha-list "0.05,0.0" \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
