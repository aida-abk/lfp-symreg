#!/bin/bash
#SBATCH --job-name=lfp_raw_grid_t10000
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-raw-grid-t10000-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-raw-grid-t10000-%A_%a.err

# Small fitting-only sweep at STLSQ threshold=10000.
# Four array tasks: one polynomial degree each (2, 3, 5, 7), single low-pass 35 Hz.
# Per task: n_delays {4,6} x delay {1,5} x smooth {5} x alpha {0.05, 0.0} = 8 configs.
# Total across the array: 4 degrees x 8 = 32 configs.
# 48G is generous; the heaviest task (degree 7, n_delays 6) needs far less.
#
# Merge after all four tasks complete:
#   /oscar/data/tdesroch/dbuyukya/symreg_env/bin/python scripts/pysindy/merge_raw_grid.py \
#     --input-dir outputs/pysindy/raw_grid_deg2357_thr10000_lp35_sw5/parts \
#     --output-csv outputs/pysindy/raw_grid_deg2357_thr10000_lp35_sw5/raw_grid_merged.csv \
#     --expected 32
#
# Example submit:
#   sbatch scripts/hpc/run_raw_grid_deg2357_thr10000_array.slurm

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_deg2357_thr10000_lp35_sw5}"
STLSQ_THRESHOLD="${STLSQ_THRESHOLD:-10000}"
NORMALIZE_COLUMNS="${NORMALIZE_COLUMNS:-true}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# One degree per task; single low-pass cutoff.
DEGREES=(2 3 5 7)
LOWPASS=35
DEGREE="${DEGREES[$SLURM_ARRAY_TASK_ID]}"
PART_NAME="part_lp${LOWPASS}_degree${DEGREE}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXTRA_ARGS=()
if [[ "$NORMALIZE_COLUMNS" == "true" ]]; then
  EXTRA_ARGS+=(--normalize-columns)
elif [[ "$NORMALIZE_COLUMNS" == "false" ]]; then
  EXTRA_ARGS+=(--no-normalize-columns)
else
  echo "NORMALIZE_COLUMNS must be true or false, got: $NORMALIZE_COLUMNS" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/pysindy/raw_grid_sweep.py \
  --trial-type fixation \
  --channel 0 \
  --test-fraction 0.25 \
  --seed 0 \
  --downsample 2 \
  --lowpass-list "$LOWPASS" \
  --degree-list "$DEGREE" \
  --n-delays-list 4,6 \
  --delay-list 1,5 \
  --smooth-window-list 5 \
  --threshold "$STLSQ_THRESHOLD" \
  "${EXTRA_ARGS[@]}" \
  --optimizer stlsq \
  --alpha-list "0.05,0.0" \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
