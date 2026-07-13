#!/bin/bash
#SBATCH --job-name=lfp_raw_grid_deg9
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-raw-grid-deg9-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-raw-grid-deg9-%A_%a.err

# Two tasks: degree 9 × two low-pass cutoffs (35, 80 Hz).
# n_delays capped at 6: deg9 n_delays=8 requires ~700GB (24310 features).
#
# Usage (run for all four thresholds):
#   STLSQ_THRESHOLD=1000  OUTPUT_DIR=outputs/pysindy/raw_grid_deg9_t1000  sbatch scripts/hpc/run_raw_grid_deg9_array.slurm
#   STLSQ_THRESHOLD=10000 OUTPUT_DIR=outputs/pysindy/raw_grid_deg9_t10000 sbatch scripts/hpc/run_raw_grid_deg9_array.slurm
#   STLSQ_THRESHOLD=20000 OUTPUT_DIR=outputs/pysindy/raw_grid_deg9_t20000 sbatch scripts/hpc/run_raw_grid_deg9_array.slurm
#   STLSQ_THRESHOLD=50000 OUTPUT_DIR=outputs/pysindy/raw_grid_deg9_t50000 sbatch scripts/hpc/run_raw_grid_deg9_array.slurm

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_deg9_t1000}"
STLSQ_THRESHOLD="${STLSQ_THRESHOLD:-1000}"
NORMALIZE_COLUMNS="${NORMALIZE_COLUMNS:-true}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

LOWPASSES=(35 80)
LOWPASS="${LOWPASSES[$SLURM_ARRAY_TASK_ID]}"
PART_NAME="part_lp${LOWPASS}_degree9"

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
  --degree-list 9 \
  --n-delays-list 2,4,6 \
  --delay-list 1,2,5 \
  --smooth-window-list 0,5,9 \
  --threshold "$STLSQ_THRESHOLD" \
  "${EXTRA_ARGS[@]}" \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
