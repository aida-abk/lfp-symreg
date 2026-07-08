#!/bin/bash
#SBATCH --job-name=lfp_raw_grid_deg7
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=2-3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-raw-grid-deg7-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-raw-grid-deg7-%A_%a.err

# Resubmission of degree-7 tasks only (array tasks 2-3 of the original deg57 job).
# degree=5 parts are complete; this overwrites only the degree=7 part CSVs.
# mem raised to 32G for n_delays=8 (6435 features; Gram matrix ~330 MB).

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_deg57_t100}"
STLSQ_THRESHOLD="${STLSQ_THRESHOLD:-100}"
NORMALIZE_COLUMNS="${NORMALIZE_COLUMNS:-true}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# Task 2 → degree=7, lowpass=35 Hz
# Task 3 → degree=7, lowpass=80 Hz
DEGREES=(5 7)
LOWPASSES=(35 80)
DEGREE_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
LOWPASS_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
DEGREE="${DEGREES[$DEGREE_INDEX]}"
LOWPASS="${LOWPASSES[$LOWPASS_INDEX]}"
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
  --n-delays-list 2,4,6,8 \
  --delay-list 1,2,5 \
  --smooth-window-list 0,5,9 \
  --threshold "$STLSQ_THRESHOLD" \
  "${EXTRA_ARGS[@]}" \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
