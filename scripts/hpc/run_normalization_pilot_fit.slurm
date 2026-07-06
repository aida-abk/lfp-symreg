#!/bin/bash
#SBATCH --job-name=lfp_norm_pilot
#SBATCH --array=0-19%10
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --output=slurm/slurm-lfp-norm-pilot-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-norm-pilot-%A_%a.err

set -euo pipefail

# Run one explicitly selected equation per array task.
export SLURM_EXPORT_ENV=ALL
cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/normalization_pilot_nonorm}"
SELECTION_CSV="${SELECTION_CSV:-scripts/hpc/normalization_pilot_20.csv}"
NORMALIZE_COLUMNS="${NORMALIZE_COLUMNS:-false}"

NORMALIZATION_ARGUMENT="--normalize-columns"
if [[ "$NORMALIZE_COLUMNS" == "false" ]]; then
  NORMALIZATION_ARGUMENT="--no-normalize-columns"
elif [[ "$NORMALIZE_COLUMNS" != "true" ]]; then
  echo "NORMALIZE_COLUMNS must be true or false, got: $NORMALIZE_COLUMNS" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" scripts/pysindy/fit_selected_grid_configuration.py \
  --selection-csv "$SELECTION_CSV" \
  --selection-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_DIR" \
  --threshold 0.1 \
  "$NORMALIZATION_ARGUMENT"
