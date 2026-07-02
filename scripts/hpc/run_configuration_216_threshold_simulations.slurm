#!/bin/bash
#SBATCH --job-name=lfp_c216_thresholds
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-lfp-c216-thresholds-%A_%a.out
#SBATCH --error=slurm-lfp-c216-thresholds-%A_%a.err

set -euo pipefail

# Run from the repository submitted to Slurm.
cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
THRESHOLDS=(0.1 1000 10000 30000 100000)
THRESHOLD="${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# Each task simulates one thresholded equation on all nine held-out trials.
"$PYTHON_BIN" \
  scripts/pysindy/sweep_analysis/simulate_threshold_comparison.py \
  --threshold "$THRESHOLD"
