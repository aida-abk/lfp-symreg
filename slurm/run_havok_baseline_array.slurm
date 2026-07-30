#!/bin/bash
#SBATCH --job-name=lfp_havok_baseline
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-havok-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-havok-%A_%a.err

# HAVOK baseline: fit SINDy in Hankel-SVD coordinates instead of raw delay
# coordinates, holding channel, preprocessing, trial split, simulation, and
# metrics identical to the learning-curve runs.
#
# One array task per retained-mode count, because cost is dominated by the
# mode count: at 12 modes a degree-3 polynomial library has 455 features and
# each right-hand-side evaluation is correspondingly slow.
#   task 0 -> 3 modes    task 1 -> 5 modes
#   task 2 -> 8 modes    task 3 -> 12 modes  (slowest)
#
# Each task sweeps degree in {1,2,3} and threshold in {0,100,1000,10000},
# i.e. 12 configurations, and simulates all 9 held-out trials per config.
#
# Concatenate the four per-mode CSVs after all tasks finish:
#   cat outputs/pysindy/havok_baseline/parts/modes*/havok_baseline.csv \
#     | awk 'NR==1 || !/^n_modes,/' \
#     > outputs/pysindy/havok_baseline/havok_baseline_all.csv

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pysindy/havok_baseline/parts}"
N_DELAYS="${N_DELAYS:-80}"
DELAY="${DELAY:-1}"
LOWPASS="${LOWPASS:-35}"
DEGREE_LIST="${DEGREE_LIST:-1,2,3}"
THRESHOLD_LIST="${THRESHOLD_LIST:-0,100,1000,10000}"

MODES=(3 5 8 12)
N_MODES="${MODES[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$OUTPUT_ROOT/modes${N_MODES}"
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" -u scripts/pysindy/havok_baseline.py \
  --n-delays "$N_DELAYS" \
  --delay "$DELAY" \
  --lowpass "$LOWPASS" \
  --modes-list "$N_MODES" \
  --degree-list "$DEGREE_LIST" \
  --threshold-list "$THRESHOLD_LIST" \
  --out-dir "$OUT_DIR"
