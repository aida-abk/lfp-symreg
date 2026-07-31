#!/bin/bash
#SBATCH --job-name=lfp_learning_curve
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-27
#SBATCH --time=6:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-learning-curve-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-learning-curve-%A_%a.err

# Learning curve: does held-out simulation quality still improve as the number
# of training trials grows from 4 to 28? The held-out set is the archived
# 9-trial split and never changes; only the training subset varies.
#
# Sizing history -- the first version of this script arrayed over the four
# configurations only (4 tasks) and nd6/nd8 hit TIMEOUT at exactly 12:00:13
# with 100% CPU and 1.0GB of 32GB used. Runtime, not memory, is the binding
# constraint: 31 fits x 9 held-out simulations = 279 integrations per case,
# and at the 240s per-simulation cap the worst case is 18.6 hours. Raising
# --time alone would not reliably fix that, so this version splits the work
# 7x finer and caps the per-simulation cost.
#
# Grid: 4 configurations x 7 training-set sizes = 28 tasks.
#   task = case_index * 7 + size_index
# Each task fits 5 replicates (1 at n=28) and simulates 9 held-out trials,
# i.e. at most 45 integrations -- roughly 1/7th of the previous per-task load.
#
# --sim-timeout 60 replaces the 240s default. A simulation still running after
# 60s is a stiff or diverging model whose metrics would be discarded anyway,
# so the shorter cap bounds the tail without changing the science. It is
# recorded in the CSV via n_completed either way.
#
# Memory is cut 32G -> 4G on the measured 1.0GB peak, which also shortens
# queue time.
#
# Collect all parts after the array finishes:
#   find outputs/pysindy/learning_curve/parts -name learning_curve.csv \
#     | xargs awk 'FNR==1 && NR!=1 {next} {print}' \
#     > outputs/pysindy/learning_curve/learning_curve_all.csv

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pysindy/learning_curve/parts}"
REPLICATES="${REPLICATES:-5}"
SEED="${SEED:-0}"
MAX_HORIZON="${MAX_HORIZON:-6}"
SIM_TIMEOUT="${SIM_TIMEOUT:-60}"

CASES=(nd2 nd4 nd6 nd8)
SIZES=(4 8 12 16 20 24 28)

CASE_INDEX=$(( SLURM_ARRAY_TASK_ID / ${#SIZES[@]} ))
SIZE_INDEX=$(( SLURM_ARRAY_TASK_ID % ${#SIZES[@]} ))
CASE="${CASES[$CASE_INDEX]}"
N_TRAIN="${SIZES[$SIZE_INDEX]}"

# Subsets are seeded as base_seed + 1000*n_train + replicate, so a given
# (case, n_train, replicate) draws the same trials no matter which task
# computes it. Splitting the array this way is reproducible.
OUT_DIR="$OUTPUT_ROOT/$CASE/n${N_TRAIN}"
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "task ${SLURM_ARRAY_TASK_ID}: case=${CASE} n_train=${N_TRAIN}"

"$PYTHON_BIN" -u scripts/pysindy/learning_curve.py \
  --case "$CASE" \
  --n-train-list "$N_TRAIN" \
  --replicates "$REPLICATES" \
  --seed "$SEED" \
  --max-horizon "$MAX_HORIZON" \
  --sim-timeout "$SIM_TIMEOUT" \
  --out-dir "$OUT_DIR"
