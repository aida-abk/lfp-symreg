#!/bin/bash
#SBATCH --job-name=lfp_learning_curve
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --mem=32G
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
# One array task per archived configuration (the four slide-deck cases):
#   task 0 -> nd2   task 1 -> nd4   task 2 -> nd6   task 3 -> nd8
#
# Each task fits 31 models (6 sizes x 5 random subsets, plus the single
# full-pool fit at n=28) and simulates all 9 held-out trials per fit.
#
# OMP_NUM_THREADS=1 is correct here: parallelism comes from the array, and
# letting each task grab all cores would oversubscribe the node. This differs
# from a future autoencoder job, which is a single threaded process and should
# keep its threads.
#
# Collect the four per-case CSVs after all tasks finish:
#   cat outputs/pysindy/learning_curve/parts/nd*/learning_curve.csv \
#     | awk 'NR==1 || !/^case,/' \
#     > outputs/pysindy/learning_curve/learning_curve_all.csv

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pysindy/learning_curve/parts}"
N_TRAIN_LIST="${N_TRAIN_LIST:-4,8,12,16,20,24,28}"
REPLICATES="${REPLICATES:-5}"
SEED="${SEED:-0}"

CASES=(nd2 nd4 nd6 nd8)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$OUTPUT_ROOT/$CASE"
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" -u scripts/pysindy/learning_curve.py \
  --case "$CASE" \
  --n-train-list "$N_TRAIN_LIST" \
  --replicates "$REPLICATES" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
