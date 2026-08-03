#!/bin/bash
#SBATCH --job-name=lfp_aesindy_thresh
#SBATCH --partition=batch
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-aesindy-thresh-%A_%a.out
#SBATCH --error=slurm/slurm-aesindy-thresh-%A_%a.err

# Does sparsifying the latent ODE escape the blow-up / collapse dichotomy?
#
# The unmasked poly_order=2 sweep (outputs/aesindy_grid) produced no middle
# ground. latent_dim=4 integrated every trajectory, but only because its
# latent ODE has a globally attracting fixed point: all nine held-out fixation
# free runs converge to -136.87 uV, spread 0.0000 uV across trials, residual
# oscillation std 0.0009 uV against a measured 45 uV. latent_dim 8/12/16
# diverge instead. Nothing sustained an oscillation.
#
# Every one of those runs was fully dense: coefficient_threshold sat at the
# reference default of 1e-6, so the RFE callback never masked anything. This
# array asks whether a sparse quadratic system behaves differently.
#
# Thresholds are calibrated against the final coefficient distribution of the
# unmasked runs, whose median |coefficient| is ~6.8:
#
#     threshold 3  -> ~25% masked      threshold 8  -> ~60% masked
#     threshold 5  -> ~40% masked      threshold 12 -> ~75% masked
#
# Below 3 masks too little to change the result; above 15 leaves too few terms
# to fit with. Treat those percentages as estimates: RFE fires during training,
# when the coefficients differ from their final values, and masking is
# monotone -- once a term is dropped it never returns.
#
# Note this is NOT pysindy's STLSQ. RFE masks and then lets gradient descent
# continue on the survivors; STLSQ masks and re-solves least squares on the
# retained columns. The genuine STLSQ path (--use-sindycall) is unavailable:
# the reference's SindyCall is broken, see scripts/aesindy/run_lfp.py.
#
# Prerequisite, once:
#   bash scripts/aesindy/setup_oscar.sh
#
# Smoke-test the array shape first:
#   SMOKE=1 PYTHON_BIN=$HOME/aesindy_env/bin/python \
#     sbatch slurm/run_aesindy_threshold_array.slurm
#
# Then the real sweep:
#   PYTHON_BIN=$HOME/aesindy_env/bin/python \
#     sbatch slurm/run_aesindy_threshold_array.slurm
#
# Resource notes: --time=12:00:00 because training time proved unpredictable
#   from config alone -- latent_dim=4 trained in 2117 s in one submission and
#   15619 s in another with an identical config and seed, early stopping
#   firing at different epochs. --mem=16G because d4 and d8 peaked around 3 GB;
#   the 21.5 GB seen previously was d12/d16, which this array does not run.

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$HOME/aesindy_env/bin/python}"
SMOKE="${SMOKE:-0}"

# Task layout: latent dimension varies slowest, threshold fastest.
LATENT_DIMS=(4 8)
THRESHOLDS=(3 5 8 12)

N_THRESH=${#THRESHOLDS[@]}
LATENT_DIM="${LATENT_DIMS[$((SLURM_ARRAY_TASK_ID / N_THRESH))]}"
COEFFICIENT_THRESHOLD="${THRESHOLDS[$((SLURM_ARRAY_TASK_ID % N_THRESH))]}"

INPUT_DIM="${INPUT_DIM:-80}"
POLY_ORDER=2
MAX_EPOCHS="${MAX_EPOCHS:-300}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"

# Masking passes at epochs 50, 100, ... 300. The 100-epoch default gives only
# three passes over a 300-epoch run, too few for the network to re-converge
# between them; 50 doubles that while still leaving 50 epochs of recovery.
THRESHOLD_FREQUENCY="${THRESHOLD_FREQUENCY:-50}"

# Off, as in the unmasked sweep. The RK4 integral loss diverges to NaN when the
# latent ODE blows up mid-training, and a NaN in the weights is permanent.
LOSS_WEIGHT_INTEGRAL="${LOSS_WEIGHT_INTEGRAL:-0}"

TRIAL_TYPES="${TRIAL_TYPES:-fixation,non_fixation}"
FIXATION_HOLDOUT="${FIXATION_HOLDOUT:-9}"
SEQUENCE_HOLDOUT="${SEQUENCE_HOLDOUT:-20}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-outputs/aesindy_threshold}/d${LATENT_DIM}_t${COEFFICIENT_THRESHOLD}"
mkdir -p "$OUT_DIR"

export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if ! "$PYTHON_BIN" -c "import tensorflow, aesindy.training, aesindy.config" 2>/dev/null; then
  echo "ERROR: the aesindy environment is not importable from $PYTHON_BIN" >&2
  echo "Run the one-time setup, which also creates the missing" >&2
  echo "aesindy/config.py and strips pickle5 / tensorflow_macos:" >&2
  echo "  bash scripts/aesindy/setup_oscar.sh" >&2
  exit 1
fi

SMOKE_FLAG=""
if [ "$SMOKE" != "0" ]; then
  SMOKE_FLAG="--smoke"
  echo "=== SMOKE RUN: 2 epochs, coarse forecast grid ==="
  echo "=== NOTE: masking fires every $THRESHOLD_FREQUENCY epochs, so a 2-epoch"
  echo "=== smoke run never masks. It checks plumbing only. ==="
fi

echo "=== task $SLURM_ARRAY_TASK_ID: latent_dim=$LATENT_DIM "\
"coefficient_threshold=$COEFFICIENT_THRESHOLD -> $OUT_DIR ==="

"$PYTHON_BIN" -u scripts/aesindy/run_lfp.py \
  --input-dim "$INPUT_DIM" \
  --latent-dim "$LATENT_DIM" \
  --poly-order "$POLY_ORDER" \
  --max-epochs "$MAX_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --loss-weight-integral "$LOSS_WEIGHT_INTEGRAL" \
  --coefficient-threshold "$COEFFICIENT_THRESHOLD" \
  --threshold-frequency "$THRESHOLD_FREQUENCY" \
  --trial-types "$TRIAL_TYPES" \
  --fixation-holdout "$FIXATION_HOLDOUT" \
  --sequence-holdout "$SEQUENCE_HOLDOUT" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  $SMOKE_FLAG
