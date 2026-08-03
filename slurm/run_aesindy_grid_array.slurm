#!/bin/bash
#SBATCH --job-name=lfp_aesindy_grid
#SBATCH --partition=batch
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-aesindy-grid-%A_%a.out
#SBATCH --error=slurm/slurm-aesindy-grid-%A_%a.err

# Quadratic latent models across latent dimension, at input_dim=80.
#
# The existing runs cover poly_order=1 at latent_dim 8/12/16 and poly_order=2
# at input_dim 40 and 160 with latent_dim=12. This array fills the gap beside
# them: poly_order=2 at latent_dim 4, 8, 12, 16, with input_dim held at 80 so
# latent dimension is the only variable.
#
# Expect some of these not to integrate. A dense quadratic latent ODE with
# ~1000 live coefficients diverges in finite time, which is why the earlier
# quadratic runs produced no usable trajectory. That is a result, and
# scripts/aesindy/run_lfp.py now reports the integrated/attempted counts
# rather than silently dropping the failures.
#
# Prerequisite, once:
#   bash scripts/aesindy/setup_oscar.sh
#
# Smoke-test the whole array shape first (2 epochs each, a few minutes):
#   SMOKE=1 PYTHON_BIN=$HOME/aesindy_env/bin/python \
#     sbatch slurm/run_aesindy_grid_array.slurm
#
# Then the real sweep:
#   PYTHON_BIN=$HOME/aesindy_env/bin/python \
#     sbatch slurm/run_aesindy_grid_array.slurm
#
# Resource notes, relative to the single-run job:
#   --time=8:00:00 and --mem=24G, both raised. Training now pools ~1000 trials
#     instead of 28, roughly 5x the Hankel rows, and the whole-trial free runs
#     add one LSODA integration per held-out trial over its full ~14 s.
#   --cpus-per-task=4 with no OMP_NUM_THREADS=1: a single TensorFlow process
#     benefits from threads, unlike the PySINDy array jobs where parallelism
#     comes from the array itself.

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$HOME/aesindy_env/bin/python}"
SMOKE="${SMOKE:-0}"

LATENT_DIMS=(4 8 12 16)
LATENT_DIM="${LATENT_DIMS[$SLURM_ARRAY_TASK_ID]}"

INPUT_DIM="${INPUT_DIM:-80}"
POLY_ORDER=2
MAX_EPOCHS="${MAX_EPOCHS:-300}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"

# The RK4 integral loss is off by default here. It integrates the latent ODE
# during training, and a quadratic system with randomly initialised
# coefficients of order 10 diverges in finite time -- a single NaN then
# propagates into the weights permanently. Pooling in the sequence trials
# raises the optimizer steps per epoch from 342 to ~1665, which multiplies the
# chances of hitting it. Set LOSS_WEIGHT_INTEGRAL=0.1 to restore the
# reference's value if you want that term back.
LOSS_WEIGHT_INTEGRAL="${LOSS_WEIGHT_INTEGRAL:-0}"

TRIAL_TYPES="${TRIAL_TYPES:-fixation,non_fixation}"
FIXATION_HOLDOUT="${FIXATION_HOLDOUT:-9}"
SEQUENCE_HOLDOUT="${SEQUENCE_HOLDOUT:-20}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-outputs/aesindy_grid}/q${INPUT_DIM}_d${LATENT_DIM}"
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
fi

echo "=== task $SLURM_ARRAY_TASK_ID: poly_order=$POLY_ORDER "\
"latent_dim=$LATENT_DIM input_dim=$INPUT_DIM -> $OUT_DIR ==="

"$PYTHON_BIN" -u scripts/aesindy/run_lfp.py \
  --input-dim "$INPUT_DIM" \
  --latent-dim "$LATENT_DIM" \
  --poly-order "$POLY_ORDER" \
  --max-epochs "$MAX_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --loss-weight-integral "$LOSS_WEIGHT_INTEGRAL" \
  --trial-types "$TRIAL_TYPES" \
  --fixation-holdout "$FIXATION_HOLDOUT" \
  --sequence-holdout "$SEQUENCE_HOLDOUT" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  $SMOKE_FLAG
