#!/bin/bash
#SBATCH --job-name=lfp_aesindy
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-aesindy-%j.out
#SBATCH --error=slurm/slurm-lfp-aesindy-%j.err

# Reference deep delay autoencoder (josephbakarji/deep-delay-autoencoder,
# package `aesindy`) applied to LFP -- one targeted run, no sweep.
#
# Prerequisite, once:
#   bash scripts/aesindy/setup_oscar.sh
#
# Then run the plumbing check before spending a full allocation:
#   SMOKE=1 PYTHON_BIN=$HOME/aesindy_env/bin/python \
#     sbatch slurm/run_aesindy_lfp.slurm
#
# Resource notes:
#   --cpus-per-task=4, and deliberately NO OMP_NUM_THREADS=1. This is a single
#     TensorFlow process that benefits from threads, unlike the PySINDy array
#     jobs where parallelism comes from the array and each task must stay
#     single-threaded.
#   --mem=16G. The Hankel matrix is ~128k x 80 float32 (~41MB) doubled for
#     derivatives, and the network is small -- but TensorFlow's own runtime
#     is heavy, so this is more generous than the 4G that suffices for the
#     PySINDy jobs. The learning-curve job peaked at 1.0GB; TF will not.
#   --time=3:00:00. Training is 300 epochs over ~115k Hankel rows on a small
#     MLP, plus ~240 LSODA integrations of a 3-dimensional latent ODE.
#
# CPU-only by design. The network is small enough that a GPU allocation would
# spend longer in the queue than it saves, and would add a CUDA/TF version
# constraint on top of the Keras-2 pin.

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$HOME/aesindy_env/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/aesindy}"

INPUT_DIM="${INPUT_DIM:-80}"
LATENT_DIM="${LATENT_DIM:-3}"
POLY_ORDER="${POLY_ORDER:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
MAX_LEAD="${MAX_LEAD:-1.0}"
ORIGIN_STRIDE="${ORIGIN_STRIDE:-0.5}"
SMOKE="${SMOKE:-0}"

# Sparsification, off by default to match the reference. Set USE_SINDYCALL=1
# to enable the periodic STLSQ refit of the latent coefficients, and lower
# COEFFICIENT_THRESHOLD from 1e-6 to make the RFE callback actually mask.
# Training data. "fixation" is the archived 28-trial set; adding
# "non_fixation" pools in 960 valid sequence trials (~1.9 s each), taking the
# training set to 988 trials and roughly 5x the Hankel rows. The 9-trial
# fixation holdout is unchanged either way, so scores stay comparable; a
# separate sample of sequence trials is also withheld and scored, because
# training dominated by sequence trials would otherwise hide a distribution
# shift behind a fixation-only score.
TRIAL_TYPES="${TRIAL_TYPES:-fixation}"
SEQUENCE_HOLDOUT="${SEQUENCE_HOLDOUT:-20}"

USE_SINDYCALL="${USE_SINDYCALL:-0}"
SINDY_THRESHOLD="${SINDY_THRESHOLD:-0.4}"
SINDYCALL_FREQ="${SINDYCALL_FREQ:-1}"
COEFFICIENT_THRESHOLD="${COEFFICIENT_THRESHOLD:-1e-6}"
THRESHOLD_FREQUENCY="${THRESHOLD_FREQUENCY:-100}"

mkdir -p "$OUT_DIR"

# TensorFlow writes a lot of INFO noise that buries real errors in the log.
export TF_CPP_MIN_LOG_LEVEL=2
# Keep BLAS from oversubscribing the 4 allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Fail immediately, with the fix, rather than dying inside an import after the
# allocation has already started and the .mat file has loaded.
if ! "$PYTHON_BIN" -c "import tensorflow, aesindy.training, aesindy.config" 2>/dev/null; then
  echo "ERROR: the aesindy environment is not importable from $PYTHON_BIN" >&2
  echo "Run the one-time setup, which also creates the missing" >&2
  echo "aesindy/config.py and strips pickle5 / tensorflow_macos:" >&2
  echo "  bash scripts/aesindy/setup_oscar.sh" >&2
  echo "then resubmit with PYTHON_BIN=\$HOME/aesindy_env/bin/python" >&2
  exit 1
fi

# Report versions, but never let a diagnostic abort the run. `set -e` is on,
# and `tf.keras.__version__` does not exist in TF 2.15 -- the lazy loader
# raises AttributeError -- which is enough to kill the job before any work
# starts. Keras is imported directly, and the whole line is made non-fatal.
"$PYTHON_BIN" -c "
import tensorflow as tf
print('tensorflow', tf.__version__)
try:
    import keras
    print('keras', keras.__version__)
except Exception as exc:
    print('keras version unavailable:', exc)
" || echo "version report failed; continuing"

SMOKE_FLAG=""
if [ "$SMOKE" != "0" ]; then
  SMOKE_FLAG="--smoke"
  echo "=== SMOKE RUN: 2 epochs, coarse forecast grid ==="
fi

SINDYCALL_FLAG=""
if [ "$USE_SINDYCALL" != "0" ]; then
  SINDYCALL_FLAG="--use-sindycall"
  echo "=== STLSQ refit ENABLED (threshold ${SINDY_THRESHOLD}) ==="
fi

"$PYTHON_BIN" -u scripts/aesindy/run_lfp.py \
  --input-dim "$INPUT_DIM" \
  --latent-dim "$LATENT_DIM" \
  --poly-order "$POLY_ORDER" \
  --max-epochs "$MAX_EPOCHS" \
  --max-lead "$MAX_LEAD" \
  --origin-stride "$ORIGIN_STRIDE" \
  --trial-types "$TRIAL_TYPES" \
  --sequence-holdout "$SEQUENCE_HOLDOUT" \
  --sindy-threshold "$SINDY_THRESHOLD" \
  --sindycall-freq "$SINDYCALL_FREQ" \
  --coefficient-threshold "$COEFFICIENT_THRESHOLD" \
  --threshold-frequency "$THRESHOLD_FREQUENCY" \
  --out-dir "$OUT_DIR" \
  $SMOKE_FLAG $SINDYCALL_FLAG
