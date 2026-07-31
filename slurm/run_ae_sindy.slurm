#!/bin/bash
#SBATCH --job-name=lfp_ae_sindy
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-ae-sindy-%j.out
#SBATCH --error=slurm/slurm-lfp-ae-sindy-%j.err

# Deep delay autoencoder on LFP -- one targeted run, no sweep.
#
# Mirrors testcases/lorenzww_basic.py from the reference repository (the only
# experimental-data example there), with LFP trials in place of waterwheel
# recordings, scored with the same forecast-skill metric used for the PySINDy
# baselines so the numbers sit in one table.
#
# Hyperparameters are the reference's own values for its real-data case:
#   input_dim 80, poly_order 2, max_epochs 300, learning_rate 1e-3,
#   widths_ratios [0.5, 0.25] -> 80 -> 40 -> 20 -> latent, elu
# latent_dim 3 follows the Lorenz-like systems in the paper.
#
# Resource notes, from measurement rather than guesswork:
#   --cpus-per-task=4 and NO OMP_NUM_THREADS=1. This is the opposite of the
#     PySINDy array jobs: those parallelise across array tasks so each task
#     must stay single-threaded, whereas this is one process that benefits
#     from threads.
#   --mem=8G. The Hankel matrix is ~128k x 80 float32 (~41MB), doubled for
#     derivatives; the network itself is tiny. 8G covers the torch runtime
#     with wide margin. The earlier learning-curve job peaked at 1.0GB.
#   --time=2:00:00. Training is ~150k optimizer steps on a small MLP, plus
#     ~240 latent ODE integrations of 1s each in a 3-dimensional space.
#     Expect well under an hour; 2h is headroom, not an estimate.
#
# The array-job walltime failures earlier in this project came from 6s
# integrations of stiff 152-term models. Nothing here does that: the latent
# system is 3-dimensional and the forecasts are 1s.

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

# Point this at an environment that has torch. The shared symreg_env does not,
# so override PYTHON_BIN or create one:
#   python -m venv ~/ae_env && ~/ae_env/bin/pip install torch numpy scipy \
#     matplotlib h5py hdf5storage scikit-learn pysindy
PYTHON_BIN="${PYTHON_BIN:-$HOME/ae_env/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/pysindy/ae_sindy}"

INPUT_DIM="${INPUT_DIM:-80}"
LATENT_DIM="${LATENT_DIM:-3}"
POLY_ORDER="${POLY_ORDER:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
MAX_LEAD="${MAX_LEAD:-1.0}"
ORIGIN_STRIDE="${ORIGIN_STRIDE:-0.5}"
SEED="${SEED:-0}"

mkdir -p "$OUT_DIR"

# Fail immediately with an actionable message rather than dying inside the
# first import several minutes into the allocation.
if ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
  echo "ERROR: torch is not importable from $PYTHON_BIN" >&2
  echo "Create an environment with torch and resubmit, for example:" >&2
  echo "  python -m venv \$HOME/ae_env" >&2
  echo "  \$HOME/ae_env/bin/pip install torch numpy scipy matplotlib h5py \\" >&2
  echo "    hdf5storage scikit-learn pysindy" >&2
  echo "  PYTHON_BIN=\$HOME/ae_env/bin/python sbatch slurm/run_ae_sindy.slurm" >&2
  exit 1
fi

"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__,
'threads', torch.get_num_threads())"

"$PYTHON_BIN" -u scripts/pysindy/ae_sindy_lfp.py \
  --input-dim "$INPUT_DIM" \
  --latent-dim "$LATENT_DIM" \
  --poly-order "$POLY_ORDER" \
  --max-epochs "$MAX_EPOCHS" \
  --max-lead "$MAX_LEAD" \
  --origin-stride "$ORIGIN_STRIDE" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
