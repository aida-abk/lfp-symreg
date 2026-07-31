#!/bin/bash
# Set up the reference deep-delay-autoencoder implementation on Oscar.
#
# Creates a TensorFlow environment and clones josephbakarji/deep-delay-autoencoder,
# repairing three things that stop it installing or importing as published:
#
#   1. requirements.txt pins `pickle5`, a backport of pickle protocol 5 for
#      Python < 3.8. It is unnecessary on any supported Python and fails to
#      build on modern ones.
#   2. requirements.txt also pins `tensorflow_macos`, which has no Linux
#      distribution and will abort the whole install on Oscar.
#   3. aesindy/config.py does not exist in the repository. Only
#      config_template.py ships, containing `ROOTPATH='todo'`, while the
#      testcases do `from aesindy.config import ROOTPATH`.
#
# TensorFlow is pinned to 2.15 with Python 3.11 because the repository is
# Keras-2-era code: it subclasses tf.keras.Model with a custom train_step and
# uses GradientTape.batch_jacobian and callbacks that mutate tf.Variable masks.
# Keras 3 became the TensorFlow default at 2.16 and changes those contracts.
# Pinning is the low-risk path; the alternative is TF_USE_LEGACY_KERAS=1 with
# the tf-keras package on a newer TensorFlow.
#
# Usage:
#   bash scripts/aesindy/setup_oscar.sh
#   # then, from the project root:
#   PYTHON_BIN=$HOME/aesindy_env/bin/python sbatch slurm/run_aesindy_lfp.slurm

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/aesindy_env}"
REPO_DIR="${REPO_DIR:-$HOME/deep-delay-autoencoder}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.0s}"

echo "=== loading Python module ${PYTHON_MODULE} ==="
module load "${PYTHON_MODULE}" || {
  echo "Could not load ${PYTHON_MODULE}. Run 'module avail python' and set" >&2
  echo "PYTHON_MODULE to a 3.11 build, then rerun." >&2
  exit 1
}

echo "=== creating ${ENV_DIR} ==="
python -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/pip" install --upgrade pip

echo "=== cloning reference implementation ==="
if [ ! -d "${REPO_DIR}" ]; then
  git clone https://github.com/josephbakarji/deep-delay-autoencoder.git "${REPO_DIR}"
fi

echo "=== repairing requirements.txt ==="
# Keep a copy of what shipped, so the edit is auditable.
cp "${REPO_DIR}/requirements.txt" "${REPO_DIR}/requirements.txt.orig"
grep -v -E '^(pickle5|tensorflow_macos|tensorflow)$' \
  "${REPO_DIR}/requirements.txt.orig" > "${REPO_DIR}/requirements.txt"

echo "=== installing ==="
"${ENV_DIR}/bin/pip" install "tensorflow==2.15.*"
"${ENV_DIR}/bin/pip" install -r "${REPO_DIR}/requirements.txt"
# Needed by this project's data layer, which the runner imports.
"${ENV_DIR}/bin/pip" install h5py hdf5storage
"${ENV_DIR}/bin/pip" install -e "${REPO_DIR}"

echo "=== creating the missing aesindy/config.py ==="
printf "ROOTPATH='%s'\n" "${REPO_DIR}" > "${REPO_DIR}/aesindy/config.py"

echo "=== verifying ==="
"${ENV_DIR}/bin/python" - <<'PY'
import tensorflow as tf
from aesindy.training import TrainModel
from aesindy.solvers import RealData
from aesindy.config import ROOTPATH
print("tensorflow", tf.__version__)
print("keras", tf.keras.__version__)
print("ROOTPATH", ROOTPATH)
print("imports OK")
PY

echo
echo "Done. Submit with:"
echo "  PYTHON_BIN=${ENV_DIR}/bin/python sbatch slurm/run_aesindy_lfp.slurm"
