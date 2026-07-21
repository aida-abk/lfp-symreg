#!/bin/bash
#SBATCH --job-name=lfp_sim_alpha0_subthreshold
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-12
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-sim-alpha0-subthreshold-%A_%a.out
#SBATCH --error=slurm/slurm-sim-alpha0-subthreshold-%A_%a.err

# Run after: probe_alpha0_subthreshold.py produces
# outputs/pysindy/nc_false_gsz_t1_alpha0_probe/alpha0_subthreshold_comparison.csv
#
# One array task per probed configuration (12 total). Each task simulates
# both the stored alpha=0.05 model and a freshly refit alpha=0 model on every
# held-out test trial, writing per-task CSVs to
# outputs/pysindy/nc_false_gsz_t1_alpha0_probe/simulation_parts/.
# Concatenate the parts afterward, e.g.:
#   python - <<'PY'
#   import pandas as pd, glob
#   for kind in ("trials", "summary"):
#     pd.concat([pd.read_csv(f) for f in sorted(
#       glob.glob("outputs/pysindy/nc_false_gsz_t1_alpha0_probe/simulation_parts/*_{}.csv".format(kind)))
#     ]).to_csv(f"outputs/pysindy/nc_false_gsz_t1_alpha0_probe/alpha0_subthreshold_simulation_{kind}.csv", index=False)
#   PY

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-60}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" scripts/pysindy/normalization_validation/simulate_alpha0_subthreshold.py \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --trial-timeout-s "$TRIAL_TIMEOUT_S"
