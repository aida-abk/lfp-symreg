#!/bin/bash
#SBATCH --job-name=lfp_sim_selected
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-sim-selected-%A_%a.out
#SBATCH --error=slurm/slurm-sim-selected-%A_%a.err

# One array task per configuration_index in CONFIGURATION_INDICES. The
# --array range above is a placeholder (1-1); override it at submit time to
# match how many configurations you listed, e.g. for 3 configurations:
#
#   sbatch --array=1-3 \
#     --export=ALL,CONFIGURATION_INDICES="29,64,173",ALPHA_OVERRIDES="0.05,0.0",MAX_ITER_OVERRIDES="20,150" \
#     scripts/hpc/run_sim_selected_configurations.slurm
#
# To test a different preprocessing/STLSQ scheme than a sweep was originally
# fit under (e.g. global z-scoring with normalize_columns=False, regardless of
# what the source grid-csv's own metadata says), set SIGNAL_NORMALIZATION and
# NORMALIZE_COLUMNS explicitly:
#
#   sbatch --array=1-16 \
#     --export=ALL,GRID_CSV="outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/raw_grid_merged.csv",\
#CONFIGURATION_INDICES="46,47,49,50,55,56,58,59,82,83,85,86,118,119,121,122",\
#ALPHA_OVERRIDES="0.05,0.0",MAX_ITER_OVERRIDES="20,150",\
#SIGNAL_NORMALIZATION=global_zscore,NORMALIZE_COLUMNS=false \
#     scripts/hpc/run_sim_selected_configurations.slurm
#
# Each task writes to <grid-csv's parent>/configuration_probes/parts/ and
# .../configuration_probes/plots/. Merge the parts afterward, e.g.:
#   python - <<'PY'
#   import pandas as pd, glob
#   base = "outputs/pysindy/raw_grid_nc_false_gsz_t1/configuration_probes"
#   for kind in ("trials", "summary"):
#     pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{base}/parts/*_{kind}.csv"))]
#     ).to_csv(f"{base}/simulation_{kind}.csv", index=False)
#   PY

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
CONFIGURATION_INDICES="${CONFIGURATION_INDICES:?Set CONFIGURATION_INDICES, e.g. \"29,64,173\"}"
GRID_CSV="${GRID_CSV:-}"
ALPHA_OVERRIDES="${ALPHA_OVERRIDES:-}"
THRESHOLD_OVERRIDES="${THRESHOLD_OVERRIDES:-}"
MAX_ITER_OVERRIDES="${MAX_ITER_OVERRIDES:-}"
SIGNAL_NORMALIZATION="${SIGNAL_NORMALIZATION:-}"
NORMALIZE_COLUMNS="${NORMALIZE_COLUMNS:-}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-60}"
HORIZON_S="${HORIZON_S:-2.0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXTRA_ARGS=()
if [[ -n "$GRID_CSV" ]]; then
  EXTRA_ARGS+=(--grid-csv "$GRID_CSV")
fi
if [[ -n "$ALPHA_OVERRIDES" ]]; then
  EXTRA_ARGS+=(--alpha-overrides "$ALPHA_OVERRIDES")
fi
if [[ -n "$THRESHOLD_OVERRIDES" ]]; then
  EXTRA_ARGS+=(--threshold-overrides "$THRESHOLD_OVERRIDES")
fi
if [[ -n "$MAX_ITER_OVERRIDES" ]]; then
  EXTRA_ARGS+=(--max-iter-overrides "$MAX_ITER_OVERRIDES")
fi
if [[ -n "$SIGNAL_NORMALIZATION" ]]; then
  EXTRA_ARGS+=(--signal-normalization "$SIGNAL_NORMALIZATION")
fi
if [[ -n "$NORMALIZE_COLUMNS" ]]; then
  if [[ "$NORMALIZE_COLUMNS" == "true" ]]; then
    EXTRA_ARGS+=(--normalize-columns)
  else
    EXTRA_ARGS+=(--no-normalize-columns)
  fi
fi

"$PYTHON_BIN" scripts/pysindy/simulate_selected_configurations.py \
  --configuration-indices "$CONFIGURATION_INDICES" \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --trial-timeout-s "$TRIAL_TIMEOUT_S" \
  --horizon-s "$HORIZON_S" \
  "${EXTRA_ARGS[@]}"
