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
#   export GRID_CSV="outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/raw_grid_merged.csv"
#   export CONFIGURATION_INDICES="46,47,49,50,55,56,58,59,82,83,85,86,118,119,121,122"
#   export ALPHA_OVERRIDES="0.05,0.0"
#   export MAX_ITER_OVERRIDES="20,150"
#   export SIGNAL_NORMALIZATION=global_zscore
#   export NORMALIZE_COLUMNS=false
#   sbatch --array=1-16 scripts/hpc/run_sim_selected_configurations.slurm
#
# (values containing commas, like CONFIGURATION_INDICES, must be exported as
# real shell variables first -- sbatch's --export flag is itself comma
# separated and will silently mis-split any value containing a comma)
#
# Each task writes to <grid-csv's parent>/configuration_probes/parts/ and
# .../configuration_probes/plots/ by default. IMPORTANT: filenames are keyed
# only by configuration_index (cfgNN_*), not by normalization scheme -- if you
# run more than one scheme against the same --grid-csv (e.g. z-scored vs not),
# set OUTPUT_DIR explicitly to a distinct directory per scheme, or a later run
# will silently overwrite an earlier run's parts/plots for matching configs:
#
#   export OUTPUT_DIR="outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/configuration_probes_zscored"
#
# Merge the parts afterward, e.g.:
#   python - <<'PY'
#   import pandas as pd, glob
#   base = "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/configuration_probes_zscored"
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
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-60}"
HORIZON_S="${HORIZON_S:-2.0}"
FULL_TRIAL_HORIZON="${FULL_TRIAL_HORIZON:-}"

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
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ "$FULL_TRIAL_HORIZON" == "true" ]]; then
  EXTRA_ARGS+=(--full-trial-horizon)
fi

"$PYTHON_BIN" scripts/pysindy/simulate_selected_configurations.py \
  --configuration-indices "$CONFIGURATION_INDICES" \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --trial-timeout-s "$TRIAL_TIMEOUT_S" \
  --horizon-s "$HORIZON_S" \
  "${EXTRA_ARGS[@]}"
