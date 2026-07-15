#!/bin/bash
#SBATCH --job-name=lfp_sim_deg2357_t20000_winstack
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-288%32
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-sim-deg2357-t20000-winstack-%A_%a.out
#SBATCH --error=slurm/slurm-sim-deg2357-t20000-winstack-%A_%a.err

# Re-anchored windowed simulation of the deg2357 t=20000 grid, rendered as
# vertically-stacked zoomed 2-second window panels (one PNG per trial).
# Each 2 s window is free-run from the MEASURED state at its start; a ~14 s trial
# yields ~7-8 stacked panels sharing a "time from window start" x-axis so that
# recurring features are easy to compare across windows.
#
# Reads the merged grid from the global_analysis folder. Override paths via env
# vars if your layout differs.

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
GRID_CSV="${GRID_CSV:-outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/raw_grid_merged.csv}"
METADATA_JSON="${METADATA_JSON:-outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts/part_lp35_degree2_metadata.json}"
SIMULATION_OUTPUT_DIR="${SIMULATION_OUTPUT_DIR:-outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/simulations_windowed_stacked}"
WINDOW_S="${WINDOW_S:-2}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-120}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" scripts/pysindy/visualize_raw_grid_simulations.py \
  --grid-csv "$GRID_CSV" \
  --metadata-json "$METADATA_JSON" \
  --output-dir "$SIMULATION_OUTPUT_DIR" \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --max-test-trials 4 \
  --window-s "$WINDOW_S" \
  --stacked-windows \
  --per-trial-figures \
  --trial-timeout-s "$TRIAL_TIMEOUT_S"
