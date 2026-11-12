#!/bin/bash
#SBATCH --job-name=lfp_sim_deg2357_t20000_win2s
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-288%32
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-sim-deg2357-t20000-win2s-pertrial-%A_%a.out
#SBATCH --error=slurm/slurm-sim-deg2357-t20000-win2s-pertrial-%A_%a.err

# Re-anchored 2-second windowed simulation of the already-fitted deg 2,3,5,7,
# threshold=20000 equations. No re-fitting: this reuses raw_grid_merged.csv and
# only simulates. Each ~14 s held-out trial is tiled into consecutive 2 s windows,
# each free-run from the measured delay-embedded state at that window's start.
#
# Prerequisite (already done for t20000): the grid must be merged, e.g.
#   python scripts/pysindy/merge_raw_grid.py \
#     --input-dir outputs/pysindy/raw_grid_deg2357_t20000 --expected 288
# The merged CSV lives at global_analysis/raw_grid_deg2357_t20000/raw_grid_merged.csv.

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
GRID_CSV="${GRID_CSV:-outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/raw_grid_merged.csv}"
METADATA_JSON="${METADATA_JSON:-outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts/part_lp35_degree2_metadata.json}"
SIMULATION_OUTPUT_DIR="${SIMULATION_OUTPUT_DIR:-outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/simulations_pertrial_win2s}"
WINDOW_S="${WINDOW_S:-2.0}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-1800}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" scripts/pysindy/visualize_raw_grid_simulations.py \
  --grid-csv "$GRID_CSV" \
  --metadata-json "$METADATA_JSON" \
  --output-dir "$SIMULATION_OUTPUT_DIR" \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --window-s "$WINDOW_S" \
  --max-test-trials 4 \
  --per-trial-figures \
  --trial-timeout-s "$TRIAL_TIMEOUT_S"
