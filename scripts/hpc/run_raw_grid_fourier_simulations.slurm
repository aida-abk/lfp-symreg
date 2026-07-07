#!/bin/bash
#SBATCH --job-name=lfp_fourier_sim
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-216%32
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --output=slurm/slurm-lfp-fourier-sim-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-fourier-sim-%A_%a.err

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GRID_CSV="${GRID_CSV:-outputs/pysindy/raw_grid_fourier_t1000/raw_grid_merged.csv}"
METADATA_JSON="${METADATA_JSON:-outputs/pysindy/raw_grid_fourier_t1000/parts/part_lp35_nfreq1_metadata.json}"
SIMULATION_OUTPUT_DIR="${SIMULATION_OUTPUT_DIR:-outputs/pysindy/raw_grid_fourier_t1000/simulations}"
TRIAL_TIMEOUT_S="${TRIAL_TIMEOUT_S:-1800}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# visualize_raw_grid_simulations.py auto-detects the Fourier library from the
# CSV header (presence of n_frequencies column) and uses StoredFourierModel.
"$PYTHON_BIN" scripts/pysindy/visualize_raw_grid_simulations.py \
  --grid-csv "$GRID_CSV" \
  --metadata-json "$METADATA_JSON" \
  --output-dir "$SIMULATION_OUTPUT_DIR" \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --trial-timeout-s "$TRIAL_TIMEOUT_S"
