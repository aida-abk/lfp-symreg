#!/bin/bash
#SBATCH --job-name=lfp_fourier_grid
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-fourier-grid-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-fourier-grid-%A_%a.err

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysindy/raw_grid_fourier_t1000}"
STLSQ_THRESHOLD="${STLSQ_THRESHOLD:-1000}"
NORMALIZE_COLUMNS="${NORMALIZE_COLUMNS:-true}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# Six tasks: three n_frequencies values crossed with two low-pass cutoffs.
N_FREQS=(1 2 3)
LOWPASSES=(35 80)
NFREQ_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
LOWPASS_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
N_FREQ="${N_FREQS[$NFREQ_INDEX]}"
LOWPASS="${LOWPASSES[$LOWPASS_INDEX]}"
PART_NAME="part_lp${LOWPASS}_nfreq${N_FREQ}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXTRA_ARGS=()
if [[ "$NORMALIZE_COLUMNS" == "true" ]]; then
  EXTRA_ARGS+=(--normalize-columns)
elif [[ "$NORMALIZE_COLUMNS" == "false" ]]; then
  EXTRA_ARGS+=(--no-normalize-columns)
else
  echo "NORMALIZE_COLUMNS must be true or false, got: $NORMALIZE_COLUMNS" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/pysindy/raw_grid_fourier_sweep.py \
  --trial-type fixation \
  --channel 0 \
  --test-fraction 0.25 \
  --seed 0 \
  --downsample 2 \
  --lowpass-list "$LOWPASS" \
  --n-frequencies-list "$N_FREQ" \
  --n-delays-list 2,4,6,8 \
  --delay-list 1,2,5 \
  --smooth-window-list 0,5,9 \
  --threshold "$STLSQ_THRESHOLD" \
  --global-zscore \
  "${EXTRA_ARGS[@]}" \
  --out-csv "$PARTS_DIR/$PART_NAME.csv" \
  --equations-out "$PARTS_DIR/${PART_NAME}_equations.txt" \
  --metadata-out "$PARTS_DIR/${PART_NAME}_metadata.json"
