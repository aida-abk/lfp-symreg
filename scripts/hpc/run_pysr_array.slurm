#!/bin/bash
#SBATCH --job-name=lfp_pysr
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-36
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output=slurm/slurm-lfp-pysr-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-pysr-%A_%a.err

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pysr}"
PARTS_DIR="$OUTPUT_DIR/parts"
mkdir -p "$PARTS_DIR"

# Allow PySR's Julia backend to use all allocated CPUs.
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Each array task runs one of the 36 configurations (1-indexed).
"$PYTHON_BIN" scripts/pysr/pysr_sweep.py \
  --configuration-index "$SLURM_ARRAY_TASK_ID" \
  --trial-type fixation \
  --channel 0 \
  --test-fraction 0.25 \
  --seed 0 \
  --downsample 2 \
  --niterations 100 \
  --maxsize 20 \
  --populations 8 \
  --max-train-samples 10000 \
  --output-dir "$OUTPUT_DIR" \
  --out-csv "$PARTS_DIR/config_$(printf '%04d' "$SLURM_ARRAY_TASK_ID").csv"
