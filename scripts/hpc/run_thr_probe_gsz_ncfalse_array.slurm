#!/bin/bash
#SBATCH --job-name=lfp_thr_probe_gsz_ncfalse
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-user=aida_abkenova@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/slurm-lfp-thr-probe-gsz-ncfalse-%A_%a.out
#SBATCH --error=slurm/slurm-lfp-thr-probe-gsz-ncfalse-%A_%a.err

# STLSQ threshold probe on a couple of fixed configurations.
#
# Purpose: with global z-score AND normalize_columns=False, fitted coefficients
# live on the true z-scored data scale (state std ~ 1), so the STLSQ threshold
# compares directly against coefficient magnitudes. This sweep finds where the
# optimal threshold sits before committing to a full deg 2/3/5/7 grid.
#
# One array task per threshold value; each task writes its own output dir so the
# result feeds sweep_analysis/compare_thresholds.py directly (one dir per
# threshold). Each task fits the same two configurations (see DEGREES below):
#
#   Config A: lowpass 35, degree 2, n_delays 4, delay 2, smooth 5
#   Config B: lowpass 35, degree 5, n_delays 4, delay 2, smooth 5
#
# No per-task merge is needed: a single raw_grid_sweep invocation writes both
# configurations into one CSV. After the array finishes, compare across the
# seven per-threshold CSVs, e.g.:
#   python scripts/pysindy/sweep_analysis/compare_thresholds.py \
#     --sweep-dirs outputs/pysindy/thr_probe_gsz_ncfalse/t0.001 ... t0.5 \
#     --labels 0.001 0.005 0.01 0.05 0.1 0.2 0.5 \
#     --output-dir outputs/pysindy/thr_probe_gsz_ncfalse/comparison

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

cd "$SLURM_SUBMIT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/oscar/data/tdesroch/dbuyukya/symreg_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pysindy/thr_probe_gsz_ncfalse}"

# Fixed "couple of configurations". Both degrees share one embedding so the
# only thing that changes between them is polynomial order. Edit these to
# re-pick the probe configs.
LOWPASS="${LOWPASS:-35}"
DEGREES="${DEGREES:-2,5}"
NDELAYS="${NDELAYS:-4}"
DELAY="${DELAY:-2}"
SMOOTH="${SMOOTH:-5}"

# One threshold per array task. Range chosen from the coefficient-magnitude
# probe: with global z-score + normalize_columns=False the fitted coefficients
# are ~10^0-10^2 (median ~8-18, max ~270), so thresholding only bites between
# ~2 and ~100; the term-count knee sits around 10-20.
THRESHOLDS=(1 2 5 10 20 50 100)
THRESHOLD="${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_DIR="$OUTPUT_ROOT/t${THRESHOLD}"
mkdir -p "$OUTPUT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Write straight to raw_grid_merged.csv: a single invocation already contains
# every configuration in the (small) grid, so compare_thresholds.py can read it
# without a merge step.
"$PYTHON_BIN" scripts/pysindy/raw_grid_sweep.py \
  --trial-type fixation \
  --channel 0 \
  --test-fraction 0.25 \
  --seed 0 \
  --downsample 2 \
  --lowpass-list "$LOWPASS" \
  --degree-list "$DEGREES" \
  --n-delays-list "$NDELAYS" \
  --delay-list "$DELAY" \
  --smooth-window-list "$SMOOTH" \
  --threshold "$THRESHOLD" \
  --no-normalize-columns \
  --global-zscore \
  --out-csv "$OUTPUT_DIR/raw_grid_merged.csv" \
  --equations-out "$OUTPUT_DIR/equations.txt" \
  --metadata-out "$OUTPUT_DIR/metadata.json"
