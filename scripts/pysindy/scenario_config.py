"""Configuration for the z-score / normalize-columns / alpha scenario probe.

This file is the single place to edit the experiment. It defines a fixed
"backbone" (everything held equal across runs) and the six scenarios that vary
only signal z-scoring, STLSQ ``normalize_columns``, and the ridge ``alpha``.
The runner in ``scenario_probe.py`` imports these definitions; the canonical
model code in ``models/sindy.py`` is intentionally left untouched.

The scenarios deliberately never pair ``normalize_columns=False`` with an
unnormalized signal, which is the numerically hopeless corner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Backbone: held identical across every run ------------------------------
CHANNEL = 0
DOWNSAMPLE = 2
LOWPASS_HZ = 35.0
N_DELAYS = 4
DELAY_SAMPLES = 2
SMOOTH_WINDOW = 9
SMOOTHING_POLYORDER = 3
MAX_ITER = 20
DEGREES = [2, 3, 5, 7]

# --- Train/test split reused from the existing deg2357 grid -----------------
# Avoids re-decoding the slow ``bhvTrialTbl``; keeps trial selection identical
# to prior work. The runner reads split.train_trial_ids / test_trial_ids here.
SPLIT_METADATA_GLOB = (
  _PROJECT_ROOT
  / "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts"
)
# Single held-out trial simulated for every run (must be in test_trial_ids).
SIMULATION_TEST_TRIAL = 4

# --- Threshold calibration --------------------------------------------------
# One threshold is calibrated per (signal_normalization, normalize_columns)
# REGIME, not merely per z-score group. This is deliberate: z-scoring shifts the
# coefficient scale by orders of magnitude, AND normalize_columns=True makes
# STLSQ threshold in a normalized-column space that is unit-incompatible with
# normalize_columns=False. A single threshold shared across those spaces prunes
# incomparably (verified empirically). The rule: for each regime, sweep
# thresholds at CALIBRATION_DEGREE (alpha=0.05 baseline) and pick the one whose
# actual refit keeps closest to CALIBRATION_TARGET_TERMS terms. Chosen values
# and the sweep tables are reported at runtime.
CALIBRATION_DEGREE = 3
CALIBRATION_TARGET_TERMS = 12
CALIBRATION_N_CANDIDATES = 30

# --- Divergence flagging (diagnostic only, not a rejection rule) ------------
DIVERGENCE_THRESHOLD_STD = 10.0
DIVERGENCE_PERSISTENCE_S = 0.05
# Cap the simulated horizon so integration stays fast and comparable across
# runs; 3 s at the processed rate is ample to see tracking vs. divergence.
MAX_SIMULATION_HORIZON_S = 3.0
# Hard wall-clock cap per simulation so a divergent fit cannot hang the batch.
SIMULATION_WALL_TIMEOUT_S = 60.0

# --- Output locations -------------------------------------------------------
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/scenario_probe"


@dataclass(frozen=True)
class Scenario:
  """One feature-isolation scenario.

  Attributes:
    name: Short human-readable label used in tables, filenames, and plots.
    signal_normalization: ``"none"`` or ``"global_zscore"``; applied to the
      scalar signal before delay embedding.
    normalize_columns: STLSQ ``normalize_columns`` setting (the "NC" knob).
    alpha: STLSQ ridge regularization strength.
  """

  name: str
  signal_normalization: str
  normalize_columns: bool
  alpha: float

  @property
  def threshold_group(self) -> str:
    """Return the calibration group key ('none' or 'global_zscore')."""
    return self.signal_normalization


# The six scenarios, in the order requested.
SCENARIOS: list[Scenario] = [
  Scenario("no_zscore_nc_a05", "none", True, 0.05),
  Scenario("no_zscore_nc_a00", "none", True, 0.00),
  Scenario("zscore_nc_a05", "global_zscore", True, 0.05),
  Scenario("zscore_nc_a00", "global_zscore", True, 0.00),
  Scenario("zscore_nonc_a05", "global_zscore", False, 0.05),
  Scenario("zscore_nonc_a00", "global_zscore", False, 0.00),
]
