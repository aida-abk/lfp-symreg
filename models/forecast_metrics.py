"""Lead-time-resolved forecast skill, in numpy only.

Scoring a free-running simulation with one correlation over a long window
cannot distinguish a model that is accurate briefly and then diverges from one
that never worked. These functions resolve skill against lead time instead:
for each lead, the correlation is taken *across* forecasts rather than across
time, which is the standard anomaly-correlation construction.

Persistence -- correlating the measured signal at a lead against its own value
at the forecast origin -- is the reference. It is the signal's autocorrelation
evaluated on exactly the same origins, so it states how far ahead the signal
remains self-predictable at all, and therefore what any model is competing
against.

This module deliberately depends on numpy alone. It is imported both by the
PySINDy scripts and by the TensorFlow runner for the reference deep delay
autoencoder, whose environment pins ``numpy<2`` and cannot install PySINDy.
"""
from __future__ import annotations

import numpy as np


def skill_by_lead(predicted: np.ndarray, measured: np.ndarray) -> np.ndarray:
  """Correlate prediction against truth separately at each lead time.

  Args:
    predicted: Predictions with shape ``(n_forecasts, n_leads)``.
    measured: Matching measurements with the same shape.

  Returns:
    Correlation at each lead time, with ``nan`` where it is undefined.
  """
  n_leads = predicted.shape[1]
  skill = np.full(n_leads, np.nan)
  for lead in range(n_leads):
    a, b = predicted[:, lead], measured[:, lead]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
      continue
    skill[lead] = float(np.corrcoef(a, b)[0, 1])
  return skill


def persistence_by_lead(measured: np.ndarray) -> np.ndarray:
  """Correlate the measured signal at each lead against its own origin value.

  Args:
    measured: Measurements with shape ``(n_forecasts, n_leads)``.

  Returns:
    Persistence correlation at each lead time.
  """
  origin_values = measured[:, 0]
  n_leads = measured.shape[1]
  skill = np.full(n_leads, np.nan)
  for lead in range(n_leads):
    b = measured[:, lead]
    if np.std(origin_values) < 1e-12 or np.std(b) < 1e-12:
      continue
    skill[lead] = float(np.corrcoef(origin_values, b)[0, 1])
  return skill
