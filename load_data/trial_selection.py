from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TrialValidityConfig:
  """Column rules for selecting valid trials by trial type."""

  trial_type_columns: dict[str, str] = field(
    default_factory=lambda: {
      "fixation": "is_fixation_trial",
      "non_fixation": "is_sequence_trial",
    }
  )
  validity_columns: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "fixation": ("goodFix",),
      "non_fixation": ("goodFix_wholeseq",),
    }
  )


DEFAULT_TRIAL_VALIDITY = TrialValidityConfig()


def boolean_column(table: Mapping[str, np.ndarray], column: str) -> np.ndarray:
  """Return one behavioral table column as a boolean NumPy vector."""
  if column not in table:
    raise KeyError(f"bhvTrialTbl is missing required column: {column}")
  values = np.asarray(table[column]).squeeze()
  return values.astype(bool)


def select_valid_trials(
  table: Mapping[str, np.ndarray],
  trial_type: str,
  config: TrialValidityConfig = DEFAULT_TRIAL_VALIDITY,
) -> list[int]:
  """Select trials of one type that pass the configured validity columns."""
  if trial_type not in config.trial_type_columns:
    raise ValueError(
      f"Unknown trial_type {trial_type!r}; expected one of "
      f"{sorted(config.trial_type_columns)}"
    )
  if trial_type not in config.validity_columns:
    raise ValueError(f"No validity columns configured for trial_type {trial_type!r}")

  mask = boolean_column(table, config.trial_type_columns[trial_type])
  for column in config.validity_columns[trial_type]:
    column_mask = boolean_column(table, column)
    if column_mask.shape != mask.shape:
      raise ValueError(
        f"Column {column!r} has shape {column_mask.shape}, "
        f"expected {mask.shape}."
      )
    mask &= column_mask

  return np.flatnonzero(mask).tolist()
