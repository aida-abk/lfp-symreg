from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from load_data.convert import TrialData
from load_data.trial_selection import select_valid_trials


def fixation_trials(data: TrialData) -> list[int]:
  """Return fixation-like trials.

  In this dataset, trial_rows gives the behavioral table row indices attached
  to each LFP trial. Fixation trials have one behavioral row, while sequence
  trials have four behavioral rows. This separates the dataset into 50
  fixation-like trials and 1500 sequence trials, matching the expected
  blocknum=0 fixation pattern.
  """
  return [trial for trial in range(data.n_trials) if data.trial_row(trial).size == 1]


def non_fixation_trials(data: TrialData) -> list[int]:
  """Return all trials not classified as fixation-like."""
  fixation = set(fixation_trials(data))
  return [trial for trial in range(data.n_trials) if trial not in fixation]


def good_fixation_trials(table: Mapping[str, np.ndarray]) -> list[int]:
  """Return trial indices marked as both fixation and ``goodFix``."""
  return select_valid_trials(table, trial_type="fixation")


def good_non_fixation_trials(table: Mapping[str, np.ndarray]) -> list[int]:
  """Return sequence trials passing the configured non-fixation validity rule."""
  return select_valid_trials(table, trial_type="non_fixation")


def limit_trials(trials: list[int], max_trials: int | None) -> list[int]:
  """Optionally keep only the first trials, preserving recording order."""
  if max_trials is None:
    return trials
  return trials[:max_trials]
