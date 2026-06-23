from __future__ import annotations

from load_data.convert import TrialData


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
  fixation = set(fixation_trials(data))
  return [trial for trial in range(data.n_trials) if trial not in fixation]


def limit_trials(trials: list[int], max_trials: int | None) -> list[int]:
  if max_trials is None:
    return trials
  return trials[:max_trials]


