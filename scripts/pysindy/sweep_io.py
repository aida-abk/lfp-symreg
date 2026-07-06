"""Shared data-loading helpers used by both exploration_sweep and raw_grid_sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import argparse

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT,):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from load_data.convert import TrialData, load_bhv_trial_table
from load_data.trial_selection import select_valid_trials, split_trials_random


def parse_optional_float_list(value: str) -> list[float | None]:
  """Parse comma-separated filter cutoffs in hertz, allowing ``none``.

  Args:
    value: Comma-separated string of floats or the literal ``none``.

  Returns:
    Parsed values with ``None`` representing no filtering.
  """
  def _parse_one(part: str) -> float | None:
    part = part.strip().lower()
    return None if part in {"none", "null"} else float(part)

  return [_parse_one(part) for part in value.split(",") if part.strip()]


def prepare_lfp_trials(
  args: argparse.Namespace,
) -> tuple[TrialData, list[int], list[int]]:
  """Load valid trial identifiers and make one reproducible whole-trial split.

  Args:
    args: Parsed CLI namespace with ``mat_file``, ``trial_type``,
      ``max_trials``, ``test_fraction``, and ``seed`` attributes.

  Returns:
    Loaded trial data, training trial IDs, and test trial IDs.
  """
  data = TrialData.load(args.mat_file)
  table = load_bhv_trial_table(args.mat_file)
  trials = select_valid_trials(table, args.trial_type)
  if args.max_trials is not None:
    trials = trials[: args.max_trials]
  train_ids, test_ids = split_trials_random(
    trials,
    test_fraction=args.test_fraction,
    seed=args.seed,
  )
  return data, train_ids, test_ids
