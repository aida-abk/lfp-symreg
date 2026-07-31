"""The archived train/test trial split, readable without PySINDy.

Every model comparison in this project is scored on one fixed split, recorded
in the metadata written alongside an earlier raw-grid sweep. Reusing it is
what makes results from different methods directly comparable.

``scripts/pysindy/unbias_comparison.py`` also exposes a ``load_split``, but
importing it pulls in PySINDy at module scope. The TensorFlow environment for
the reference deep delay autoencoder pins ``numpy<2`` (TensorFlow 2.15 is
compiled against the NumPy 1.x ABI) while PySINDy 2.1 requires ``numpy>=2``,
so the two cannot coexist. This module provides the same split with no
dependency beyond the standard library.
"""
from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: LFP channel every comparison is run on.
CHANNEL = 0

#: Keep every Nth sample after filtering, giving 250 Hz from a 500 Hz source.
DOWNSAMPLE = 2

#: Sweep metadata holding the canonical split.
SPLIT_METADATA_DIR = (
  _PROJECT_ROOT / "outputs/pysindy/global_analysis/raw_grid_deg2357_t20000/parts"
)


def load_archived_split(
  metadata_dir: Path = SPLIT_METADATA_DIR,
) -> tuple[list[int], list[int]]:
  """Return the archived ``(train_trial_ids, test_trial_ids)``.

  Every metadata file in the directory records the same split, so the first
  is read.

  Args:
    metadata_dir: Directory holding ``*_metadata.json`` from the sweep.

  Returns:
    Training and held-out trial identifiers.

  Raises:
    FileNotFoundError: If no metadata file is present.
  """
  candidates = sorted(metadata_dir.glob("*_metadata.json"))
  if not candidates:
    raise FileNotFoundError(
      f"No split metadata found in {metadata_dir}. This directory is produced "
      f"by the raw-grid sweep and is required to reproduce the canonical "
      f"28/9 trial split."
    )
  split = json.loads(candidates[0].read_text())["split"]
  return split["train_trial_ids"], split["test_trial_ids"]
