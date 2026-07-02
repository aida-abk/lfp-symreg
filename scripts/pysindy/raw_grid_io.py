from __future__ import annotations

import csv
from pathlib import Path


def load_successful_grid(path: Path) -> list[dict[str, str]]:
  """Load successful stored-equation rows from a raw-grid CSV.

  Args:
    path: Raw-grid CSV containing a ``fit_status`` column.

  Returns:
    Grid rows in their stored order.
  """
  with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
  if not rows:
    raise ValueError(f"No configurations found in {path}.")
  failed = [row for row in rows if row["fit_status"] != "success"]
  if failed:
    raise ValueError(f"The grid contains {len(failed)} unsuccessful fits.")
  return rows
