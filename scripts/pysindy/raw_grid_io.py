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


def write_csv_checkpoint(
  path: Path,
  fieldnames: list[str],
  rows: list[dict[str, object]],
) -> None:
  """Atomically write rows so interrupted jobs retain completed work.

  Args:
    path: Destination CSV path.
    fieldnames: Explicit output column order.
    rows: Rows completed so far.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = path.with_suffix(path.suffix + ".tmp")
  with temporary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  temporary_path.replace(path)
