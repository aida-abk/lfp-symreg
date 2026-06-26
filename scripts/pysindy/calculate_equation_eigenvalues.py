from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "pysindy" / "threshold0_equations"


def eigenvalue_rows(dataset: str, matrix_path: Path) -> list[dict[str, object]]:
  saved = np.load(matrix_path)
  system_matrix = saved["system_matrix"]
  eigenvalues = np.linalg.eigvals(system_matrix)
  eigenvalues = sorted(eigenvalues, key=lambda value: (value.real, value.imag), reverse=True)

  rows = []
  for index, value in enumerate(eigenvalues):
    rows.append(
      {
        "dataset": dataset,
        "eigenvalue_index": index,
        "real_part": float(value.real),
        "imaginary_part": float(value.imag),
        "magnitude": float(abs(value)),
        "stable_real_part_negative": bool(value.real < 0),
        "oscillation_frequency_hz": float(abs(value.imag) / (2 * np.pi)),
        "matrix_path": str(matrix_path),
      }
    )
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Calculate eigenvalues of saved fixation and all-trial linear SINDy systems."
  )
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument(
    "--out-csv",
    type=Path,
    default=DEFAULT_MODEL_DIR / "eigenvalue_comparison.csv",
  )
  args = parser.parse_args()

  rows = []
  rows.extend(
    eigenvalue_rows(
      "fixation",
      args.model_dir / "fixation_linear_system.npz",
    )
  )
  rows.extend(
    eigenvalue_rows(
      "all_trials",
      args.model_dir / "all_trials_linear_system.npz",
    )
  )

  args.out_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  for dataset in ("fixation", "all_trials"):
    print(f"\n{dataset} eigenvalues:")
    for row in rows:
      if row["dataset"] != dataset:
        continue
      print(
        f"  lambda{row['eigenvalue_index']} = "
        f"{row['real_part']:.6f} {row['imaginary_part']:+.6f}j, "
        f"|lambda|={row['magnitude']:.6f}"
      )
  print(f"\nsaved: {args.out_csv}")


if __name__ == "__main__":
  main()
