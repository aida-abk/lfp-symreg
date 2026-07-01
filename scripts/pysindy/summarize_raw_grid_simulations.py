from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

# Default simulation outputs
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid" / "simulations"


def merge_status_files(
  status_dir: Path,
  output_csv: Path,
  expected_configurations: int,
) -> list[dict[str, str]]:
  """Merge per-configuration simulation status files.

  Args:
    status_dir: Directory containing one ``config_*.csv`` per equation.
    output_csv: Destination for all trial-level simulation outcomes.
    expected_configurations: Required number of configuration files.

  Returns:
    Merged trial-level status rows sorted by configuration and trial order.
  """
  paths = sorted(status_dir.glob("config_*.csv"))
  if len(paths) != expected_configurations:
    raise ValueError(
      f"Found {len(paths)} status files; expected {expected_configurations}."
    )

  rows = []
  fieldnames = None
  for path in paths:
    with path.open(newline="") as file:
      reader = csv.DictReader(file)
      if fieldnames is None:
        fieldnames = reader.fieldnames
      elif reader.fieldnames != fieldnames:
        raise ValueError(f"CSV header mismatch in {path}.")
      rows.extend(reader)
  if not fieldnames:
    raise ValueError("Simulation status files have no CSV header.")

  rows.sort(key=lambda row: (int(row["configuration_index"]), int(row["test_trial_id"])))
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  return rows


def write_html_index(
  figures_dir: Path,
  output_html: Path,
  expected_configurations: int,
) -> None:
  """Write a clickable thumbnail index for all configuration figures."""
  paths = sorted(figures_dir.glob("config_*.png"))
  if len(paths) != expected_configurations:
    raise ValueError(
      f"Found {len(paths)} figures; expected {expected_configurations}."
    )

  cards = []
  for path in paths:
    relative = path.relative_to(output_html.parent)
    label = path.stem.replace("_", " ").title()
    source = html.escape(relative.as_posix())
    cards.append(
      f'<a class="card" href="{source}">'
      f'<img src="{source}" loading="lazy" alt="{html.escape(label)}">'
      f'<span>{html.escape(label)}</span></a>'
    )
  document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raw-grid simulation figures</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ color: inherit; text-decoration: none; border: 1px solid #dadce0; padding: 8px; }}
    .card img {{ display: block; width: 100%; height: 180px; object-fit: contain; background: #fff; }}
    .card span {{ display: block; margin-top: 8px; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>Measured vs simulated x0: all configurations</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
  output_html.parent.mkdir(parents=True, exist_ok=True)
  output_html.write_text(document)


def main() -> None:
  """Merge Oscar outputs and build the visual index."""
  parser = argparse.ArgumentParser(
    description="Merge raw-grid simulation statuses and index all figures."
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--expected", type=int, default=216)
  args = parser.parse_args()

  status_output = args.output_dir / "simulation_status_merged.csv"
  rows = merge_status_files(
    args.output_dir / "status",
    status_output,
    expected_configurations=args.expected,
  )
  index_output = args.output_dir / "index.html"
  write_html_index(
    args.output_dir / "figures",
    index_output,
    expected_configurations=args.expected,
  )
  successful = sum(row["simulation_status"] == "success" for row in rows)
  print(f"merged trial simulations: {len(rows)}")
  print(f"successful trial simulations: {successful}/{len(rows)}")
  print(f"saved: {status_output}")
  print(f"saved: {index_output}")


if __name__ == "__main__":
  main()
