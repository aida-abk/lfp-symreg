from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Project imports
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from load_data.convert import MAT_FILE, TrialData
from load_data.preprocessing import channel_traces, pooled_trace_rms

# Historical raw-grid artifacts
DEFAULT_GRID = ROOT / "outputs" / "pysindy" / "raw_grid" / "raw_grid_merged.csv"
DEFAULT_METADATA = (
  ROOT
  / "outputs"
  / "pysindy"
  / "raw_grid"
  / "parts"
  / "part_lp35_degree1_metadata.json"
)
RMS_FIELD = "training_lfp_rms_uv"


def read_grid(path: Path) -> tuple[list[str], list[dict[str, str]]]:
  """Read the raw grid while preserving its column order."""
  with path.open(newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
  if not rows or not fieldnames:
    raise ValueError(f"No grid data found in {path}.")
  return fieldnames, rows


def cutoff_rms_values(
  rows: list[dict[str, str]],
  metadata: dict[str, object],
  mat_file: Path,
) -> dict[float, float]:
  """Calculate pooled training LFP RMS once per filter cutoff."""
  data = TrialData.load(mat_file)
  if float(metadata["raw_sampling_hz"]) != float(data.fs):
    raise ValueError("MAT sampling frequency does not match the grid metadata.")
  train_ids = [int(value) for value in metadata["split"]["train_trial_ids"]]
  cutoffs = sorted({float(row["lowpass_hz"]) for row in rows})
  return {
    cutoff: pooled_trace_rms(
      channel_traces(
        data,
        channel=int(metadata["channel"]),
        trials=train_ids,
        downsample=int(metadata["downsample_factor"]),
        lowpass_hz=cutoff,
        normalize="none",
      )
    )
    for cutoff in cutoffs
  }


def write_enriched_grid(
  output_path: Path,
  fieldnames: list[str],
  rows: list[dict[str, str]],
  rms_by_cutoff: dict[float, float],
) -> None:
  """Write the grid with training LFP RMS in microvolts."""
  output_fields = [field for field in fieldnames if field != RMS_FIELD]
  lowpass_index = output_fields.index("lowpass_hz")
  output_fields.insert(lowpass_index + 1, RMS_FIELD)
  for row in rows:
    row[RMS_FIELD] = str(rms_by_cutoff[float(row["lowpass_hz"])])

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
  with temporary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(rows)
  temporary_path.replace(output_path)


def main() -> None:
  """Parse CLI arguments and add pooled training LFP RMS to a raw grid."""
  parser = argparse.ArgumentParser(
    description="Add pooled preprocessed training LFP RMS in microvolts to a grid."
  )
  parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID)
  parser.add_argument("--output-csv", type=Path, default=None)
  parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
  parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
  args = parser.parse_args()

  fieldnames, rows = read_grid(args.grid_csv)
  metadata = json.loads(args.metadata_json.read_text())
  rms_by_cutoff = cutoff_rms_values(rows, metadata, args.mat_file)
  output_path = args.output_csv or args.grid_csv
  write_enriched_grid(output_path, fieldnames, rows, rms_by_cutoff)
  for cutoff, rms_uv in rms_by_cutoff.items():
    print(f"lowpass={cutoff:g} Hz: training LFP RMS={rms_uv:.6f} uV")
  print(f"saved: {output_path}")


if __name__ == "__main__":
  main()
