"""Measure whether delay-embedded SINDy is data-limited at 37 fixation trials.

The question this answers is narrow and worth stating precisely: holding the
held-out set fixed, does held-out simulation quality still improve as the
number of *training* trials grows? If the curve has flattened by 28 training
trials, the pipeline has extracted what this data has to offer and collecting
more trials would not help it. If the curve is still climbing, the pipeline is
data-limited and more trials would.

Interpretation limit -- this measures delay-embedded SINDy, not an
autoencoder. The two directions are not symmetric:

* Still climbing at 28 trials: decisive. A model with this few free parameters
  is already starved, so a deep autoencoder (orders of magnitude more
  parameters) would be starved too. Collect more data before building one.
* Flat at 28 trials: suggestive only. It shows the data has hit its
  information ceiling *for this model class*. An autoencoder has far more
  capacity and could still be data-limited where SINDy is not.

Method: the archived train/test split is reused so results stay comparable to
the sweep outputs. The 9 held-out trials never change. Training subsets of
size n are drawn at random from the 28-trial training pool, several
replicates per n, so that the spread across subsets can be separated from the
trend across n.

Run with:

    .venv/bin/python scripts/pysindy/learning_curve.py --case nd4

"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from load_data.convert import MAT_FILE, TrialData  # noqa: E402
from load_data.preprocessing import channel_traces  # noqa: E402
from models.sindy import delay_embed_trace, delay_embed_trajectories  # noqa: E402
from models.validation import (  # noqa: E402
  SimulationConfig,
  evaluate_simulation,
  simulate_model_detailed,
)

# Case definitions, the archived split, and the fitting routine are reused
# rather than restated so that a learning-curve point at n=28 reproduces the
# corresponding unbias=True row of the unbias comparison exactly.
from unbias_comparison import (  # noqa: E402
  CASES,
  CHANNEL,
  DOWNSAMPLE,
  MAX_HORIZON_S,
  SIM_WALL_TIMEOUT_S,
  Case,
  fit_case,
  load_split,
)

DEFAULT_N_TRAIN = (4, 8, 12, 16, 20, 24, 28)
DEFAULT_REPLICATES = 5
OUTPUT_DIR = _PROJECT_ROOT / "outputs/pysindy/learning_curve"

# Metrics tracked across the curve. The sign convention records, for each,
# whether larger is better -- used only for axis labelling and the printed
# trend direction, never for filtering.
METRICS = {
  "psd_similarity": True,
  "x0_correlation": True,
  "collapse_std_ratio": True,
  "x0_rmse": False,
}


@dataclass(frozen=True)
class CurvePoint:
  """One fit at one training-set size with one random subset.

  Attributes:
    case: Label of the archived configuration being refit.
    n_train: Number of training trials used. Unitless count.
    replicate: Index of the random subset drawn at this size.
    seed: Seed used to draw the subset, recorded for reproducibility.
    train_ids: Trial identifiers actually used for fitting.
  """

  case: str
  n_train: int
  replicate: int
  seed: int
  train_ids: list[int]


def training_subsets(
  train_pool: list[int],
  n_train_list: tuple[int, ...],
  replicates: int,
  base_seed: int,
  case_label: str,
) -> list[CurvePoint]:
  """Enumerate the random training subsets to fit.

  Subsets are drawn without replacement from the training pool. When ``n``
  equals the full pool size there is only one possible subset, so a single
  replicate is emitted regardless of ``replicates``.

  Args:
    train_pool: Trial identifiers available for training.
    n_train_list: Training-set sizes to evaluate. Unitless counts.
    replicates: Random subsets to draw per size.
    base_seed: Seed offset making the whole curve reproducible.
    case_label: Configuration label recorded on each point.

  Returns:
    Curve points in increasing order of training-set size.
  """
  points: list[CurvePoint] = []
  for n_train in n_train_list:
    if n_train > len(train_pool):
      raise ValueError(
        f"n_train={n_train} exceeds the {len(train_pool)}-trial training pool."
      )
    n_replicates = 1 if n_train == len(train_pool) else replicates
    for replicate in range(n_replicates):
      seed = base_seed + 1000 * n_train + replicate
      rng = np.random.default_rng(seed)
      chosen = rng.choice(len(train_pool), size=n_train, replace=False)
      points.append(
        CurvePoint(
          case=case_label,
          n_train=n_train,
          replicate=replicate,
          seed=seed,
          train_ids=[train_pool[i] for i in sorted(chosen)],
        )
      )
  return points


def evaluate_point(
  point: CurvePoint,
  case: Case,
  data: TrialData,
  test_ids: list[int],
  dt: float,
) -> dict:
  """Fit one training subset and score it on the fixed held-out trials.

  Args:
    point: The training subset to fit.
    case: Archived configuration supplying preprocessing and optimizer settings.
    data: Loaded trial data.
    test_ids: Held-out trial identifiers. Identical for every point.
    dt: Processed sample interval in seconds.

  Returns:
    A result row combining the curve point, the fitted term count, and the
    median of each metric over the held-out trials that simulated to the
    requested horizon.
  """
  fs = 1.0 / dt
  train = channel_traces(
    data, channel=CHANNEL, trials=point.train_ids,
    downsample=DOWNSAMPLE, lowpass_hz=case.lowpass, normalize="none",
  )
  test = channel_traces(
    data, channel=CHANNEL, trials=test_ids,
    downsample=DOWNSAMPLE, lowpass_hz=case.lowpass, normalize="none",
  )
  emb_train = delay_embed_trajectories(train, n_delays=case.n_delays, delay=case.delay)
  emb_test = [
    delay_embed_trace(t, n_delays=case.n_delays, delay=case.delay) for t in test
  ]

  model = fit_case(emb_train, dt, case, unbias=True)
  nonzero_terms = int(np.count_nonzero(np.asarray(model.coefficients())))

  # Which held-out trials integrate to the horizon depends on the fitted
  # model, so the set of completers changes with training-set size. The
  # completed indices are recorded alongside the metrics because a median
  # taken over a shifting subset of trials is not directly comparable across
  # sizes -- see ``completed_trials`` in the output CSV.
  metrics_list = []
  completed_trials: list[int] = []
  for trial_index, measured in enumerate(emb_test):
    horizon = min((len(measured) - 1) * dt, MAX_HORIZON_S)
    sim = simulate_model_detailed(
      model, initial_state=measured[0], dt=dt, horizon_s=horizon,
      wall_timeout_s=SIM_WALL_TIMEOUT_S,
    )
    if sim.completed and sim.trajectory is not None:
      completed_trials.append(trial_index)
      metrics_list.append(
        evaluate_simulation(
          measured, sim.trajectory, fs=fs,
          config=SimulationConfig(simulation_horizon_s=horizon),
        )
      )

  def median_metric(key: str) -> float:
    values = [
      float(m[key]) for m in metrics_list if np.isfinite(float(m[key]))
    ]
    return float(np.median(values)) if values else float("nan")

  row = {
    "case": point.case,
    "n_train": point.n_train,
    "replicate": point.replicate,
    "seed": point.seed,
    "nonzero_terms": nonzero_terms,
    "n_completed": len(metrics_list),
    "n_test": len(emb_test),
    "completed_trials": ";".join(str(i) for i in completed_trials),
    "train_ids": ";".join(str(i) for i in point.train_ids),
  }
  row.update({key: median_metric(key) for key in METRICS})
  return row


def summarize(rows: list[dict], case_label: str) -> list[dict]:
  """Aggregate replicate rows into one summary row per training-set size.

  Args:
    rows: Result rows for a single case.
    case_label: Configuration label used to filter ``rows``.

  Returns:
    One row per training-set size holding the mean and standard deviation of
    each metric across replicates.
  """
  case_rows = [r for r in rows if r["case"] == case_label]
  sizes = sorted({r["n_train"] for r in case_rows})
  summary = []
  for n_train in sizes:
    at_size = [r for r in case_rows if r["n_train"] == n_train]
    entry = {
      "case": case_label,
      "n_train": n_train,
      "replicates": len(at_size),
      "mean_completed": float(np.mean([r["n_completed"] for r in at_size])),
      "mean_terms": float(np.mean([r["nonzero_terms"] for r in at_size])),
    }
    for key in METRICS:
      values = [r[key] for r in at_size if np.isfinite(r[key])]
      entry[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
      entry[f"{key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
    summary.append(entry)
  return summary


def plot_curves(summary: list[dict], case_label: str, output_dir: Path) -> Path:
  """Plot each metric against training-set size with replicate spread.

  Args:
    summary: Summary rows produced by :func:`summarize`.
    case_label: Configuration label, used in the figure title and filename.
    output_dir: Directory to write the figure into.

  Returns:
    Path of the written figure.
  """
  sizes = [s["n_train"] for s in summary]
  fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))
  for ax, (key, higher_is_better) in zip(np.atleast_1d(axes), METRICS.items()):
    means = np.array([s[f"{key}_mean"] for s in summary])
    stds = np.array([s[f"{key}_std"] for s in summary])
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, lw=1.4)
    ax.fill_between(sizes, means - stds, means + stds, alpha=0.15)
    direction = "higher better" if higher_is_better else "lower better"
    ax.set_title(f"{key}\n({direction})", fontsize=9)
    ax.set_xlabel("training trials", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
  fig.suptitle(
    f"Learning curve, case {case_label} -- fixed 9-trial held-out set",
    fontsize=11,
  )
  fig.tight_layout()
  path = output_dir / f"learning_curve_{case_label}.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def print_summary(summary: list[dict]) -> None:
  """Print the curve and the change over the final third of the sizes.

  The reported change is the difference between the largest training-set size
  and the smallest, alongside the replicate spread at the largest size, so
  that a trend can be compared against subset-to-subset noise. No pass/fail
  threshold is applied.

  Args:
    summary: Summary rows produced by :func:`summarize`.
  """
  print(f"\n{'n_train':>8} {'reps':>5} {'done':>6} {'terms':>7}", end="")
  for key in METRICS:
    print(f" {key[:12]:>14}", end="")
  print()
  for entry in summary:
    print(
      f"{entry['n_train']:>8} {entry['replicates']:>5} "
      f"{entry['mean_completed']:>6.1f} {entry['mean_terms']:>7.1f}",
      end="",
    )
    for key in METRICS:
      print(f" {entry[f'{key}_mean']:>+8.3f}±{entry[f'{key}_std']:<5.3f}", end="")
    print()

  print("\nchange from smallest to largest training set:")
  first, last = summary[0], summary[-1]
  for key, higher_is_better in METRICS.items():
    delta = last[f"{key}_mean"] - first[f"{key}_mean"]
    spread = last[f"{key}_std"]
    improved = (delta > 0) == higher_is_better
    verdict = "improves" if improved else "worsens"
    ratio = abs(delta) / spread if spread > 0 else float("inf")
    print(
      f"  {key:>20}: {first[f'{key}_mean']:+.3f} -> {last[f'{key}_mean']:+.3f} "
      f"({verdict} by {abs(delta):.3f}; replicate sd at n={last['n_train']} "
      f"is {spread:.3f}, ratio {ratio:.1f})"
    )


def main() -> None:
  """Fit the learning curve for one or more archived configurations."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--case", default="nd4",
    help="Archived case label to refit, or 'all' for every case.",
  )
  parser.add_argument(
    "--n-train-list", default=",".join(str(n) for n in DEFAULT_N_TRAIN),
    help="Comma-separated training-set sizes.",
  )
  parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()

  n_train_list = tuple(int(n) for n in args.n_train_list.split(","))
  selected = CASES if args.case == "all" else [c for c in CASES if c.label == args.case]
  if not selected:
    raise SystemExit(
      f"Unknown case {args.case!r}; expected one of "
      f"{[c.label for c in CASES]} or 'all'."
    )

  args.out_dir.mkdir(parents=True, exist_ok=True)
  train_pool, test_ids = load_split()
  print(f"Loading {MAT_FILE} ...", flush=True)
  data = TrialData.load(MAT_FILE)
  dt = DOWNSAMPLE / data.fs
  print(
    f"training pool: {len(train_pool)} trials, "
    f"held-out: {len(test_ids)} trials (fixed)"
  )

  rows: list[dict] = []
  for case in selected:
    points = training_subsets(
      train_pool, n_train_list, args.replicates, args.seed, case.label
    )
    print(
      f"\n=== case {case.label} (deg={case.degree}, nd={case.n_delays}, "
      f"delay={case.delay}, thr={case.threshold:g}) -- {len(points)} fits ==="
    )
    for index, point in enumerate(points, start=1):
      row = evaluate_point(point, case, data, test_ids, dt)
      rows.append(row)
      print(
        f"  [{index:>3}/{len(points)}] n_train={point.n_train:>3} "
        f"rep={point.replicate} terms={row['nonzero_terms']:>4} "
        f"done={row['n_completed']}/{row['n_test']} "
        f"psd={row['psd_similarity']:+.3f} "
        f"corr={row['x0_correlation']:+.3f} "
        f"rmse={row['x0_rmse']:.1f}",
        flush=True,
      )

  path = args.out_dir / "learning_curve.csv"
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nwrote {path}")

  for case in selected:
    summary = summarize(rows, case.label)
    print(f"\n===== learning curve, case {case.label} =====")
    print_summary(summary)
    summary_path = args.out_dir / f"learning_curve_summary_{case.label}.csv"
    with open(summary_path, "w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
      writer.writeheader()
      writer.writerows(summary)
    print(f"wrote {summary_path}")
    print(f"wrote {plot_curves(summary, case.label, args.out_dir)}")


if __name__ == "__main__":
  main()
