"""Compare fitting and simulation metrics across multiple STLSQ threshold sweeps.

Usage (from repo root):
    .venv/bin/python scripts/pysindy/sweep_analysis/compare_thresholds.py \
        --sweep-dirs outputs/pysindy/raw_grid \
                     outputs/pysindy/raw_grid_threshold100 \
                     outputs/pysindy/raw_grid_threshold1000 \
        --labels 0.1 100 1000 \
        --output-dir outputs/pysindy/threshold_comparison
"""

from __future__ import annotations

import argparse
import csv
import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

csv.field_size_limit(10 ** 7)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV; return [] gracefully if the file is missing."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── Per-sweep data loading ────────────────────────────────────────────────────

def load_sweep(sweep_dir: Path) -> dict:
    """Load grid + simulation status for one sweep directory."""
    grid_path = sweep_dir / "raw_grid_merged.csv"
    sim_path  = sweep_dir / "simulations" / "simulation_status_merged.csv"
    grid = read_csv(grid_path)
    sim  = read_csv(sim_path)
    grid_by_idx = {r["configuration_index"]: r for r in grid}
    return {"grid": grid, "sim": sim, "grid_by_idx": grid_by_idx}


# ── Metric extraction ─────────────────────────────────────────────────────────

def term_stats(
    grid: list[dict],
) -> dict[tuple[int, float], dict]:
    """Mean nonzero terms and utilization per (degree, lowpass) cell."""
    buckets: dict[tuple, list] = collections.defaultdict(list)
    for r in grid:
        if r["fit_status"] != "success":
            continue
        key = (int(r["degree"]), float(r["lowpass_hz"]))
        buckets[key].append(
            (int(r["nonzero_terms"]), int(r["possible_terms"]))
        )
    out = {}
    for key, pairs in buckets.items():
        used = [p[0] for p in pairs]
        poss = [p[1] for p in pairs]
        out[key] = {
            "mean_nonzero": statistics.mean(used),
            "mean_possible": statistics.mean(poss),
            "mean_utilization": statistics.mean(100 * u / p for u, p in zip(used, poss)),
        }
    return out


def sim_success_rate(
    sim: list[dict],
    grid_by_idx: dict,
) -> dict[tuple[int, float], float]:
    """Trial-level simulation success rate per (degree, lowpass)."""
    buckets: dict[tuple, list] = collections.defaultdict(list)
    for r in sim:
        g = grid_by_idx.get(r["configuration_index"], {})
        key = (int(g.get("degree", 0)), float(g.get("lowpass_hz", 0)))
        buckets[key].append(r["simulation_status"] == "success")
    return {k: 100 * sum(v) / len(v) for k, v in buckets.items()}


def median_rmse(
    sim: list[dict],
    grid_by_idx: dict,
    cap: float = 1000.0,
) -> dict[tuple[int, float], float | None]:
    """Median x0 RMSE (µV) per (degree, lowpass), capped to exclude diverged sims."""
    buckets: dict[tuple, list] = collections.defaultdict(list)
    for r in sim:
        if r.get("simulation_status") != "success":
            continue
        raw = r.get("x0_rmse_uv", "")
        if not raw:
            continue
        v = float(raw)
        if not math.isfinite(v) or v > cap:
            continue
        g = grid_by_idx.get(r["configuration_index"], {})
        key = (int(g.get("degree", 0)), float(g.get("lowpass_hz", 0)))
        buckets[key].append(v)
    return {k: statistics.median(v) for k, v in buckets.items() if v}


def config_allok_rate(sim: list[dict]) -> float:
    """Fraction of configurations where every held-out trial succeeded."""
    by_config: dict[str, list[bool]] = collections.defaultdict(list)
    for r in sim:
        by_config[r["configuration_index"]].append(r["simulation_status"] == "success")
    if not by_config:
        return float("nan")
    n_all = sum(1 for v in by_config.values() if all(v))
    return 100 * n_all / len(by_config)


# ── Plotting ──────────────────────────────────────────────────────────────────

DEGREE_COLORS = {1: "#176b57", 2: "#d97706", 3: "#4f46e5"}
LOWPASS_MARKERS = {35.0: "o", 80.0: "s"}
LOWPASS_LINESTYLES = {35.0: "-", 80.0: "--"}


def plot_term_utilization(
    path: Path,
    labels: list[str],
    data: list[dict],  # one dict per sweep from term_stats()
) -> None:
    """Grouped bar chart: mean nonzero terms vs possible, by threshold and degree."""
    degrees = [1, 2, 3]
    n_thresh = len(labels)
    x = np.arange(len(degrees))
    width = 0.8 / n_thresh

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax_idx, (metric_key, ylabel, title) in enumerate([
        ("mean_nonzero", "Mean nonzero terms", "Terms used (mean across configs)"),
        ("mean_utilization", "Mean utilisation (%)", "Library utilisation (%)"),
    ]):
        ax = axes[ax_idx]
        for i, (label, sweep_data) in enumerate(zip(labels, data)):
            # average over lowpass values for display
            vals = []
            for deg in degrees:
                cell_vals = [
                    v[metric_key]
                    for (d, lp), v in sweep_data.items()
                    if d == deg
                ]
                vals.append(statistics.mean(cell_vals) if cell_vals else float("nan"))
            ax.bar(x + i * width, vals, width, label=f"t={label}", alpha=0.85)
        ax.set_xticks(x + width * (n_thresh - 1) / 2)
        ax.set_xticklabels([f"Degree {d}" for d in degrees])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("STLSQ threshold comparison: model sparsity", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_simulation_success(
    path: Path,
    labels: list[str],
    success_data: list[dict],  # sim_success_rate dicts
) -> None:
    """Line chart: simulation trial success rate by degree, split by lowpass."""
    degrees = [1, 2, 3]
    lowpasses = [35.0, 80.0]
    thresh_colors = plt.cm.tab10(np.linspace(0, 0.5, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, lp in zip(axes, lowpasses):
        for i, (label, sdata) in enumerate(zip(labels, success_data)):
            vals = [sdata.get((deg, lp), float("nan")) for deg in degrees]
            ax.plot(
                degrees, vals,
                marker="o", color=thresh_colors[i],
                linewidth=1.8, label=f"t={label}",
            )
        ax.set_xticks(degrees)
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("Trial success rate (%)")
        ax.set_title(f"Low-pass {lp:.0f} Hz")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("STLSQ threshold comparison: simulation success rate", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rmse_comparison(
    path: Path,
    labels: list[str],
    rmse_data: list[dict],
) -> None:
    """Bar chart: median x0 RMSE by degree and threshold (where available)."""
    degrees = [1, 2, 3]
    lowpasses = [35.0, 80.0]
    n_thresh = len(labels)
    x = np.arange(len(degrees))
    width = 0.8 / n_thresh
    thresh_colors = plt.cm.tab10(np.linspace(0, 0.5, n_thresh))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, lp in zip(axes, lowpasses):
        any_data = False
        for i, (label, rdata) in enumerate(zip(labels, rmse_data)):
            vals = [rdata.get((deg, lp)) for deg in degrees]
            if any(v is not None for v in vals):
                any_data = True
            plot_vals = [v if v is not None else float("nan") for v in vals]
            ax.bar(
                x + i * width, plot_vals, width,
                label=f"t={label}", color=thresh_colors[i], alpha=0.85,
            )
        if not any_data:
            ax.text(0.5, 0.5, "No RMSE data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="grey")
        ax.set_xticks(x + width * (n_thresh - 1) / 2)
        ax.set_xticklabels([f"Degree {d}" for d in degrees])
        ax.set_ylabel("Median x0 RMSE (µV, cap=1000)")
        ax.set_title(f"Low-pass {lp:.0f} Hz")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("STLSQ threshold comparison: median simulation RMSE", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_term_count_change(
    path: Path,
    labels: list[str],
    grids: list[list[dict]],
) -> None:
    """Show how many configs change nonzero term count vs the first (reference) sweep."""
    ref_grid = {r["configuration_index"]: int(r["nonzero_terms"]) for r in grids[0]}
    x_labels = [f"t={labels[0]} → t={l}" for l in labels[1:]]
    counts = []
    for grid in grids[1:]:
        other = {r["configuration_index"]: int(r["nonzero_terms"]) for r in grid}
        changed = sum(1 for k in ref_grid if ref_grid[k] != other.get(k, ref_grid[k]))
        counts.append(changed)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x_labels, counts, color="#4f46e5", alpha=0.8)
    for xi, c in enumerate(counts):
        ax.text(xi, c + 1, str(c), ha="center", fontsize=10)
    total = len(ref_grid)
    ax.axhline(total, linestyle="--", color="grey", linewidth=0.8, label=f"Total configs ({total})")
    ax.set_ylabel("Configs with changed term count")
    ax.set_title(f"Configs where threshold changes nonzero terms\n(reference: t={labels[0]})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fitting and simulation metrics across STLSQ threshold sweeps."
    )
    parser.add_argument(
        "--sweep-dirs", nargs="+", type=Path, required=True,
        help="One or more sweep output directories (e.g. outputs/pysindy/raw_grid).",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, required=True,
        help="Short label for each sweep (same order as --sweep-dirs), e.g. 0.1 100 1000.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "pysindy" / "threshold_comparison",
    )
    args = parser.parse_args()

    if len(args.sweep_dirs) != len(args.labels):
        parser.error("--sweep-dirs and --labels must have the same number of entries.")
    if len(args.sweep_dirs) < 2:
        parser.error("Provide at least two sweep directories to compare.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweeps   = [load_sweep(d) for d in args.sweep_dirs]
    labels   = args.labels
    grids    = [s["grid"] for s in sweeps]
    sims     = [s["sim"] for s in sweeps]
    grid_idxs = [s["grid_by_idx"] for s in sweeps]

    t_stats   = [term_stats(g) for g in grids]
    s_rates   = [sim_success_rate(s, gi) for s, gi in zip(sims, grid_idxs)]
    rmse_vals = [median_rmse(s, gi) for s, gi in zip(sims, grid_idxs)]

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_rows = []
    for label, g, s, gi, ts, sr, rv in zip(
        labels, grids, sims, grid_idxs, t_stats, s_rates, rmse_vals
    ):
        n_success = sum(1 for r in g if r["fit_status"] == "success")
        for deg in (1, 2, 3):
            for lp in (35.0, 80.0):
                key = (deg, lp)
                cell = ts.get(key, {})
                summary_rows.append({
                    "threshold": label,
                    "degree": deg,
                    "lowpass_hz": lp,
                    "fit_success_count": n_success,
                    "mean_nonzero_terms": round(cell.get("mean_nonzero", float("nan")), 2),
                    "mean_possible_terms": round(cell.get("mean_possible", float("nan")), 2),
                    "mean_utilization_pct": round(cell.get("mean_utilization", float("nan")), 2),
                    "sim_trial_success_pct": round(sr.get(key, float("nan")), 2),
                    "config_allok_pct": round(config_allok_rate(s), 2),
                    "median_x0_rmse_uv": round(rv[key], 2) if key in rv else "",
                })

    write_csv(
        args.output_dir / "threshold_comparison_summary.csv",
        summary_rows,
        ["threshold", "degree", "lowpass_hz", "fit_success_count",
         "mean_nonzero_terms", "mean_possible_terms", "mean_utilization_pct",
         "sim_trial_success_pct", "config_allok_pct", "median_x0_rmse_uv"],
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_term_utilization(
        args.output_dir / "threshold_term_utilization.png", labels, t_stats
    )
    plot_simulation_success(
        args.output_dir / "threshold_simulation_success.png", labels, s_rates
    )
    plot_rmse_comparison(
        args.output_dir / "threshold_rmse_comparison.png", labels, rmse_vals
    )
    plot_term_count_change(
        args.output_dir / "threshold_term_count_change.png", labels, grids
    )

    # ── JSON summary ──────────────────────────────────────────────────────────
    json_out = []
    for label, g, s in zip(labels, grids, sims):
        terms_all = [int(r["nonzero_terms"]) for r in g if r["fit_status"] == "success"]
        json_out.append({
            "threshold": label,
            "n_configs": len(g),
            "fit_success": sum(1 for r in g if r["fit_status"] == "success"),
            "mean_nonzero_terms": round(statistics.mean(terms_all), 1) if terms_all else None,
            "sim_trial_success_pct": round(
                100 * sum(1 for r in s if r["simulation_status"] == "success") / len(s), 1
            ) if s else None,
            "config_allok_pct": round(config_allok_rate(s), 1),
        })
    (args.output_dir / "threshold_comparison.json").write_text(
        json.dumps(json_out, indent=2) + "\n"
    )

    print(f"saved: {args.output_dir}/threshold_comparison_summary.csv")
    print(f"saved: {args.output_dir}/threshold_term_utilization.png")
    print(f"saved: {args.output_dir}/threshold_simulation_success.png")
    print(f"saved: {args.output_dir}/threshold_rmse_comparison.png")
    print(f"saved: {args.output_dir}/threshold_term_count_change.png")
    print(f"saved: {args.output_dir}/threshold_comparison.json")


if __name__ == "__main__":
    main()
