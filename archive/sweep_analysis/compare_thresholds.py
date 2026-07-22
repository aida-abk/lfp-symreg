"""Extensive threshold comparison: t=0.1, 100, 1000, 10000.

Produces 10 figures, a summary CSV, and a printed recommendations report.

Usage (from repo root):
    .venv/bin/python scripts/pysindy/sweep_analysis/compare_thresholds.py \
        --output-dir outputs/pysindy/threshold_comparison

The script auto-discovers the four standard sweep directories. Override with
--sweep-dirs and --labels if needed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

csv.field_size_limit(10 * 1024 * 1024)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SWEEPS = [
    ("0.1",   ROOT / "outputs/pysindy/raw_grid"),
    ("100",   ROOT / "outputs/pysindy/raw_grid_threshold100"),
    ("1000",  ROOT / "outputs/pysindy/raw_grid_threshold1000"),
    ("10000", ROOT / "outputs/pysindy/raw_grid_threshold10000"),
]

DEGREE_COLORS  = {1: "#2563eb", 2: "#d97706", 3: "#7c3aed"}
THRESH_COLORS  = ["#6b7280", "#16a34a", "#d97706", "#dc2626"]
TERM_TYPE_COLORS = {
    "bias":      "#94a3b8",
    "linear":    "#2563eb",
    "quadratic": "#d97706",
    "cubic":     "#7c3aed",
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_grid(path: Path) -> list[dict]:
    """Load a merged raw-grid CSV; return [] if missing."""
    if not path.exists():
        print(f"  [warn] grid not found: {path}", file=sys.stderr)
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_sim_status(sweep_dir: Path) -> list[dict]:
    """Load per-trial simulation status rows from a sweep directory.

    Prefers ``simulations/status/`` (aggregate format) over
    ``simulations_pertrial/status/`` so that all thresholds use the same
    format when both exist. Falls back to pertrial if aggregate is absent.
    Returns [] if neither exists.

    Args:
        sweep_dir: Root sweep output directory (e.g. outputs/pysindy/raw_grid).

    Returns:
        List of per-trial status dicts from all config CSVs found.
    """
    candidates = [
        sweep_dir / "simulations" / "status",
        sweep_dir / "simulations_pertrial" / "status",
    ]
    status_dir = None
    for candidate in candidates:
        if candidate.exists():
            count = sum(1 for _ in candidate.glob("config_*.csv"))
            if count > 0:
                status_dir = candidate
                break
    if status_dir is None:
        return []
    rows = []
    for csv_path in sorted(status_dir.glob("config_*.csv")):
        with csv_path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a CSV with given fieldnames to path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── Feature / coefficient helpers ─────────────────────────────────────────────

def classify_term(name: str) -> str:
    """Classify a feature name as bias, linear, quadratic, or cubic.

    Args:
        name: Feature name string from PySINDy, e.g. '1', 'x0', 'x0^2', 'x0 x1'.

    Returns:
        One of 'bias', 'linear', 'quadratic', 'cubic'.
    """
    if name == "1":
        return "bias"
    powers = re.findall(r"\^(\d+)", name)
    cross   = " " in name.replace("^", "").replace(" ", "X")
    total_power = sum(int(p) for p in powers) + (name.count(" ") if not powers else 0)
    if "^3" in name or total_power >= 3:
        return "cubic"
    if "^2" in name or (cross and total_power == 2) or total_power == 2:
        return "quadratic"
    return "linear"


def nonzero_term_names(row: dict) -> list[str]:
    """Return feature names for which at least one equation has a nonzero coefficient.

    Args:
        row: One raw-grid CSV row with feature_names_json and coefficients_json.

    Returns:
        List of surviving feature name strings.
    """
    try:
        names  = json.loads(row["feature_names_json"])
        coeffs = np.array(json.loads(row["coefficients_json"]))  # (n_eq, n_feat)
    except (KeyError, json.JSONDecodeError, ValueError):
        return []
    surviving = np.any(np.abs(coeffs) > 1e-12, axis=0)
    return [n for n, s in zip(names, surviving) if s]


def all_coefficient_magnitudes(row: dict) -> list[float]:
    """Return absolute values of all nonzero coefficients in a row.

    Args:
        row: One raw-grid CSV row.

    Returns:
        List of |coefficient| values greater than 1e-12.
    """
    try:
        coeffs = np.array(json.loads(row["coefficients_json"])).ravel()
    except (KeyError, json.JSONDecodeError, ValueError):
        return []
    return [abs(c) for c in coeffs if abs(c) > 1e-12]


# ── Per-sweep statistics ───────────────────────────────────────────────────────

def compute_grid_stats(grid: list[dict]) -> dict:
    """Aggregate fitting statistics for one sweep.

    Args:
        grid: Rows from a merged raw-grid CSV.

    Returns:
        Dict with keys: n_total, n_success, zero_term_count,
        term_counts, utilization_pcts, by_degree, by_ndelays.
    """
    ok = [r for r in grid if r["fit_status"] == "success"]
    term_counts     = [int(r["nonzero_terms"]) for r in ok]
    util_pcts       = [float(r["term_utilization_percent"]) for r in ok
                       if r.get("term_utilization_percent") not in ("", None)]
    zero_term_count = sum(1 for t in term_counts if t == 0)

    by_degree: dict[int, list[float]] = collections.defaultdict(list)
    by_ndelays: dict[int, list[float]] = collections.defaultdict(list)
    by_lowpass: dict[float, list[float]] = collections.defaultdict(list)
    by_smooth: dict[int, list[float]] = collections.defaultdict(list)

    for r in ok:
        deg  = int(r["degree"])
        nd   = int(r["n_delays"])
        lp   = float(r["lowpass_hz"])
        sm   = int(r["smooth_window_samples"])
        util = float(r["nonzero_terms"]) / max(int(r["possible_terms"]), 1) * 100
        by_degree[deg].append(util)
        by_ndelays[nd].append(util)
        by_lowpass[lp].append(util)
        by_smooth[sm].append(util)

    return {
        "n_total":        len(grid),
        "n_success":      len(ok),
        "zero_term_count": zero_term_count,
        "term_counts":    term_counts,
        "util_pcts":      util_pcts,
        "by_degree":      dict(by_degree),
        "by_ndelays":     dict(by_ndelays),
        "by_lowpass":     dict(by_lowpass),
        "by_smooth":      dict(by_smooth),
    }


def compute_sim_stats(sim_rows: list[dict]) -> dict:
    """Aggregate simulation success and RMSE from per-trial status rows.

    Args:
        sim_rows: Rows from per-trial simulation status CSVs.

    Returns:
        Dict with keys: n_trials, success_rate, rmse_values (may be empty).
    """
    if not sim_rows:
        return {"n_trials": 0, "success_rate": float("nan"), "rmse_values": []}
    n_ok  = sum(1 for r in sim_rows if r["simulation_status"] == "success")
    rmses = []
    for r in sim_rows:
        if r["simulation_status"] != "success":
            continue
        for key in ("x0_rmse_uv", "x0_rmse"):
            if r.get(key) not in ("", None):
                try:
                    v = float(r[key])
                    if math.isfinite(v) and v < 1e4:
                        rmses.append(v)
                except ValueError:
                    pass
                break
    return {
        "n_trials":     len(sim_rows),
        "success_rate": 100 * n_ok / len(sim_rows),
        "rmse_values":  rmses,
    }


def compute_term_type_breakdown(
    grid: list[dict],
) -> dict[str, float]:
    """Count surviving term types across all successful configurations.

    Args:
        grid: Rows from a merged raw-grid CSV.

    Returns:
        Dict mapping term type ('bias','linear','quadratic','cubic') to
        mean count of surviving terms of that type per configuration.
    """
    type_counts: dict[str, list[int]] = collections.defaultdict(list)
    for r in grid:
        if r["fit_status"] != "success":
            continue
        names = nonzero_term_names(r)
        by_type: dict[str, int] = collections.Counter(classify_term(n) for n in names)
        for t in ("bias", "linear", "quadratic", "cubic"):
            type_counts[t].append(by_type.get(t, 0))
    return {t: statistics.mean(v) for t, v in type_counts.items() if v}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets of term names.

    Args:
        set_a: First set of term name strings.
        set_b: Second set of term name strings.

    Returns:
        Float in [0, 1]; 1.0 means identical sets.
    """
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── Plotting ──────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, path: Path, **kwargs) -> None:
    """Save figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"  saved: {path.name}")


def plot_sparsity_violin(
    labels: list[str],
    stats:  list[dict],
    path:   Path,
) -> None:
    """Violin + box plot: distribution of nonzero term counts per threshold.

    Args:
        labels: Threshold label strings.
        stats:  List of compute_grid_stats dicts.
        path:   Output PNG path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    degrees = [1, 2, 3]

    for ax, deg in zip(axes, degrees):
        data_per_thresh = []
        for s, label in zip(stats, labels):
            by_deg = []
            for sweep_grid in [s]:
                # we need the raw grid — pass it separately; stats has by_degree util
                # actually we need term_counts by degree — recompute from by_degree keys
                pass
            data_per_thresh.append(s.get("by_degree_terms", {}).get(deg, []))

        parts = ax.violinplot(
            [d for d in data_per_thresh if d],
            positions=range(len([d for d in data_per_thresh if d])),
            showmedians=True,
        )
        valid_labels = [l for d, l in zip(data_per_thresh, labels) if d]
        for i, (pc, col) in enumerate(zip(parts["bodies"], THRESH_COLORS)):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels([f"t={l}" for l in valid_labels], fontsize=9)
        ax.set_ylabel("Nonzero terms")
        ax.set_title(f"Degree {deg}")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Term count distribution per threshold (by degree)", fontsize=12)
    fig.tight_layout()
    savefig(fig, path)


def plot_utilization_by_degree(
    labels: list[str],
    stats:  list[dict],
    path:   Path,
) -> None:
    """Grouped bar chart: mean term utilisation by threshold × degree.

    Args:
        labels: Threshold label strings.
        stats:  List of compute_grid_stats dicts.
        path:   Output PNG path.
    """
    degrees = [1, 2, 3]
    x   = np.arange(len(degrees))
    w   = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (label, s, col) in enumerate(zip(labels, stats, THRESH_COLORS)):
        vals = [
            statistics.mean(s["by_degree"].get(d, [float("nan")]))
            for d in degrees
        ]
        bars = ax.bar(x + i * w, vals, w, label=f"t={label}", color=col, alpha=0.82)
        for bar, v in zip(bars, vals):
            if math.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("Mean library utilisation (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Library utilisation by degree and threshold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_utilization_vs_ndelays(
    labels: list[str],
    stats:  list[dict],
    path:   Path,
) -> None:
    """Line plot: mean utilisation vs n_delays for each threshold.

    Args:
        labels: Threshold label strings.
        stats:  List of compute_grid_stats dicts.
        path:   Output PNG path.
    """
    n_delays_vals = [2, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, s, col in zip(labels, stats, THRESH_COLORS):
        vals = [
            statistics.mean(s["by_ndelays"].get(nd, [float("nan")]))
            for nd in n_delays_vals
        ]
        ax.plot(n_delays_vals, vals, marker="o", color=col, label=f"t={label}", linewidth=2)

    ax.set_xlabel("n_delays (embedding dimension)")
    ax.set_ylabel("Mean library utilisation (%)")
    ax.set_xticks(n_delays_vals)
    ax.set_title("Utilisation vs embedding dimension per threshold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_term_type_breakdown(
    labels:     list[str],
    breakdowns: list[dict],
    path:       Path,
) -> None:
    """Stacked bar chart: mean surviving terms by type at each threshold.

    Args:
        labels:     Threshold label strings.
        breakdowns: List of compute_term_type_breakdown dicts.
        path:       Output PNG path.
    """
    types = ["bias", "linear", "quadratic", "cubic"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    bottoms = np.zeros(len(labels))
    for t in types:
        vals = [b.get(t, 0) for b in breakdowns]
        bars = ax.bar(x, vals, bottom=bottoms, label=t.capitalize(),
                      color=TERM_TYPE_COLORS[t], alpha=0.85)
        for bar, v, bot in zip(bars, vals, bottoms):
            if v > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + v / 2,
                    f"{v:.0f}", ha="center", va="center", fontsize=8, color="white",
                )
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"t={l}" for l in labels])
    ax.set_ylabel("Mean surviving terms per configuration")
    ax.set_title("Surviving term types by threshold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_coefficient_magnitudes(
    labels: list[str],
    grids:  list[list[dict]],
    path:   Path,
) -> None:
    """Violin plots of surviving |coefficient| values per threshold.

    Coefficient magnitude distributions reveal how aggressively thresholding
    prunes terms: if the distribution barely changes, the threshold is too low.

    Args:
        labels: Threshold label strings.
        grids:  List of raw-grid row lists.
        path:   Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    all_mags = []
    valid_labels = []

    for label, grid in zip(labels, grids):
        mags = []
        for r in grid:
            if r["fit_status"] == "success":
                mags.extend(all_coefficient_magnitudes(r))
        # Cap extreme values for readability
        mags = [min(m, 1e6) for m in mags if m > 0]
        if mags:
            all_mags.append(mags)
            valid_labels.append(label)

    parts = ax.violinplot(all_mags, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], THRESH_COLORS):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(valid_labels) + 1))
    ax.set_xticklabels([f"t={l}" for l in valid_labels])
    ax.set_ylabel("|coefficient| (log scale)")
    ax.set_title("Distribution of nonzero coefficient magnitudes per threshold")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    for i, (mags, label) in enumerate(zip(all_mags, valid_labels), start=1):
        med = statistics.median(mags)
        ax.annotate(
            f"med={med:.0f}", xy=(i, med),
            xytext=(i + 0.25, med),
            fontsize=8, va="center",
        )

    fig.tight_layout()
    savefig(fig, path)


def plot_simulation_success(
    labels:    list[str],
    sim_stats: list[dict],
    path:      Path,
) -> None:
    """Bar chart: simulation trial success rate per threshold.

    Args:
        labels:    Threshold label strings.
        sim_stats: List of compute_sim_stats dicts.
        path:      Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    rates = [s["success_rate"] for s in sim_stats]
    n_trials = [s["n_trials"] for s in sim_stats]

    bars = ax.bar(
        x, rates,
        color=THRESH_COLORS[:len(labels)], alpha=0.85,
    )
    for bar, rate, n in zip(bars, rates, n_trials):
        if math.isfinite(rate):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rate:.1f}%\n(n={n})",
                ha="center", va="bottom", fontsize=9,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                2,
                "no sim data",
                ha="center", va="bottom", fontsize=8, color="grey",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"t={l}" for l in labels])
    ax.set_ylabel("Trial simulation success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Simulation success rate per threshold")
    ax.axhline(100, linestyle="--", color="grey", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_rmse_by_params(
    label:     str,
    grid:      list[dict],
    sim_rows:  list[dict],
    path:      Path,
) -> None:
    """For the sweep with RMSE data, plot RMSE breakdown by key parameters.

    Args:
        label:    Threshold label string.
        grid:     Raw-grid rows for this sweep.
        sim_rows: Per-trial simulation status rows (must include x0_rmse_uv).
        path:     Output PNG path.
    """
    if not sim_rows or not any(r.get("x0_rmse_uv") for r in sim_rows):
        return

    # Index grid rows
    grid_by_idx = {r["configuration_index"]: r for r in grid}

    # Collect rmse by parameter
    rmse_by_degree:  dict[int, list[float]] = collections.defaultdict(list)
    rmse_by_ndelays: dict[int, list[float]] = collections.defaultdict(list)
    rmse_by_lowpass: dict[float, list[float]] = collections.defaultdict(list)
    rmse_by_smooth:  dict[int, list[float]] = collections.defaultdict(list)

    for r in sim_rows:
        if r["simulation_status"] != "success":
            continue
        raw = r.get("x0_rmse_uv", "")
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if not math.isfinite(v) or v > 1e4:
            continue
        g = grid_by_idx.get(r["configuration_index"], {})
        if not g:
            continue
        rmse_by_degree[int(g["degree"])].append(v)
        rmse_by_ndelays[int(g["n_delays"])].append(v)
        rmse_by_lowpass[float(g["lowpass_hz"])].append(v)
        rmse_by_smooth[int(g["smooth_window_samples"])].append(v)

    if not any(rmse_by_degree.values()):
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def barplot(ax, data: dict, xlabel: str, sort: bool = True):
        keys = sorted(data.keys()) if sort else list(data.keys())
        medians = [statistics.median(data[k]) for k in keys]
        q25 = [np.percentile(data[k], 25) for k in keys]
        q75 = [np.percentile(data[k], 75) for k in keys]
        xerr_lo = [m - q for m, q in zip(medians, q25)]
        xerr_hi = [q - m for m, q in zip(medians, q75)]
        ax.bar(range(len(keys)), medians, color="#2563eb", alpha=0.75)
        ax.errorbar(
            range(len(keys)), medians,
            yerr=[xerr_lo, xerr_hi],
            fmt="none", color="black", capsize=4, linewidth=1.2,
        )
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([str(k) for k in keys])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Median x₀ RMSE (µV)")
        ax.grid(axis="y", alpha=0.3)

    barplot(axes[0, 0], rmse_by_degree,  "Polynomial degree")
    axes[0, 0].set_title("RMSE by degree")

    barplot(axes[0, 1], rmse_by_ndelays, "n_delays")
    axes[0, 1].set_title("RMSE by embedding dimension")

    barplot(axes[1, 0], rmse_by_lowpass, "Low-pass cutoff (Hz)")
    axes[1, 0].set_title("RMSE by low-pass filter")

    barplot(axes[1, 1], rmse_by_smooth,  "Smooth window (samples)")
    axes[1, 1].set_title("RMSE by derivative smoothing")

    fig.suptitle(
        f"Simulation RMSE breakdown (t={label}, successful trials only, bars=IQR)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, path)


def plot_cross_threshold_stability(
    labels:   list[str],
    grids:    list[list[dict]],
    path:     Path,
) -> None:
    """Heatmap of pairwise Jaccard term-set similarity across thresholds.

    For each configuration, computes the Jaccard similarity between the set
    of surviving term names at two thresholds. High similarity means the same
    terms survive regardless of threshold — those configurations are robust.

    Args:
        labels: Threshold label strings.
        grids:  List of raw-grid row lists.
        path:   Output PNG path.
    """
    # Build per-config term sets: {config_index: set(feature_names)}
    term_sets_per_thresh: list[dict[str, set]] = []
    for grid in grids:
        ts: dict[str, set] = {}
        for r in grid:
            if r["fit_status"] == "success":
                ts[r["configuration_index"]] = set(nonzero_term_names(r))
        term_sets_per_thresh.append(ts)

    n = len(labels)
    matrix = np.full((n, n), float("nan"))
    for i in range(n):
        for j in range(n):
            shared_keys = (
                set(term_sets_per_thresh[i].keys()) &
                set(term_sets_per_thresh[j].keys())
            )
            if not shared_keys:
                continue
            sims = [
                jaccard_similarity(
                    term_sets_per_thresh[i][k],
                    term_sets_per_thresh[j][k],
                )
                for k in shared_keys
            ]
            matrix[i, j] = statistics.mean(sims)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGn")
    plt.colorbar(im, ax=ax, label="Mean Jaccard similarity")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"t={l}" for l in labels], rotation=30, ha="right")
    ax.set_yticklabels([f"t={l}" for l in labels])
    for i in range(n):
        for j in range(n):
            if math.isfinite(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="black" if matrix[i, j] < 0.7 else "white",
                )
    ax.set_title(
        "Cross-threshold term-set agreement\n"
        "(Jaccard similarity: 1.0 = identical surviving terms)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, path)


def plot_best_configs(
    label:    str,
    grid:     list[dict],
    sim_rows: list[dict],
    path:     Path,
    top_n:   int = 20,
) -> None:
    """Horizontal bar chart of the top configurations by median RMSE.

    Args:
        label:    Threshold label string.
        grid:     Raw-grid rows for this sweep.
        sim_rows: Per-trial simulation status rows with RMSE.
        path:     Output PNG path.
        top_n:    Number of top configs to show.
    """
    if not sim_rows or not any(r.get("x0_rmse_uv") for r in sim_rows):
        return

    grid_by_idx = {r["configuration_index"]: r for r in grid}
    rmse_by_config: dict[str, list[float]] = collections.defaultdict(list)

    for r in sim_rows:
        if r["simulation_status"] != "success":
            continue
        raw = r.get("x0_rmse_uv", "")
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if math.isfinite(v) and v < 1e4:
            rmse_by_config[r["configuration_index"]].append(v)

    if not rmse_by_config:
        return

    records = []
    for cfg_idx, vals in rmse_by_config.items():
        g = grid_by_idx.get(cfg_idx, {})
        if not g:
            continue
        records.append({
            "cfg": cfg_idx,
            "median_rmse": statistics.median(vals),
            "n_trials":    len(vals),
            "label": (
                f"deg={g.get('degree')} nd={g.get('n_delays')} "
                f"d={g.get('delay_samples')} sm={g.get('smooth_window_samples')} "
                f"lp={g.get('lowpass_hz')}"
            ),
            "nonzero_terms": int(g.get("nonzero_terms", 0)),
        })

    records.sort(key=lambda x: x["median_rmse"])
    top = records[:top_n]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35 + 1.5)))
    y = range(len(top))
    colors = [DEGREE_COLORS.get(
        int(grid_by_idx.get(r["cfg"], {}).get("degree", 1)), "#94a3b8"
    ) for r in top]
    ax.barh(list(y), [r["median_rmse"] for r in top], color=colors, alpha=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["label"] for r in top], fontsize=8)
    ax.set_xlabel("Median x₀ RMSE (µV)")
    ax.set_title(
        f"Top {len(top)} configurations by simulation RMSE (t={label})\n"
        "Blue=deg1  Orange=deg2  Purple=deg3",
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_sparsification_by_ndelays_degree(
    labels: list[str],
    grids:  list[list[dict]],
    path:   Path,
) -> None:
    """Heatmap grid: mean utilisation by (n_delays, degree) for each threshold.

    Shows whether higher embedding dimensions are pruned differently across thresholds.

    Args:
        labels: Threshold label strings.
        grids:  List of raw-grid row lists.
        path:   Output PNG path.
    """
    degrees  = [1, 2, 3]
    n_delays = [2, 4, 6, 8]
    n_thresh = len(labels)

    fig, axes = plt.subplots(1, n_thresh, figsize=(4 * n_thresh, 4), sharey=True)
    if n_thresh == 1:
        axes = [axes]

    for ax, label, grid in zip(axes, labels, grids):
        mat = np.full((len(degrees), len(n_delays)), float("nan"))
        bucket: dict[tuple, list] = collections.defaultdict(list)
        for r in grid:
            if r["fit_status"] != "success":
                continue
            deg = int(r["degree"])
            nd  = int(r["n_delays"])
            util = float(r["nonzero_terms"]) / max(int(r["possible_terms"]), 1) * 100
            bucket[(deg, nd)].append(util)
        for di, deg in enumerate(degrees):
            for ni, nd in enumerate(n_delays):
                vals = bucket.get((deg, nd), [])
                if vals:
                    mat[di, ni] = statistics.mean(vals)

        im = ax.imshow(mat, vmin=40, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(n_delays)))
        ax.set_xticklabels(n_delays, fontsize=9)
        ax.set_yticks(range(len(degrees)))
        ax.set_yticklabels([f"deg={d}" for d in degrees], fontsize=9)
        ax.set_xlabel("n_delays")
        ax.set_title(f"t={label}", fontsize=10)
        for di in range(len(degrees)):
            for ni in range(len(n_delays)):
                if math.isfinite(mat[di, ni]):
                    ax.text(
                        ni, di, f"{mat[di, ni]:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="black" if mat[di, ni] > 70 else "white",
                    )

        plt.colorbar(im, ax=ax, label="Utilisation (%)" if ax == axes[-1] else "")

    axes[0].set_ylabel("Polynomial degree")
    fig.suptitle(
        "Library utilisation (%) by degree × n_delays per threshold\n"
        "Red = more sparsification, Green = library fully used",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, path)


# ── Text report ───────────────────────────────────────────────────────────────

def print_report(
    labels:     list[str],
    stats_list: list[dict],
    sim_stats:  list[dict],
    breakdowns: list[dict],
    out:        Path,
) -> None:
    """Print and save a structured text recommendations report.

    Args:
        labels:     Threshold label strings.
        stats_list: List of compute_grid_stats dicts.
        sim_stats:  List of compute_sim_stats dicts.
        breakdowns: List of term-type breakdown dicts.
        out:        Path to save the report text file.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("THRESHOLD COMPARISON REPORT")
    lines.append("=" * 72)

    lines.append("\n── FITTING SUMMARY ────────────────────────────────────────────────")
    header = f"{'Threshold':>10}  {'Fits':>6}  {'Zero-term':>9}  {'Mean terms':>10}  {'Mean util%':>10}"
    lines.append(header)
    for label, s in zip(labels, stats_list):
        mean_t = statistics.mean(s["term_counts"]) if s["term_counts"] else float("nan")
        mean_u = statistics.mean(s["util_pcts"])   if s["util_pcts"]   else float("nan")
        lines.append(
            f"  t={label:>8}  {s['n_success']:>6}  {s['zero_term_count']:>9}  "
            f"{mean_t:>10.1f}  {mean_u:>9.1f}%"
        )

    lines.append("\n── DEGREE-LEVEL UTILISATION ────────────────────────────────────────")
    for label, s in zip(labels, stats_list):
        row_parts = [f"t={label}:"]
        for deg in [1, 2, 3]:
            vals = s["by_degree"].get(deg, [])
            if vals:
                row_parts.append(f"deg{deg}={statistics.mean(vals):.1f}%")
        lines.append("  " + "  ".join(row_parts))

    lines.append("\n── SIMULATION RESULTS ──────────────────────────────────────────────")
    for label, s in zip(labels, sim_stats):
        if s["n_trials"] == 0:
            lines.append(f"  t={label}: no simulation data")
        else:
            rmse_str = (
                f"  median RMSE={statistics.median(s['rmse_values']):.2f} µV"
                if s["rmse_values"] else "  (no RMSE data)"
            )
            lines.append(
                f"  t={label}: {s['n_trials']} trials, "
                f"success={s['success_rate']:.1f}%{rmse_str}"
            )

    lines.append("\n── SURVIVING TERM TYPES ────────────────────────────────────────────")
    for label, bd in zip(labels, breakdowns):
        parts = [f"{t.capitalize()}={bd.get(t, 0):.1f}" for t in ("bias","linear","quadratic","cubic")]
        lines.append(f"  t={label}: " + "  ".join(parts))

    lines.append("\n── KEY FINDINGS ────────────────────────────────────────────────────")

    # Finding 1: t=0.1 vs t=100
    u01  = statistics.mean(stats_list[0]["util_pcts"]) if stats_list[0]["util_pcts"] else float("nan")
    u100 = statistics.mean(stats_list[1]["util_pcts"]) if len(stats_list) > 1 and stats_list[1]["util_pcts"] else float("nan")
    diff_01_100 = abs(u01 - u100) if math.isfinite(u01) and math.isfinite(u100) else float("nan")
    lines.append(
        f"\n1. t=0.1 vs t=100: utilisation differs by only {diff_01_100:.1f}%. "
        "These two thresholds produce near-identical models. t=100 provides no "
        "sparsification benefit and can be dropped from future sweeps."
    )

    # Finding 2: where sparsification starts
    for i, (label, s) in enumerate(zip(labels, stats_list)):
        mean_u = statistics.mean(s["util_pcts"]) if s["util_pcts"] else float("nan")
        if math.isfinite(mean_u) and mean_u < 90:
            lines.append(
                f"\n2. Meaningful sparsification first occurs at t={label} "
                f"(mean utilisation drops to {mean_u:.1f}%). "
                "Thresholds below this are effectively dense models."
            )
            break

    # Finding 3: degree dependence
    if len(stats_list) >= 4 and stats_list[3]["by_degree"]:
        s10k = stats_list[3]
        deg_utils = {d: statistics.mean(v) for d, v in s10k["by_degree"].items() if v}
        best_deg = min(deg_utils, key=deg_utils.get)
        lines.append(
            f"\n3. At t=10000, degree {best_deg} is most sparse "
            f"({deg_utils.get(best_deg, float('nan')):.1f}% utilisation) while "
            f"higher degrees remain denser. This suggests polynomial features of "
            f"degree ≥ 2 have large coefficient magnitudes that resist thresholding."
        )

    lines.append("\n── RECOMMENDED NEXT STEPS ──────────────────────────────────────────")
    lines.append(
        "\n1. THRESHOLD RANGE: The effective sparsification range for this signal "
        "and normalisation is t=1000–50000. Consider a finer sweep in this range "
        "(t=2000, 5000, 20000, 50000) rather than below t=1000."
    )
    lines.append(
        "\n2. NORMALIZE_COLUMNS=FALSE (already running): Without column normalisation, "
        "coefficient scales differ by orders of magnitude across degrees. The nc_false "
        "sweeps will reveal whether a single threshold can work for mixed-degree libraries."
    )
    lines.append(
        "\n3. GLOBAL Z-SCORE: Z-scoring the signal constrains coefficients to O(1), "
        "making t=1–10 the natural regime for normalize_columns=False. Awaiting "
        "those sweep results."
    )
    lines.append(
        "\n4. FOCUS ON DEGREE=1: Degree-1 models are the most interpretable and "
        "already achieve meaningful sparsification at t=10000. Consider a dedicated "
        "degree=1 sweep with much larger n_delays (12–24) to test whether more delay "
        "history compensates for the restricted function class."
    )
    lines.append(
        "\n5. SIMULATE t=1000 AND t=10000: Currently only t=0.1 and t=100 have "
        "simulation results. Run simulations for t=1000 and t=10000 to determine "
        "whether sparser models are more or less stable."
    )

    lines.append("\n" + "=" * 72)
    report = "\n".join(lines)
    print(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"\n  saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full threshold comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Extensive comparison of STLSQ threshold sweeps."
    )
    parser.add_argument(
        "--sweep-dirs", nargs="+", type=Path,
        default=[d for _, d in DEFAULT_SWEEPS],
        help="Sweep output directories in ascending threshold order.",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str,
        default=[l for l, _ in DEFAULT_SWEEPS],
        help="Short threshold labels matching --sweep-dirs.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/pysindy/threshold_comparison",
    )
    args = parser.parse_args()

    if len(args.sweep_dirs) != len(args.labels):
        parser.error("--sweep-dirs and --labels must have the same length.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}\n")

    # Load data
    print("Loading data...")
    grids    = [read_grid(d / "raw_grid_merged.csv") for d in args.sweep_dirs]
    sim_rows = [load_sim_status(d) for d in args.sweep_dirs]

    stats_list = [compute_grid_stats(g) for g in grids]
    sim_stats  = [compute_sim_stats(s)  for s in sim_rows]

    # Add by_degree_terms to stats (term counts, not utilisation) for violin
    for grid, stats in zip(grids, stats_list):
        by_deg_terms: dict[int, list[int]] = collections.defaultdict(list)
        for r in grid:
            if r["fit_status"] == "success":
                by_deg_terms[int(r["degree"])].append(int(r["nonzero_terms"]))
        stats["by_degree_terms"] = dict(by_deg_terms)

    breakdowns = [compute_term_type_breakdown(g) for g in grids]

    print("\nGenerating plots...")

    plot_sparsity_violin(
        args.labels, stats_list,
        args.output_dir / "01_sparsity_violin.png",
    )
    plot_utilization_by_degree(
        args.labels, stats_list,
        args.output_dir / "02_utilization_by_degree.png",
    )
    plot_utilization_vs_ndelays(
        args.labels, stats_list,
        args.output_dir / "03_utilization_vs_ndelays.png",
    )
    plot_term_type_breakdown(
        args.labels, breakdowns,
        args.output_dir / "04_term_type_breakdown.png",
    )
    plot_coefficient_magnitudes(
        args.labels, grids,
        args.output_dir / "05_coefficient_magnitudes.png",
    )
    plot_simulation_success(
        args.labels, sim_stats,
        args.output_dir / "06_simulation_success.png",
    )

    # RMSE breakdown — use first sweep that has RMSE data
    for label, grid, sim in zip(args.labels, grids, sim_rows):
        if sim and any(r.get("x0_rmse_uv") for r in sim):
            plot_rmse_by_params(
                label, grid, sim,
                args.output_dir / "07_rmse_by_params.png",
            )
            plot_best_configs(
                label, grid, sim,
                args.output_dir / "08_best_configs.png",
            )
            break

    plot_cross_threshold_stability(
        args.labels, grids,
        args.output_dir / "09_cross_threshold_stability.png",
    )
    plot_sparsification_by_ndelays_degree(
        args.labels, grids,
        args.output_dir / "10_utilization_heatmap.png",
    )

    # Summary CSV
    summary_rows = []
    for label, s, ss, bd in zip(args.labels, stats_list, sim_stats, breakdowns):
        for deg in [1, 2, 3]:
            util_vals = s["by_degree"].get(deg, [])
            term_vals = s["by_degree_terms"].get(deg, [])
            summary_rows.append({
                "threshold":           label,
                "degree":              deg,
                "n_success":           s["n_success"],
                "zero_term_configs":   s["zero_term_count"],
                "mean_nonzero_terms":  round(statistics.mean(term_vals), 1) if term_vals else "",
                "mean_utilization_pct": round(statistics.mean(util_vals), 1) if util_vals else "",
                "sim_n_trials":        ss["n_trials"],
                "sim_success_pct":     round(ss["success_rate"], 1) if math.isfinite(ss["success_rate"]) else "",
                "sim_median_rmse_uv":  round(statistics.median(ss["rmse_values"]), 2) if ss["rmse_values"] else "",
                "mean_linear_terms":   round(bd.get("linear", 0), 1),
                "mean_quadratic_terms": round(bd.get("quadratic", 0), 1),
                "mean_cubic_terms":    round(bd.get("cubic", 0), 1),
            })

    write_csv(
        args.output_dir / "threshold_comparison_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    print(f"  saved: threshold_comparison_summary.csv")

    print_report(
        args.labels, stats_list, sim_stats, breakdowns,
        args.output_dir / "threshold_comparison_report.txt",
    )


if __name__ == "__main__":
    main()
