"""Compare polynomial degree 1–7 models at t=1000.

Analyses whether higher-degree polynomials capture nonlinearity better by
examining term structure, coefficient mass distribution, simulation stability,
and RMSE across degrees 1, 2, 3, 5, 7.

Data sources (both use threshold=1000):
  - outputs/pysindy/raw_grid_threshold1000/   degrees 1, 2, 3
  - outputs/pysindy/raw_grid_deg57_t1000/     degrees 5, 7 (may be partial)

Usage (from repo root):
    .venv/bin/python scripts/pysindy/sweep_analysis/compare_degrees.py \\
        --output-dir outputs/pysindy/degree_comparison
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
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

DIR_DEG123 = ROOT / "outputs/pysindy/raw_grid_threshold1000"
DIR_DEG57  = ROOT / "outputs/pysindy/raw_grid_deg57_t1000"

DEGREE_COLORS = {1: "#2563eb", 2: "#16a34a", 3: "#d97706", 5: "#7c3aed", 7: "#dc2626"}
ORDER_COLORS  = {
    0: "#94a3b8",  # bias
    1: "#2563eb",  # linear
    2: "#16a34a",  # quadratic
    3: "#d97706",  # cubic
    4: "#7c3aed",  # quartic
    5: "#dc2626",  # quintic
    6: "#0891b2",  # sextic
    7: "#be185d",  # septic
}
ORDER_LABELS = {0: "Bias", 1: "Linear", 2: "Quadratic", 3: "Cubic",
                4: "Quartic", 5: "Quintic", 6: "Sextic", 7: "Septic"}


# ── I/O ────────────────────────────────────────────────────────────────────────

def read_grid(path: Path) -> list[dict]:
    """Load a merged raw-grid CSV, raising if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_sim_status(sweep_dir: Path) -> list[dict]:
    """Load simulation status rows from simulations/status/ or simulations_pertrial/status/."""
    for subdir in ("simulations/status", "simulations_pertrial/status"):
        candidate = sweep_dir / subdir
        if candidate.exists():
            count = sum(1 for _ in candidate.glob("config_*.csv"))
            if count > 0:
                rows = []
                for p in sorted(candidate.glob("config_*.csv")):
                    with p.open(newline="") as f:
                        rows.extend(csv.DictReader(f))
                return rows
    return []


# ── Feature helpers ─────────────────────────────────────────────────────────────

def term_order(name: str) -> int:
    """Return total polynomial order of a feature name.

    Examples: '1'->0, 'x0'->1, 'x0^2'->2, 'x0 x1'->2, 'x0^3 x1^2'->5.
    """
    if name == "1":
        return 0
    # Sum explicit exponents (x0^3 → 3)
    explicit = sum(int(p) for p in re.findall(r"\^(\d+)", name))
    # Count variables without explicit exponent (x0 in 'x0 x1^2' → 1)
    implicit = len(re.findall(r"x\d+(?!\^\d)", name))
    return explicit + implicit


def surviving_term_orders(row: dict) -> list[int]:
    """Return polynomial orders of all surviving (nonzero) terms."""
    try:
        names  = json.loads(row["feature_names_json"])
        coeffs = np.array(json.loads(row["coefficients_json"]))
        mask   = np.any(np.abs(coeffs) > 1e-12, axis=0)
        return [term_order(n) for n, m in zip(names, mask) if m]
    except (KeyError, json.JSONDecodeError, ValueError):
        return []


def nonzero_coeffs_by_order(row: dict) -> dict[int, list[float]]:
    """Return {order: [|coeff|, ...]} for all nonzero coefficients."""
    try:
        names  = json.loads(row["feature_names_json"])
        coeffs = np.array(json.loads(row["coefficients_json"]))  # (n_eq, n_feat)
    except (KeyError, json.JSONDecodeError, ValueError):
        return {}
    result: dict[int, list[float]] = collections.defaultdict(list)
    for feat_idx, name in enumerate(names):
        for eq_idx in range(coeffs.shape[0]):
            c = abs(coeffs[eq_idx, feat_idx])
            if c > 1e-12:
                result[term_order(name)].append(c)
    return dict(result)


# ── Aggregation ────────────────────────────────────────────────────────────────

def collect_by_degree(rows: list[dict]) -> dict[int, list[dict]]:
    """Group successful fitting rows by polynomial degree."""
    by_deg: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r["fit_status"] == "success":
            by_deg[int(r["degree"])].append(r)
    return dict(by_deg)


def sim_success_rate(sim_rows: list[dict], grid_rows: list[dict]) -> dict[int, float]:
    """Return simulation success rate per degree."""
    idx_to_deg = {r["configuration_index"]: int(r["degree"]) for r in grid_rows}
    by_deg: dict[int, list[bool]] = collections.defaultdict(list)
    for r in sim_rows:
        deg = idx_to_deg.get(r["configuration_index"])
        if deg is not None:
            by_deg[deg].append(r["simulation_status"] == "success")
    return {d: sum(v) / len(v) * 100 for d, v in by_deg.items()}


def rmse_by_degree(sim_rows: list[dict], grid_rows: list[dict]) -> dict[int, list[float]]:
    """Return RMSE values per degree (successful trials only)."""
    idx_to_deg = {r["configuration_index"]: int(r["degree"]) for r in grid_rows}
    by_deg: dict[int, list[float]] = collections.defaultdict(list)
    for r in sim_rows:
        if r["simulation_status"] != "success":
            continue
        raw = r.get("x0_rmse_uv", "")
        if not raw:
            continue
        try:
            v = float(raw)
            if math.isfinite(v) and 0 < v < 1e4:
                deg = idx_to_deg.get(r["configuration_index"])
                if deg is not None:
                    by_deg[deg].append(v)
        except ValueError:
            pass
    return dict(by_deg)


# ── Plots ──────────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def plot_success_rate(
    rates_123: dict[int, float],
    rates_57:  dict[int, float],
    path: Path,
) -> None:
    """Bar chart: simulation success rate per degree."""
    all_rates = {**rates_123, **rates_57}
    degrees = sorted(all_rates)
    colors  = [DEGREE_COLORS[d] for d in degrees]
    rates   = [all_rates[d] for d in degrees]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(degrees)), rates, color=colors, alpha=0.85)
    for bar, rate, deg in zip(bars, rates, degrees):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axvline(2.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(2.6, 5, "deg57 partially\ncomplete", fontsize=8, color="grey")
    ax.set_xticks(range(len(degrees)))
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("Simulation success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Simulation success rate by polynomial degree (threshold=1000)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_utilization(by_deg_all: dict[int, list[dict]], path: Path) -> None:
    """Bar chart: mean library utilisation per degree."""
    degrees, means, stds = [], [], []
    for deg in sorted(by_deg_all):
        utils = [
            float(r["nonzero_terms"]) / max(int(r["possible_terms"]), 1) * 100
            for r in by_deg_all[deg]
        ]
        degrees.append(deg)
        means.append(statistics.mean(utils))
        stds.append(statistics.stdev(utils) if len(utils) > 1 else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [DEGREE_COLORS[d] for d in degrees]
    bars = ax.bar(range(len(degrees)), means, color=colors, alpha=0.85,
                  yerr=stds, capsize=4, error_kw={"linewidth": 1.2})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(degrees)))
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("Mean library utilisation (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Library utilisation by degree (threshold=1000, bars=±1σ)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_nonlinearity_fraction(by_deg_all: dict[int, list[dict]], path: Path) -> None:
    """Bar chart: mean fraction of surviving terms that are nonlinear (order≥2)."""
    degrees, nl_fracs, lin_fracs, bias_fracs = [], [], [], []
    for deg in sorted(by_deg_all):
        nl_per_cfg, lin_per_cfg, bias_per_cfg = [], [], []
        for r in by_deg_all[deg]:
            orders = surviving_term_orders(r)
            if not orders:
                continue
            total = len(orders)
            bias_per_cfg.append(sum(1 for o in orders if o == 0) / total * 100)
            lin_per_cfg.append(sum(1 for o in orders if o == 1) / total * 100)
            nl_per_cfg.append(sum(1 for o in orders if o >= 2) / total * 100)
        if nl_per_cfg:
            degrees.append(deg)
            nl_fracs.append(statistics.mean(nl_per_cfg))
            lin_fracs.append(statistics.mean(lin_per_cfg))
            bias_fracs.append(statistics.mean(bias_per_cfg))

    x = np.arange(len(degrees))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, bias_fracs, label="Bias (order 0)", color="#94a3b8", alpha=0.85)
    ax.bar(x, lin_fracs,  label="Linear (order 1)", color="#2563eb", alpha=0.85,
           bottom=bias_fracs)
    bottom2 = [b + l for b, l in zip(bias_fracs, lin_fracs)]
    ax.bar(x, nl_fracs, label="Nonlinear (order ≥ 2)", color="#dc2626", alpha=0.85,
           bottom=bottom2)

    for i, (deg, nl, lin, bi) in enumerate(zip(degrees, nl_fracs, lin_fracs, bias_fracs)):
        if nl > 3:
            ax.text(i, bi + lin + nl / 2, f"{nl:.1f}%",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("% of surviving terms")
    ax.set_ylim(0, 110)
    ax.set_title("Composition of surviving terms by degree (threshold=1000)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_term_order_breakdown(by_deg_all: dict[int, list[dict]], path: Path) -> None:
    """Stacked bar: mean surviving term count per polynomial order, per degree."""
    degrees = sorted(by_deg_all)
    max_order = max(
        max((o for r in rows for o in surviving_term_orders(r)), default=0)
        for rows in by_deg_all.values()
    )
    orders = list(range(max_order + 1))

    # mean count of surviving terms at each order, per degree
    means: dict[int, list[float]] = {deg: [] for deg in degrees}
    for deg in degrees:
        order_counts: dict[int, list[int]] = collections.defaultdict(list)
        for r in by_deg_all[deg]:
            term_orders = surviving_term_orders(r)
            c = collections.Counter(term_orders)
            for o in orders:
                order_counts[o].append(c.get(o, 0))
        means[deg] = [statistics.mean(order_counts[o]) if order_counts[o] else 0
                      for o in orders]

    x = np.arange(len(degrees))
    w = 0.65
    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(degrees))

    for o in orders:
        vals = np.array([means[d][o] for d in degrees])
        color = ORDER_COLORS.get(o, "#999999")
        ax.bar(x, vals, w, bottom=bottoms, label=ORDER_LABELS.get(o, f"Order {o}"),
               color=color, alpha=0.85)
        for i, (v, bot) in enumerate(zip(vals, bottoms)):
            if v > 0.5:
                ax.text(i, bot + v / 2, f"{v:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 2 else "black")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("Mean surviving terms per configuration")
    ax.set_title("Surviving term breakdown by polynomial order and degree (threshold=1000)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_coeff_mass_by_order(by_deg_all: dict[int, list[dict]], path: Path) -> None:
    """Stacked bar: fraction of total |coeff| mass from each polynomial order."""
    degrees = sorted(by_deg_all)
    max_order = max(
        max((o for r in rows for o in surviving_term_orders(r)), default=0)
        for rows in by_deg_all.values()
    )
    orders = list(range(max_order + 1))

    frac_means: dict[int, list[float]] = {deg: [] for deg in degrees}
    for deg in degrees:
        mass_by_order: dict[int, list[float]] = {o: [] for o in orders}
        for r in by_deg_all[deg]:
            by_o = nonzero_coeffs_by_order(r)
            total_mass = sum(v for vals in by_o.values() for v in vals)
            if total_mass == 0:
                continue
            for o in orders:
                mass = sum(by_o.get(o, []))
                mass_by_order[o].append(mass / total_mass * 100)
        frac_means[deg] = [
            statistics.mean(mass_by_order[o]) if mass_by_order[o] else 0
            for o in orders
        ]

    x = np.arange(len(degrees))
    w = 0.65
    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(degrees))

    for o in orders:
        vals = np.array([frac_means[d][o] for d in degrees])
        color = ORDER_COLORS.get(o, "#999999")
        ax.bar(x, vals, w, bottom=bottoms, label=ORDER_LABELS.get(o, f"Order {o}"),
               color=color, alpha=0.85)
        for i, (v, bot) in enumerate(zip(vals, bottoms)):
            if v > 2:
                ax.text(i, bot + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("% of total |coefficient| mass")
    ax.set_ylim(0, 110)
    ax.set_title(
        "Coefficient mass fraction by polynomial order and degree (threshold=1000)\n"
        "If nonlinear orders carry large mass → nonlinearity is dynamically important"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_rmse(rmse_all: dict[int, list[float]], path: Path) -> None:
    """Box plot: x0 RMSE distribution per degree (µV)."""
    degrees = sorted(rmse_all)
    if not degrees:
        print("  [warn] no RMSE data — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    data   = [rmse_all[d] for d in degrees]
    colors = [DEGREE_COLORS[d] for d in degrees]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    for i, (deg, vals) in enumerate(zip(degrees, data), start=1):
        med = statistics.median(vals)
        ax.text(i, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else med,
                f"med={med:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(1, len(degrees) + 1))
    ax.set_xticklabels([f"Degree {d}\n(n={len(rmse_all[d])})" for d in degrees])
    ax.set_ylabel("x₀ RMSE (µV)")
    ax.set_title(
        "Simulation RMSE by degree (threshold=1000, successful trials only)\n"
        "Lower = better reconstruction of held-out trials"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_nonlinear_coeff_mass_vs_rmse(
    by_deg_all: dict[int, list[dict]],
    rmse_all: dict[int, list[float]],
    path: Path,
) -> None:
    """Scatter: mean nonlinear coeff mass fraction vs median RMSE per degree."""
    degrees = sorted(set(by_deg_all) & set(rmse_all))
    if len(degrees) < 2:
        print("  [warn] need ≥2 degrees with RMSE — skipping scatter plot")
        return

    nl_fracs, medians = [], []
    for deg in degrees:
        total_nl, total_mass = 0.0, 0.0
        for r in by_deg_all[deg]:
            by_o = nonzero_coeffs_by_order(r)
            for o, vals in by_o.items():
                m = sum(vals)
                if o >= 2:
                    total_nl += m
                total_mass += m
        nl_fracs.append(100 * total_nl / total_mass if total_mass > 0 else 0)
        medians.append(statistics.median(rmse_all[deg]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for deg, x, y in zip(degrees, nl_fracs, medians):
        ax.scatter([x], [y], color=DEGREE_COLORS[deg], s=120, zorder=5)
        ax.annotate(f"Degree {deg}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Nonlinear coefficient mass fraction (%)")
    ax.set_ylabel("Median x₀ RMSE (µV)")
    ax.set_title(
        "Does more nonlinear mass → better RMSE?\n"
        "Down-left = nonlinearity helps; down-right = nonlinearity needed for accuracy"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(
    by_deg_all: dict[int, list[dict]],
    success_rates: dict[int, float],
    rmse_all: dict[int, list[float]],
    out: Path,
) -> None:
    lines = ["=" * 68,
             "DEGREE COMPARISON REPORT (threshold=1000)",
             "=" * 68]

    lines.append("\n── FITTING ──────────────────────────────────────────────────────")
    lines.append(f"  {'Degree':>7}  {'N configs':>10}  {'Mean util%':>10}  "
                 f"{'% nonlinear terms':>18}  {'% nonlinear coeff mass':>22}")
    for deg in sorted(by_deg_all):
        rows = by_deg_all[deg]
        utils = [float(r["nonzero_terms"]) / max(int(r["possible_terms"]), 1) * 100
                 for r in rows]
        nl_term_fracs = []
        nl_mass_fracs = []
        for r in rows:
            orders = surviving_term_orders(r)
            if orders:
                nl_term_fracs.append(sum(1 for o in orders if o >= 2) / len(orders) * 100)
            by_o = nonzero_coeffs_by_order(r)
            total = sum(v for vals in by_o.values() for v in vals)
            nl = sum(v for o, vals in by_o.items() if o >= 2 for v in vals)
            if total > 0:
                nl_mass_fracs.append(nl / total * 100)

        lines.append(
            f"  {deg:>7}  {len(rows):>10}  {statistics.mean(utils):>10.1f}%  "
            f"{statistics.mean(nl_term_fracs) if nl_term_fracs else float('nan'):>17.1f}%  "
            f"{statistics.mean(nl_mass_fracs) if nl_mass_fracs else float('nan'):>21.1f}%"
        )

    lines.append("\n── SIMULATION ───────────────────────────────────────────────────")
    for deg in sorted(success_rates):
        rate = success_rates[deg]
        rmses = rmse_all.get(deg, [])
        rmse_str = f"  median RMSE={statistics.median(rmses):.2f} µV  (n={len(rmses)} trials)" \
                   if rmses else "  no RMSE data"
        lines.append(f"  Degree {deg}: {rate:.1f}% success{rmse_str}")

    lines.append("\n── KEY FINDINGS ─────────────────────────────────────────────────")

    # Do higher degrees use nonlinear terms?
    nl_by_deg = {}
    for deg, rows in by_deg_all.items():
        fracs = []
        for r in rows:
            by_o = nonzero_coeffs_by_order(r)
            total = sum(v for vals in by_o.values() for v in vals)
            nl = sum(v for o, vals in by_o.items() if o >= 2 for v in vals)
            if total > 0:
                fracs.append(nl / total * 100)
        nl_by_deg[deg] = statistics.mean(fracs) if fracs else 0

    low_degs  = [d for d in sorted(nl_by_deg) if d <= 3]
    high_degs = [d for d in sorted(nl_by_deg) if d >= 5]
    if low_degs and high_degs:
        low_nl  = statistics.mean(nl_by_deg[d] for d in low_degs)
        high_nl = statistics.mean(nl_by_deg[d] for d in high_degs)
        lines.append(
            f"\n1. NONLINEAR MASS: deg1-3 have {low_nl:.1f}% nonlinear coeff mass; "
            f"deg5-7 have {high_nl:.1f}%. "
            + ("Higher degrees carry substantially more nonlinear mass — they are "
               "genuinely fitting nonlinear structure."
               if high_nl > low_nl + 10 else
               "Similar nonlinear mass across degrees — higher degrees do not add "
               "meaningfully more nonlinear dynamics at this threshold.")
        )

    # RMSE trend
    if len(rmse_all) >= 2:
        deg_rmse = [(d, statistics.median(v)) for d, v in sorted(rmse_all.items()) if v]
        best_deg, best_rmse = min(deg_rmse, key=lambda x: x[1])
        lines.append(
            f"\n2. RMSE: Best median RMSE is degree {best_deg} ({best_rmse:.2f} µV). "
            + ("RMSE improves with degree — nonlinearity helps reconstruction."
               if deg_rmse[-1][1] < deg_rmse[0][1] else
               "RMSE does not consistently improve with degree.")
        )

    # Stability vs degree
    if success_rates:
        deg_stab = sorted(success_rates.items())
        lines.append(
            f"\n3. STABILITY: "
            + "  ".join(f"deg{d}={r:.1f}%" for d, r in deg_stab)
            + (". Higher degrees are less stable — the denser, stiffer ODE is harder to integrate."
               if deg_stab[-1][1] < deg_stab[0][1] else
               ". Stability is comparable across degrees.")
        )

    lines.append("\n" + "=" * 68)
    report = "\n".join(lines)
    print(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"\n  saved: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare polynomial degrees 1-7 at threshold=1000."
    )
    parser.add_argument("--dir-deg123", type=Path, default=DIR_DEG123)
    parser.add_argument("--dir-deg57",  type=Path, default=DIR_DEG57)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/pysindy/degree_comparison",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {args.output_dir}\n")

    print("Loading grids...")
    grid_123 = read_grid(args.dir_deg123 / "raw_grid_merged.csv")
    grid_57  = read_grid(args.dir_deg57  / "raw_grid_merged.csv")
    all_grid = grid_123 + grid_57

    print("Loading simulation status...")
    sim_123 = load_sim_status(args.dir_deg123)
    sim_57  = load_sim_status(args.dir_deg57)

    by_deg_123 = collect_by_degree(grid_123)
    by_deg_57  = collect_by_degree(grid_57)
    by_deg_all = {**by_deg_123, **by_deg_57}

    success_123 = sim_success_rate(sim_123, grid_123)
    success_57  = sim_success_rate(sim_57,  grid_57)

    rmse_123 = rmse_by_degree(sim_123, grid_123)
    rmse_57  = rmse_by_degree(sim_57,  grid_57)
    rmse_all = {**rmse_123, **rmse_57}

    print(f"  Degrees loaded: {sorted(by_deg_all)}")
    for d, rows in sorted(by_deg_all.items()):
        print(f"    degree {d}: {len(rows)} configs, "
              f"{sum(1 for r in rows if int(r.get('nonzero_terms', 0)) > 0)} with nonzero terms")

    print("\nGenerating plots...")
    plot_success_rate(success_123, success_57, args.output_dir / "01_success_rate.png")
    plot_utilization(by_deg_all, args.output_dir / "02_utilization.png")
    plot_nonlinearity_fraction(by_deg_all, args.output_dir / "03_nonlinearity_fraction.png")
    plot_term_order_breakdown(by_deg_all, args.output_dir / "04_term_order_breakdown.png")
    plot_coeff_mass_by_order(by_deg_all, args.output_dir / "05_coeff_mass_by_order.png")
    plot_rmse(rmse_all, args.output_dir / "06_rmse_by_degree.png")
    plot_nonlinear_coeff_mass_vs_rmse(by_deg_all, rmse_all, args.output_dir / "07_nl_mass_vs_rmse.png")

    print_report(by_deg_all, {**success_123, **success_57}, rmse_all,
                 args.output_dir / "degree_comparison_report.txt")


if __name__ == "__main__":
    main()
