"""Compare raw (µV) vs global-z-score normalisation at t=0.1.

Matches configurations 1-to-1 by hyperparameters and produces side-by-side
figures on success rate, term structure, coefficient magnitudes, and RMSE.

Usage (from repo root):
    .venv/bin/python scripts/pysindy/sweep_analysis/compare_normalizations.py \
        --output-dir outputs/pysindy/normalization_comparison
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

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_DIR = ROOT / "outputs/pysindy/raw_grid"
GSZ_DIR = ROOT / "outputs/pysindy/raw_grid_globalzscore"

COLORS = {"raw": "#2563eb", "gsz": "#d97706"}
LABELS = {"raw": "Raw (µV, t=0.1)", "gsz": "Global z-score (t=0.1)"}


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_grid(path: Path) -> list[dict]:
    """Load a merged raw-grid CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Grid CSV not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_sim_status(sweep_dir: Path) -> list[dict]:
    """Load simulation status rows, preferring simulations/ over simulations_pertrial/."""
    candidates = [
        sweep_dir / "simulations" / "status",
        sweep_dir / "simulations_pertrial" / "status",
    ]
    for candidate in candidates:
        if candidate.exists():
            count = sum(1 for _ in candidate.glob("config_*.csv"))
            if count > 0:
                rows = []
                for p in sorted(candidate.glob("config_*.csv")):
                    with p.open(newline="") as f:
                        rows.extend(csv.DictReader(f))
                return rows
    return []


def config_key(row: dict) -> tuple:
    """Canonical hyperparameter key for matching across sweeps."""
    return (
        float(row["lowpass_hz"]),
        int(row["degree"]),
        int(row["n_delays"]),
        int(row["delay_samples"]),
        int(row["smooth_window_samples"]),
    )


# ── Feature helpers ────────────────────────────────────────────────────────────

def classify_term(name: str) -> str:
    """Classify a feature name as bias, linear, quadratic, or cubic+."""
    if name == "1":
        return "bias"
    powers = re.findall(r"\^(\d+)", name)
    total = sum(int(p) for p in powers) + (name.count(" ") if not powers else 0)
    if total >= 3:
        return "cubic+"
    if total == 2 or "^2" in name:
        return "quadratic"
    return "linear"


def surviving_terms(row: dict) -> set[str]:
    """Return feature names with at least one nonzero coefficient."""
    try:
        names = json.loads(row["feature_names_json"])
        coeffs = np.array(json.loads(row["coefficients_json"]))
        mask = np.any(np.abs(coeffs) > 1e-12, axis=0)
        return {n for n, m in zip(names, mask) if m}
    except (KeyError, json.JSONDecodeError, ValueError):
        return set()


def nonzero_coeffs(row: dict) -> list[float]:
    """Return absolute values of all nonzero coefficients."""
    try:
        coeffs = np.array(json.loads(row["coefficients_json"])).ravel()
        return [abs(c) for c in coeffs if abs(c) > 1e-12]
    except (KeyError, json.JSONDecodeError, ValueError):
        return []


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── Matching ───────────────────────────────────────────────────────────────────

def match_grids(
    raw: list[dict], gsz: list[dict]
) -> list[tuple[dict, dict]]:
    """Return matched (raw_row, gsz_row) pairs sharing the same hyperparameters."""
    gsz_by_key = {config_key(r): r for r in gsz if r["fit_status"] == "success"}
    pairs = []
    for r in raw:
        if r["fit_status"] != "success":
            continue
        key = config_key(r)
        if key in gsz_by_key:
            pairs.append((r, gsz_by_key[key]))
    return pairs


def sim_success_by_config(sim_rows: list[dict]) -> dict[str, float]:
    """Return success rate per configuration_index."""
    by_cfg: dict[str, list[bool]] = collections.defaultdict(list)
    for r in sim_rows:
        by_cfg[r["configuration_index"]].append(r["simulation_status"] == "success")
    return {k: sum(v) / len(v) for k, v in by_cfg.items()}


def rmse_by_config(sim_rows: list[dict]) -> dict[str, list[float]]:
    """Return RMSE values per configuration_index (skips rows without RMSE)."""
    by_cfg: dict[str, list[float]] = collections.defaultdict(list)
    for r in sim_rows:
        if r["simulation_status"] != "success":
            continue
        raw = r.get("x0_rmse_uv", "")
        if not raw:
            continue
        try:
            v = float(raw)
            if math.isfinite(v) and v < 1e4:
                by_cfg[r["configuration_index"]].append(v)
        except ValueError:
            pass
    return dict(by_cfg)


# ── Plotting ───────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def plot_success_rate(
    raw_sim: list[dict],
    gsz_sim: list[dict],
    path: Path,
) -> None:
    """Bar chart: overall and per-degree simulation success rates."""
    raw_grid_rows = read_grid(RAW_DIR / "raw_grid_merged.csv")
    gsz_grid_rows = read_grid(GSZ_DIR / "raw_grid_merged.csv")
    raw_idx_to_deg = {r["configuration_index"]: int(r["degree"]) for r in raw_grid_rows}
    gsz_idx_to_deg = {r["configuration_index"]: int(r["degree"]) for r in gsz_grid_rows}

    def rates_by_degree(sim_rows, idx_to_deg):
        by_deg: dict[int, list[bool]] = collections.defaultdict(list)
        overall = []
        for r in sim_rows:
            ok = r["simulation_status"] == "success"
            overall.append(ok)
            deg = idx_to_deg.get(r["configuration_index"])
            if deg is not None:
                by_deg[deg].append(ok)
        return overall, by_deg

    raw_overall, raw_by_deg = rates_by_degree(raw_sim, raw_idx_to_deg)
    gsz_overall, gsz_by_deg = rates_by_degree(gsz_sim, gsz_idx_to_deg)

    degrees = sorted(set(raw_by_deg) | set(gsz_by_deg))
    categories = ["Overall"] + [f"Degree {d}" for d in degrees]
    raw_rates = [100 * sum(raw_overall) / len(raw_overall)] + [
        100 * sum(raw_by_deg[d]) / len(raw_by_deg[d]) for d in degrees
    ]
    gsz_rates = [100 * sum(gsz_overall) / len(gsz_overall)] + [
        100 * sum(gsz_by_deg.get(d, [False])) / len(gsz_by_deg.get(d, [False])) for d in degrees
    ]

    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_raw = ax.bar(x - w / 2, raw_rates, w, label=LABELS["raw"], color=COLORS["raw"], alpha=0.82)
    bars_gsz = ax.bar(x + w / 2, gsz_rates, w, label=LABELS["gsz"], color=COLORS["gsz"], alpha=0.82)

    for bars, rates in [(bars_raw, raw_rates), (bars_gsz, gsz_rates)]:
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Simulation success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Simulation success rate: raw µV vs global z-score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_utilization(pairs: list[tuple[dict, dict]], path: Path) -> None:
    """Grouped bar: mean term utilisation by degree for each normalisation."""
    raw_by_deg: dict[int, list[float]] = collections.defaultdict(list)
    gsz_by_deg: dict[int, list[float]] = collections.defaultdict(list)

    for raw_r, gsz_r in pairs:
        deg = int(raw_r["degree"])
        raw_util = float(raw_r["nonzero_terms"]) / max(int(raw_r["possible_terms"]), 1) * 100
        gsz_util = float(gsz_r["nonzero_terms"]) / max(int(gsz_r["possible_terms"]), 1) * 100
        raw_by_deg[deg].append(raw_util)
        gsz_by_deg[deg].append(gsz_util)

    degrees = sorted(raw_by_deg)
    x = np.arange(len(degrees))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    raw_vals = [statistics.mean(raw_by_deg[d]) for d in degrees]
    gsz_vals = [statistics.mean(gsz_by_deg[d]) for d in degrees]

    bars_raw = ax.bar(x - w / 2, raw_vals, w, label=LABELS["raw"], color=COLORS["raw"], alpha=0.82)
    bars_gsz = ax.bar(x + w / 2, gsz_vals, w, label=LABELS["gsz"], color=COLORS["gsz"], alpha=0.82)

    for bars, vals in [(bars_raw, raw_vals), (bars_gsz, gsz_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("Mean library utilisation (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Library utilisation: raw µV vs global z-score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_term_types(pairs: list[tuple[dict, dict]], path: Path) -> None:
    """Stacked bar: mean surviving terms by type for each normalisation."""
    term_types = ["bias", "linear", "quadratic", "cubic+"]
    colors = {"bias": "#94a3b8", "linear": "#2563eb", "quadratic": "#d97706", "cubic+": "#7c3aed"}

    counts: dict[str, dict[str, list[int]]] = {
        "raw": collections.defaultdict(list),
        "gsz": collections.defaultdict(list),
    }
    for raw_r, gsz_r in pairs:
        for key, row in [("raw", raw_r), ("gsz", gsz_r)]:
            terms = surviving_terms(row)
            by_type = collections.Counter(classify_term(t) for t in terms)
            for t in term_types:
                counts[key][t].append(by_type.get(t, 0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, key, label in zip(axes, ["raw", "gsz"], [LABELS["raw"], LABELS["gsz"]]):
        bottoms = 0.0
        for t in term_types:
            mean_val = statistics.mean(counts[key][t]) if counts[key][t] else 0
            ax.bar([0], [mean_val], bottom=[bottoms], label=t.capitalize(),
                   color=colors[t], alpha=0.85)
            if mean_val > 0.5:
                ax.text(0, bottoms + mean_val / 2, f"{mean_val:.1f}",
                        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            bottoms += mean_val
        ax.set_xticks([])
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Mean surviving terms per configuration")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Surviving term types by normalisation", fontsize=12)
    fig.tight_layout()
    savefig(fig, path)


def plot_coefficient_magnitudes(pairs: list[tuple[dict, dict]], path: Path) -> None:
    """Violin: |coefficient| distribution for raw vs z-score (log scale)."""
    raw_mags, gsz_mags = [], []
    for raw_r, gsz_r in pairs:
        raw_mags.extend(nonzero_coeffs(raw_r))
        gsz_mags.extend(nonzero_coeffs(gsz_r))

    raw_mags = [min(m, 1e8) for m in raw_mags if m > 0]
    gsz_mags = [min(m, 1e8) for m in gsz_mags if m > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([raw_mags, gsz_mags], showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [COLORS["raw"], COLORS["gsz"]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_yscale("log")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([LABELS["raw"], LABELS["gsz"]])
    ax.set_ylabel("|coefficient| (log scale)")
    ax.set_title("Nonzero coefficient magnitudes: raw µV vs global z-score")
    ax.grid(axis="y", alpha=0.3)

    for i, (mags, key) in enumerate([(raw_mags, "raw"), (gsz_mags, "gsz")], start=1):
        med = statistics.median(mags)
        ax.annotate(
            f"median={med:.3g}", xy=(i, med),
            xytext=(i + 0.25, med), fontsize=9, va="center",
        )

    fig.tight_layout()
    savefig(fig, path)


def plot_jaccard_per_config(pairs: list[tuple[dict, dict]], path: Path) -> None:
    """Histogram of per-config Jaccard similarity between raw and z-score term sets."""
    scores = [jaccard(surviving_terms(r), surviving_terms(g)) for r, g in pairs]
    n_identical = sum(1 for s in scores if s >= 0.999)
    n_zero = sum(1 for s in scores if s < 0.01)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=20, range=(0, 1), color="#16a34a", alpha=0.8, edgecolor="white")
    ax.axvline(statistics.median(scores), color="black", linewidth=1.5,
                linestyle="--", label=f"Median = {statistics.median(scores):.2f}")
    ax.set_xlabel("Jaccard similarity (1.0 = identical term sets)")
    ax.set_ylabel("Number of matched configurations")
    ax.set_title(
        "Per-configuration term-set agreement: raw µV vs global z-score\n"
        f"Identical: {n_identical}/{len(scores)}  |  Completely different: {n_zero}/{len(scores)}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_rmse_comparison(
    gsz_sim: list[dict],
    path: Path,
) -> None:
    """Box plot of RMSE in z-score units for global z-score sweep, by degree."""
    gsz_grid = read_grid(GSZ_DIR / "raw_grid_merged.csv")
    idx_to_deg = {r["configuration_index"]: int(r["degree"]) for r in gsz_grid}

    rmse_by_deg: dict[int, list[float]] = collections.defaultdict(list)
    for cfg_idx, vals in rmse_by_config(gsz_sim).items():
        deg = idx_to_deg.get(cfg_idx)
        if deg is not None:
            rmse_by_deg[deg].extend(vals)

    if not rmse_by_deg:
        print("  [warn] no RMSE data for globalzscore simulations — skipping plot 05")
        return

    degrees = sorted(rmse_by_deg)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [rmse_by_deg[d] for d in degrees]
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
    for patch, deg in zip(bp["boxes"], degrees):
        patch.set_facecolor(COLORS["gsz"])
        patch.set_alpha(0.7)

    for i, (deg, vals) in enumerate(zip(degrees, data), start=1):
        med = statistics.median(vals)
        ax.text(i, med, f"{med:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(1, len(degrees) + 1))
    ax.set_xticklabels([f"Degree {d}" for d in degrees])
    ax.set_ylabel("x₀ RMSE (z-score units)")
    ax.set_title(
        "Simulation RMSE by degree — global z-score sweep\n"
        "(Raw µV sweep has no RMSE in old status format)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def plot_paired_term_counts(pairs: list[tuple[dict, dict]], path: Path) -> None:
    """Scatter: nonzero term count per matched config — raw vs z-score."""
    by_deg: dict[int, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
    for raw_r, gsz_r in pairs:
        deg = int(raw_r["degree"])
        by_deg[deg][0].append(int(raw_r["nonzero_terms"]))
        by_deg[deg][1].append(int(gsz_r["nonzero_terms"]))

    deg_colors = {1: "#2563eb", 2: "#d97706", 3: "#7c3aed"}
    fig, ax = plt.subplots(figsize=(7, 6))

    for deg, (raw_counts, gsz_counts) in sorted(by_deg.items()):
        ax.scatter(raw_counts, gsz_counts,
                   color=deg_colors.get(deg, "grey"), alpha=0.55, s=20,
                   label=f"Degree {deg}")

    lim_max = max(
        max(c for _, (rc, _) in by_deg.items() for c in rc),
        max(c for _, (_, gc) in by_deg.items() for c in gc),
    ) + 2
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("Nonzero terms — raw µV")
    ax.set_ylabel("Nonzero terms — global z-score")
    ax.set_title(
        "Paired term count per configuration\n"
        "Points above diagonal: z-score kept MORE terms"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, path)


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(
    pairs: list[tuple[dict, dict]],
    raw_sim: list[dict],
    gsz_sim: list[dict],
    out: Path,
) -> None:
    """Print and save a structured comparison report."""
    lines = ["=" * 68, "NORMALISATION COMPARISON REPORT: raw µV vs global z-score",
             "=" * 68]

    # Fitting
    lines.append("\n── FITTING ──────────────────────────────────────────────────────")
    lines.append(f"  Matched configurations: {len(pairs)}")

    raw_utils = [float(r["nonzero_terms"]) / max(int(r["possible_terms"]), 1) * 100
                 for r, _ in pairs]
    gsz_utils = [float(g["nonzero_terms"]) / max(int(g["possible_terms"]), 1) * 100
                 for _, g in pairs]
    lines.append(f"  Mean utilisation — raw: {statistics.mean(raw_utils):.1f}%  "
                 f"z-score: {statistics.mean(gsz_utils):.1f}%")

    jaccard_scores = [jaccard(surviving_terms(r), surviving_terms(g)) for r, g in pairs]
    lines.append(f"  Median Jaccard (term-set agreement): {statistics.median(jaccard_scores):.3f}")
    lines.append(f"  Identical term sets: "
                 f"{sum(1 for s in jaccard_scores if s >= 0.999)}/{len(jaccard_scores)}")
    lines.append(f"  Completely different (J=0): "
                 f"{sum(1 for s in jaccard_scores if s < 0.01)}/{len(jaccard_scores)}")

    # Coefficient magnitudes
    raw_mags = [m for r, _ in pairs for m in nonzero_coeffs(r)]
    gsz_mags = [m for _, g in pairs for m in nonzero_coeffs(g)]
    if raw_mags and gsz_mags:
        lines.append(f"\n  Median |coeff| — raw: {statistics.median(raw_mags):.3g} µV/s  "
                     f"z-score: {statistics.median(gsz_mags):.3g} (z/s)")
        lines.append(f"  Scale ratio (raw/gsz): {statistics.median(raw_mags)/statistics.median(gsz_mags):.1f}x")

    # Simulation
    lines.append("\n── SIMULATION ───────────────────────────────────────────────────")
    def success_rate(sim_rows):
        if not sim_rows:
            return float("nan"), 0
        n_ok = sum(1 for r in sim_rows if r["simulation_status"] == "success")
        return 100 * n_ok / len(sim_rows), len(sim_rows)

    raw_rate, raw_n = success_rate(raw_sim)
    gsz_rate, gsz_n = success_rate(gsz_sim)
    lines.append(f"  Raw µV:        {raw_rate:.1f}% success  (n={raw_n} trials)")
    lines.append(f"  Global z-score: {gsz_rate:.1f}% success  (n={gsz_n} trials)")

    gsz_rmses = [v for vals in rmse_by_config(gsz_sim).values() for v in vals]
    if gsz_rmses:
        lines.append(f"  Global z-score median RMSE: {statistics.median(gsz_rmses):.4f} z-score units")
        lines.append("  (Raw µV RMSE not available — old simulation format lacks RMSE field)")

    # Key findings
    lines.append("\n── KEY FINDINGS ─────────────────────────────────────────────────")
    med_j = statistics.median(jaccard_scores)
    if med_j > 0.95:
        lines.append(
            f"\n1. EQUATIONS ESSENTIALLY IDENTICAL (median Jaccard={med_j:.3f}): "
            "Z-scoring at t=0.1 does not change which terms survive. "
            "The threshold is too small relative to both raw and z-score coefficient scales."
        )
    elif med_j > 0.7:
        lines.append(
            f"\n1. EQUATIONS MOSTLY SIMILAR (median Jaccard={med_j:.3f}): "
            "Z-scoring changes some terms at the margin but core structure is preserved."
        )
    else:
        lines.append(
            f"\n1. EQUATIONS SUBSTANTIALLY DIFFERENT (median Jaccard={med_j:.3f}): "
            "Z-scoring meaningfully changes which terms survive at this threshold."
        )

    diff = raw_rate - gsz_rate
    lines.append(
        f"\n2. SIMULATION STABILITY: raw={raw_rate:.1f}% vs z-score={gsz_rate:.1f}% "
        f"({'raw more stable' if diff > 2 else 'z-score more stable' if diff < -2 else 'comparable'}, "
        f"Δ={abs(diff):.1f}%)."
    )

    lines.append(
        "\n3. COEFFICIENT SCALE: z-scoring brings coefficients to O(1) units (z/s) "
        "vs O(100)–O(1000) for raw µV. This means a future threshold sweep on z-scored "
        "data should use t=0.01–10 rather than t=0.1–10000."
    )

    lines.append("\n" + "=" * 68)
    report = "\n".join(lines)
    print(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"\n  saved: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the normalisation comparison."""
    parser = argparse.ArgumentParser(
        description="Compare raw-µV vs global-z-score normalisation at t=0.1."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--gsz-dir", type=Path, default=GSZ_DIR)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/pysindy/normalization_comparison",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {args.output_dir}\n")

    print("Loading grids...")
    raw_grid = read_grid(args.raw_dir / "raw_grid_merged.csv")
    gsz_grid = read_grid(args.gsz_dir / "raw_grid_merged.csv")

    print("Loading simulation status...")
    raw_sim = load_sim_status(args.raw_dir)
    gsz_sim = load_sim_status(args.gsz_dir)

    print("Matching configurations...")
    pairs = match_grids(raw_grid, gsz_grid)
    print(f"  {len(pairs)} matched pairs")

    if not pairs:
        print("ERROR: no matched configurations found — check that both sweeps used the same grid.")
        sys.exit(1)

    print("\nGenerating plots...")
    plot_success_rate(raw_sim, gsz_sim, args.output_dir / "01_success_rate.png")
    plot_utilization(pairs, args.output_dir / "02_utilization_by_degree.png")
    plot_term_types(pairs, args.output_dir / "03_term_type_breakdown.png")
    plot_coefficient_magnitudes(pairs, args.output_dir / "04_coefficient_magnitudes.png")
    plot_rmse_comparison(gsz_sim, args.output_dir / "05_rmse_zscore.png")
    plot_jaccard_per_config(pairs, args.output_dir / "06_jaccard_per_config.png")
    plot_paired_term_counts(pairs, args.output_dir / "07_paired_term_counts.png")

    print_report(pairs, raw_sim, gsz_sim, args.output_dir / "normalization_comparison_report.txt")


if __name__ == "__main__":
    main()
