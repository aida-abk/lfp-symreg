"""Consolidate every PySINDy sweep under ``global_analysis/`` into one reviewable set.

This build is deliberately read-only with respect to the original sweep outputs.
It scans each sweep folder, normalizes the per-configuration fitting rows and the
per-trial simulation status rows onto a single canonical schema, labels obvious
computational failures and diagnostic warnings (never scientific quality), and
writes:

* ``_build/configs.csv``       -- one row per attempted configuration.
* ``_build/trials.csv``        -- one row per configuration x held-out trial.
* ``_build/associations.md``   -- descriptive, non-causal observed associations.
* ``_build/dashboard_data.js`` -- embedded data consumed by ``../index.html``.

The dashboard (``global_analysis/index.html``) and the human-editable
``global_analysis/annotations.csv`` are treated as durable: annotations are only
ever read and joined, never overwritten. Re-running this script regenerates the
derived artifacts in place and never touches the sweep folders, their merged
CSVs, status files, or figures.

Run from anywhere:

    .venv/bin/python outputs/pysindy/global_analysis/_build/build_index.py

Stdlib only -- no third-party dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import median
from typing import Any

# The build lives at <root>/global_analysis/_build/build_index.py.
BUILD_DIR = Path(__file__).resolve().parent
GLOBAL_ANALYSIS_DIR = BUILD_DIR.parent
ANNOTATIONS_CSV = GLOBAL_ANALYSIS_DIR / "annotations.csv"
DASHBOARD_HTML = GLOBAL_ANALYSIS_DIR / "index.html"

# CSV fields whose raw text is bulky; kept in configs.csv but excluded from the
# embedded dashboard payload to keep the HTML small.
BULKY_COLUMNS = {"coefficients_json", "feature_names_json"}

# Simulation failure-reason substrings mapped to compact diagnostic flags.
FAILURE_FLAGS = [
    ("derivative became non-finite", "derivative_nonfinite"),
    ("wall-clock", "wallclock_timeout"),
    ("istate in LSODA", "lsoda_istate"),
]

# Raise sys recursion / field size for very wide coefficient JSON strings.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class SweepMeta:
    """Sweep-level context read from a representative ``*_metadata.json``.

    Attributes:
        sweep_id: Folder name of the sweep (used as a stable label).
        session: Recording session identifier.
        trial_type: Trial selection category (e.g. ``fixation``).
        validity_rule: Behavioral validity expression used for trial selection.
        channel: Zero-based LFP channel index.
        seed: Random split seed.
        test_fraction: Held-out fraction requested for the split.
        n_train_trials: Number of training trajectories.
        n_test_trials: Number of held-out trajectories.
        signal_normalization: Amplitude normalization applied to the signal.
        normalize_columns: Whether feature-library columns were normalized.
        optimizer: Sparse-regression optimizer name.
        threshold: Fixed STLSQ threshold recorded in metadata (may be overridden
            per row when the sweep varies threshold as a grid column).
        alpha: STLSQ ridge parameter recorded in metadata.
        savgol_polyorder: Savitzky-Golay polynomial order for derivatives.
    """

    sweep_id: str
    session: str | None = None
    trial_type: str | None = None
    validity_rule: str | None = None
    channel: int | None = None
    seed: int | None = None
    test_fraction: float | None = None
    n_train_trials: int | None = None
    n_test_trials: int | None = None
    signal_normalization: str | None = None
    normalize_columns: bool | None = None
    optimizer: str | None = None
    threshold: float | None = None
    alpha: Any = None
    savgol_polyorder: int | None = None


@dataclass
class SimAggregate:
    """Aggregated per-trial simulation outcome for a single configuration.

    Attributes:
        n_trials: Held-out trials with a recorded simulation row.
        n_success: Trials that reached the requested horizon.
        n_failed: Trials that did not reach the requested horizon.
        failure_reasons: Sorted unique failure-reason strings.
        reached_fraction_mean: Mean reached/requested duration across all trials.
        x0_rmse_median: Median x0 RMSE across successful trials, or None.
        trajectory_rmse_median: Median trajectory RMSE across successful trials.
    """

    n_trials: int = 0
    n_success: int = 0
    n_failed: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    reached_fraction_mean: float | None = None
    x0_rmse_median: float | None = None
    trajectory_rmse_median: float | None = None


# --------------------------------------------------------------------------- #
# Discovery and metadata parsing
# --------------------------------------------------------------------------- #
def discover_sweeps(root: Path) -> list[Path]:
    """Return sweep directories under ``root`` that contain a merged grid CSV.

    Args:
        root: The ``global_analysis`` directory.

    Returns:
        Sorted list of sweep directories, excluding the ``_build`` folder.
    """
    sweeps = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "raw_grid_merged.csv").exists():
            sweeps.append(child)
    return sweeps


def _first(seq: Any, default: Any = None) -> Any:
    """Return the first element of a list-like value, else the value itself."""
    if isinstance(seq, list):
        return seq[0] if seq else default
    return seq if seq is not None else default


def read_sweep_meta(sweep_dir: Path) -> SweepMeta:
    """Read a representative metadata JSON for sweep-level context.

    Metadata is written per Slurm part; sweep-level fields (session, split,
    normalization, optimizer, threshold) are constant across parts, so the first
    metadata file is representative. Grid dimensions that vary per row are read
    from the CSV rows instead.

    Args:
        sweep_dir: A sweep directory.

    Returns:
        A populated :class:`SweepMeta` (fields fall back to None when absent).
    """
    meta = SweepMeta(sweep_id=sweep_dir.name)
    parts = sorted(sweep_dir.glob("parts/*_metadata.json"))
    if not parts:
        return meta
    with parts[0].open() as fh:
        raw = json.load(fh)

    split = raw.get("split", {})
    settings = raw.get("fixed_model_settings", {})
    preproc = raw.get("preprocessing", {})

    meta.session = raw.get("session")
    meta.trial_type = raw.get("trial_type")
    meta.validity_rule = raw.get("validity_rule")
    meta.channel = raw.get("channel")
    meta.seed = split.get("seed")
    meta.test_fraction = split.get("test_fraction")
    meta.n_train_trials = len(split.get("train_trial_ids", []) or [])
    meta.n_test_trials = len(split.get("test_trial_ids", []) or [])
    meta.signal_normalization = preproc.get("normalization")
    meta.normalize_columns = settings.get("normalize_columns")
    meta.optimizer = settings.get("optimizer")
    meta.threshold = settings.get("threshold")
    meta.alpha = _first(settings.get("alpha"))
    meta.savgol_polyorder = settings.get("savgol_polyorder")
    return meta


# --------------------------------------------------------------------------- #
# Simulation status parsing
# --------------------------------------------------------------------------- #
def find_sim_dir(sweep_dir: Path) -> Path | None:
    """Return the sweep's simulation directory, or None if it did not simulate.

    Args:
        sweep_dir: A sweep directory.

    Returns:
        The ``simulations_pertrial`` or ``simulations`` subdirectory if it
        contains a ``status`` folder, otherwise None.
    """
    for name in ("simulations_pertrial", "simulations"):
        candidate = sweep_dir / name
        if (candidate / "status").is_dir():
            return candidate
    return None


def _to_float(value: str) -> float | None:
    """Parse a float, returning None for empty or non-numeric text."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_status_rows(status_csv: Path) -> list[dict[str, Any]]:
    """Read one config's per-trial simulation status CSV into row dicts.

    Args:
        status_csv: Path to ``status/config_NNNN.csv``.

    Returns:
        List of raw per-trial dicts (empty if the file is missing).
    """
    if not status_csv.exists():
        return []
    with status_csv.open(newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate_sim(rows: list[dict[str, Any]]) -> SimAggregate:
    """Aggregate per-trial simulation rows into a single-configuration summary.

    Args:
        rows: Per-trial dicts from :func:`parse_status_rows`.

    Returns:
        A :class:`SimAggregate`.
    """
    agg = SimAggregate(n_trials=len(rows))
    reached_fracs: list[float] = []
    x0_rmses: list[float] = []
    traj_rmses: list[float] = []
    reasons: set[str] = set()

    for row in rows:
        status = (row.get("simulation_status") or "").strip()
        if status == "success":
            agg.n_success += 1
            x0 = _to_float(row.get("x0_rmse_uv"))
            traj = _to_float(row.get("trajectory_rmse_uv"))
            if x0 is not None:
                x0_rmses.append(x0)
            if traj is not None:
                traj_rmses.append(traj)
        else:
            agg.n_failed += 1
            reason = (row.get("failure_reason") or "").strip()
            if reason:
                reasons.add(reason)

        requested = _to_float(row.get("requested_duration_s"))
        reached = _to_float(row.get("reached_duration_s"))
        if requested and requested > 0 and reached is not None:
            reached_fracs.append(min(reached / requested, 1.0))

    agg.failure_reasons = sorted(reasons)
    if reached_fracs:
        agg.reached_fraction_mean = sum(reached_fracs) / len(reached_fracs)
    if x0_rmses:
        agg.x0_rmse_median = median(x0_rmses)
    if traj_rmses:
        agg.trajectory_rmse_median = median(traj_rmses)
    return agg


def index_figures(sim_dir: Path | None) -> dict[int, list[str]]:
    """Map configuration index -> relative figure paths for a sweep.

    Handles both naming schemes observed in the outputs: per-trial figures
    (``config_NNNN_trial_TTTT.png``) and per-config combined figures
    (``config_NNNN.png``).

    Args:
        sim_dir: The sweep's simulation directory, or None.

    Returns:
        Mapping from configuration index to a sorted list of paths relative to
        the ``global_analysis`` directory.
    """
    mapping: dict[int, list[str]] = defaultdict(list)
    if sim_dir is None:
        return mapping
    fig_dir = sim_dir / "figures"
    if not fig_dir.is_dir():
        return mapping
    pattern = re.compile(r"config_(\d+)(?:_trial_\d+)?\.png$")
    for png in sorted(fig_dir.glob("config_*.png")):
        match = pattern.search(png.name)
        if not match:
            continue
        idx = int(match.group(1))
        rel = png.relative_to(GLOBAL_ANALYSIS_DIR).as_posix()
        mapping[idx].append(rel)
    return mapping


# --------------------------------------------------------------------------- #
# Canonical record assembly
# --------------------------------------------------------------------------- #
def _num(value: str) -> Any:
    """Coerce a CSV string to int/float when possible, else return it as-is."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except ValueError:
        return value


def config_key(sweep_id: str, params: dict[str, Any]) -> str:
    """Return a stable content hash identifying a configuration.

    The key is derived from the sweep and the normalized parameter tuple, not
    from row position, so re-merging or re-ordering a sweep cannot orphan a
    manual annotation. Annotations join on this key.

    Args:
        sweep_id: Sweep folder name.
        params: Canonical parameter values.

    Returns:
        A 12-character hex digest.
    """
    ordered = [
        sweep_id,
        params.get("lowpass_hz"),
        params.get("degree"),
        params.get("n_delays"),
        params.get("delay_samples"),
        params.get("smooth_window_samples"),
        params.get("threshold"),
        params.get("optimizer"),
        params.get("alpha"),
        params.get("normalize_columns"),
        params.get("signal_normalization"),
    ]
    payload = "|".join("" if v is None else str(v) for v in ordered)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def classify(fit_status: str, has_sim: bool, agg: SimAggregate) -> str:
    """Return the primary status label for a configuration.

    Distinguishes fit failure, never-simulated, all-failed, partial, and ok.
    A completed simulation is labeled ``sim_ok`` only in the computational sense
    (reached the horizon) -- it makes no claim about scientific quality.

    Args:
        fit_status: ``success`` or ``failed`` from the grid CSV.
        has_sim: Whether the sweep produced any simulation for this config.
        agg: The simulation aggregate.

    Returns:
        One of ``fit_failed``, ``not_simulated``, ``sim_all_failed``,
        ``sim_partial``, ``sim_ok``.
    """
    if fit_status != "success":
        return "fit_failed"
    if not has_sim or agg.n_trials == 0:
        return "not_simulated"
    if agg.n_success == 0:
        return "sim_all_failed"
    if agg.n_failed > 0:
        return "sim_partial"
    return "sim_ok"


def diagnostic_flags(row: dict[str, Any], agg: SimAggregate) -> list[str]:
    """Return non-judgmental diagnostic warning flags for a configuration.

    Flags mark computational conditions worth a human's attention, not quality
    verdicts.

    Args:
        row: Canonical config record (already numeric-coerced).
        agg: The simulation aggregate.

    Returns:
        Sorted list of flag strings.
    """
    flags: set[str] = set()
    for reason in agg.failure_reasons:
        for needle, flag in FAILURE_FLAGS:
            if needle in reason:
                flags.add(flag)

    nonzero = row.get("nonzero_terms")
    if isinstance(nonzero, (int, float)) and nonzero == 0:
        flags.add("zero_terms")

    util = row.get("term_utilization_percent")
    if isinstance(util, (int, float)) and util >= 100:
        flags.add("full_utilization")

    if agg.reached_fraction_mean is not None and agg.reached_fraction_mean < 0.5:
        flags.add("short_reached_horizon")

    return sorted(flags)


def build_config_records(sweep_dir: Path, meta: SweepMeta) -> list[dict[str, Any]]:
    """Assemble canonical per-configuration records for one sweep.

    Args:
        sweep_dir: A sweep directory.
        meta: Sweep-level metadata.

    Returns:
        List of canonical config dicts (one per row in the merged grid CSV).
    """
    sim_dir = find_sim_dir(sweep_dir)
    has_sim_dir = sim_dir is not None
    fig_map = index_figures(sim_dir)
    status_dir = (sim_dir / "status") if sim_dir else None

    merged = sweep_dir / "raw_grid_merged.csv"
    records: list[dict[str, Any]] = []
    with merged.open(newline="") as fh:
        reader = csv.DictReader(fh)
        original_columns = reader.fieldnames or []
        for raw_row in reader:
            idx = int(float(raw_row["configuration_index"]))

            # Canonical params with per-sweep fallbacks. Threshold/optimizer/alpha
            # may be a grid column or fixed in metadata depending on the sweep.
            threshold = _num(raw_row.get("stlsq_threshold", "")) if raw_row.get(
                "stlsq_threshold"
            ) else meta.threshold
            optimizer = raw_row.get("optimizer") or meta.optimizer
            alpha = _num(raw_row.get("alpha", "")) if raw_row.get("alpha") else meta.alpha
            has_gzscore = bool((raw_row.get("global_zscore_mean") or "").strip())

            params = {
                "lowpass_hz": _num(raw_row.get("lowpass_hz", "")),
                "degree": _num(raw_row.get("degree", "")),
                "n_delays": _num(raw_row.get("n_delays", "")),
                "delay_samples": _num(raw_row.get("delay_samples", "")),
                "delay_ms": _num(raw_row.get("delay_ms", "")),
                "embedding_span_ms": _num(raw_row.get("embedding_span_ms", "")),
                "smooth_window_samples": _num(raw_row.get("smooth_window_samples", "")),
                "smooth_window_ms": _num(raw_row.get("smooth_window_ms", "")),
                "derivative_method": raw_row.get("derivative_method"),
                "threshold": threshold,
                "optimizer": optimizer,
                "alpha": alpha,
                "normalize_columns": meta.normalize_columns,
                "signal_normalization": (
                    "global_zscore" if has_gzscore else meta.signal_normalization
                ),
            }

            fit_status = (raw_row.get("fit_status") or "").strip()
            agg = SimAggregate()
            if status_dir is not None:
                agg = aggregate_sim(
                    parse_status_rows(status_dir / f"config_{idx:04d}.csv")
                )

            record: dict[str, Any] = {
                "config_key": config_key(sweep_dir.name, params),
                "sweep_id": sweep_dir.name,
                "configuration_index": idx,
                "session": meta.session,
                "trial_type": meta.trial_type,
                "channel": meta.channel,
                "seed": meta.seed,
                **params,
                "fit_status": fit_status,
                "fit_failure_reason": raw_row.get("fit_failure_reason") or "",
                "nonzero_terms": _num(raw_row.get("nonzero_terms", "")),
                "possible_terms": _num(raw_row.get("possible_terms", "")),
                "term_utilization_percent": _num(
                    raw_row.get("term_utilization_percent", "")
                ),
                "fit_runtime_s": _num(raw_row.get("fit_runtime_s", "")),
                "sim_n_trials": agg.n_trials,
                "sim_n_success": agg.n_success,
                "sim_n_failed": agg.n_failed,
                "sim_success_fraction": (
                    round(agg.n_success / agg.n_trials, 4) if agg.n_trials else None
                ),
                "sim_reached_fraction_mean": (
                    round(agg.reached_fraction_mean, 4)
                    if agg.reached_fraction_mean is not None
                    else None
                ),
                "sim_x0_rmse_median": (
                    round(agg.x0_rmse_median, 6)
                    if agg.x0_rmse_median is not None
                    else None
                ),
                "sim_trajectory_rmse_median": (
                    round(agg.trajectory_rmse_median, 6)
                    if agg.trajectory_rmse_median is not None
                    else None
                ),
                "sim_failure_reasons": " | ".join(agg.failure_reasons),
                "equations": raw_row.get("equations") or "",
                "figures": fig_map.get(idx, []),
                # Preserve every original column verbatim so nothing is lost.
                "raw_extra": {
                    col: raw_row.get(col)
                    for col in original_columns
                    if col not in {"equations"}
                },
            }
            record["status_label"] = classify(fit_status, has_sim_dir, agg)
            record["diagnostic_flags"] = diagnostic_flags(record, agg)
            records.append(record)
    return records


def build_trial_records(sweep_dir: Path, meta: SweepMeta,
                        key_by_index: dict[int, str]) -> list[dict[str, Any]]:
    """Assemble canonical per-configuration x per-trial simulation records.

    Args:
        sweep_dir: A sweep directory.
        meta: Sweep-level metadata.
        key_by_index: Map from configuration index to config_key for this sweep.

    Returns:
        List of per-trial dicts (empty when the sweep did not simulate).
    """
    sim_dir = find_sim_dir(sweep_dir)
    if sim_dir is None:
        return []
    records: list[dict[str, Any]] = []
    for status_csv in sorted((sim_dir / "status").glob("config_*.csv")):
        idx = int(re.search(r"config_(\d+)", status_csv.name).group(1))
        for row in parse_status_rows(status_csv):
            records.append({
                "config_key": key_by_index.get(idx, ""),
                "sweep_id": sweep_dir.name,
                "configuration_index": idx,
                "test_trial_id": row.get("test_trial_id"),
                "simulation_status": row.get("simulation_status"),
                "failure_reason": row.get("failure_reason") or "",
                "requested_duration_s": _to_float(row.get("requested_duration_s")),
                "reached_duration_s": _to_float(row.get("reached_duration_s")),
                "x0_rmse_uv": _to_float(row.get("x0_rmse_uv")),
                "trajectory_rmse_uv": _to_float(row.get("trajectory_rmse_uv")),
                "rhs_evaluations": _to_float(row.get("rhs_evaluations")),
            })
    return records


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
ANNOTATION_FIELDS = ["config_key", "verdict", "tags", "notes"]


def ensure_annotations_file() -> None:
    """Create an empty, documented annotations CSV if none exists.

    Never overwrites an existing file. The build only reads annotations; humans
    own this file.
    """
    if ANNOTATIONS_CSV.exists():
        return
    with ANNOTATIONS_CSV.open("w", newline="") as fh:
        fh.write(
            "# Manual review annotations. Safe to edit by hand or in a spreadsheet.\n"
            "# Join key is config_key (stable content hash from build_index.py).\n"
            "# This file is READ-ONLY to the build and is never regenerated.\n"
            "# Columns: config_key, verdict (free text), tags (space or comma\n"
            "# separated), notes (free text). Add one row per annotated config.\n"
        )
        writer = csv.DictWriter(fh, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()


def load_annotations() -> dict[str, dict[str, str]]:
    """Load manual annotations keyed by config_key.

    Returns:
        Mapping from config_key to its annotation fields (empty if none).
    """
    if not ANNOTATIONS_CSV.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with ANNOTATIONS_CSV.open(newline="") as fh:
        rows = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(rows)
    for row in reader:
        key = (row.get("config_key") or "").strip()
        if key:
            out[key] = {f: (row.get(f) or "").strip() for f in ANNOTATION_FIELDS}
    return out


# --------------------------------------------------------------------------- #
# Descriptive associations (observed, non-causal)
# --------------------------------------------------------------------------- #
def _rate(subset: list[dict], predicate) -> tuple[int, int]:
    """Return (matching, total) counts for a predicate over a subset."""
    total = len(subset)
    matching = sum(1 for r in subset if predicate(r))
    return matching, total


def write_associations(configs: list[dict[str, Any]], path: Path) -> None:
    """Write a descriptive, non-causal associations report.

    Every statement is framed as an observed association or co-occurrence. No
    causal language, no ranking, no scientific-quality judgment.

    Args:
        configs: Canonical config records.
        path: Output markdown path.
    """
    lines: list[str] = [
        "# Observed associations across consolidated sweeps",
        "",
        "These are descriptive co-occurrences computed over the consolidated "
        "configurations. They are **associations, not causal claims**, and they "
        "make no judgment about scientific validity. Use them only to decide "
        "where to look during manual review.",
        "",
    ]

    total = len(configs)
    simulated = [c for c in configs if c["status_label"].startswith("sim_")]
    label_counts = Counter(c["status_label"] for c in configs)

    lines.append(f"- Consolidated configurations: **{total}** across "
                 f"**{len({c['sweep_id'] for c in configs})}** sweeps.")
    lines.append("- Status-label breakdown:")
    for label, count in label_counts.most_common():
        lines.append(f"  - `{label}`: {count} "
                     f"({100 * count / total:.1f}% of all configs)")
    lines.append("")

    # Association: simulation reach rate by degree.
    lines.append("## Simulation reach rate by polynomial degree")
    lines.append("")
    lines.append("Among configurations that were simulated, the fraction whose "
                 "held-out trials all reached the requested horizon "
                 "(`sim_ok`), grouped by degree:")
    lines.append("")
    by_degree: dict[Any, list[dict]] = defaultdict(list)
    for c in simulated:
        by_degree[c.get("degree")].append(c)
    lines.append("| degree | simulated configs | all-reached (`sim_ok`) | rate |")
    lines.append("|---|---|---|---|")
    for degree in sorted(by_degree, key=lambda d: (d is None, d)):
        subset = by_degree[degree]
        ok, tot = _rate(subset, lambda r: r["status_label"] == "sim_ok")
        rate = f"{100 * ok / tot:.0f}%" if tot else "n/a"
        lines.append(f"| {degree} | {tot} | {ok} | {rate} |")
    lines.append("")

    # Association: non-finite derivative flag by degree.
    lines.append("## Non-finite-derivative co-occurrence by degree")
    lines.append("")
    lines.append("Fraction of simulated configurations carrying the "
                 "`derivative_nonfinite` diagnostic flag, grouped by degree:")
    lines.append("")
    lines.append("| degree | simulated configs | with `derivative_nonfinite` | rate |")
    lines.append("|---|---|---|---|")
    for degree in sorted(by_degree, key=lambda d: (d is None, d)):
        subset = by_degree[degree]
        nf, tot = _rate(subset,
                        lambda r: "derivative_nonfinite" in r["diagnostic_flags"])
        rate = f"{100 * nf / tot:.0f}%" if tot else "n/a"
        lines.append(f"| {degree} | {tot} | {nf} | {rate} |")
    lines.append("")

    # Association: reach rate by threshold.
    lines.append("## Simulation reach rate by STLSQ threshold")
    lines.append("")
    by_thr: dict[Any, list[dict]] = defaultdict(list)
    for c in simulated:
        by_thr[c.get("threshold")].append(c)
    lines.append("| threshold | simulated configs | all-reached (`sim_ok`) | rate |")
    lines.append("|---|---|---|---|")
    for thr in sorted(by_thr, key=lambda t: (t is None, t)):
        subset = by_thr[thr]
        ok, tot = _rate(subset, lambda r: r["status_label"] == "sim_ok")
        rate = f"{100 * ok / tot:.0f}%" if tot else "n/a"
        lines.append(f"| {thr} | {tot} | {ok} | {rate} |")
    lines.append("")

    # Per-sweep snapshot.
    lines.append("## Per-sweep snapshot")
    lines.append("")
    lines.append("| sweep | configs | fit_failed | not_simulated | "
                 "sim_all_failed | sim_partial | sim_ok |")
    lines.append("|---|---|---|---|---|---|---|")
    by_sweep: dict[str, list[dict]] = defaultdict(list)
    for c in configs:
        by_sweep[c["sweep_id"]].append(c)
    for sweep_id in sorted(by_sweep):
        subset = by_sweep[sweep_id]
        counts = Counter(c["status_label"] for c in subset)
        lines.append(
            f"| {sweep_id} | {len(subset)} | {counts.get('fit_failed', 0)} | "
            f"{counts.get('not_simulated', 0)} | {counts.get('sim_all_failed', 0)} | "
            f"{counts.get('sim_partial', 0)} | {counts.get('sim_ok', 0)} |"
        )
    lines.append("")
    lines.append("_Generated by `_build/build_index.py`. Do not hand-edit; "
                 "re-run the build to refresh._")

    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_configs_csv(configs: list[dict[str, Any]], path: Path) -> None:
    """Write the flat per-configuration table.

    Args:
        configs: Canonical config records.
        path: Output CSV path.
    """
    columns = [
        "config_key", "sweep_id", "configuration_index", "session", "trial_type",
        "channel", "seed", "lowpass_hz", "degree", "n_delays", "delay_samples",
        "delay_ms", "embedding_span_ms", "smooth_window_samples", "smooth_window_ms",
        "derivative_method", "threshold", "optimizer", "alpha", "normalize_columns",
        "signal_normalization", "fit_status", "fit_failure_reason", "nonzero_terms",
        "possible_terms", "term_utilization_percent", "fit_runtime_s",
        "status_label", "diagnostic_flags", "sim_n_trials", "sim_n_success",
        "sim_n_failed", "sim_success_fraction", "sim_reached_fraction_mean",
        "sim_x0_rmse_median", "sim_trajectory_rmse_median", "sim_failure_reasons",
        "n_figures", "equations", "raw_extra_json",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for c in configs:
            row = dict(c)
            row["diagnostic_flags"] = " ".join(c["diagnostic_flags"])
            row["n_figures"] = len(c["figures"])
            row["raw_extra_json"] = json.dumps(c["raw_extra"], separators=(",", ":"))
            writer.writerow(row)


def write_trials_csv(trials: list[dict[str, Any]], path: Path) -> None:
    """Write the flat per-configuration x per-trial simulation table.

    Args:
        trials: Canonical trial records.
        path: Output CSV path.
    """
    columns = [
        "config_key", "sweep_id", "configuration_index", "test_trial_id",
        "simulation_status", "failure_reason", "requested_duration_s",
        "reached_duration_s", "x0_rmse_uv", "trajectory_rmse_uv", "rhs_evaluations",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trials)


def write_dashboard_data(configs: list[dict[str, Any]],
                         annotations: dict[str, dict[str, str]],
                         path: Path) -> None:
    """Write the JS payload consumed by the dashboard.

    Bulky coefficient/feature columns are excluded to keep the file small.
    Manual annotations are merged in by config_key.

    Args:
        configs: Canonical config records.
        annotations: Manual annotations keyed by config_key.
        path: Output ``.js`` path.
    """
    payload = []
    for c in configs:
        raw_extra = {k: v for k, v in c["raw_extra"].items() if k not in BULKY_COLUMNS}
        ann = annotations.get(c["config_key"], {})
        payload.append({
            "config_key": c["config_key"],
            "sweep_id": c["sweep_id"],
            "configuration_index": c["configuration_index"],
            "lowpass_hz": c["lowpass_hz"],
            "degree": c["degree"],
            "n_delays": c["n_delays"],
            "delay_samples": c["delay_samples"],
            "smooth_window_samples": c["smooth_window_samples"],
            "derivative_method": c["derivative_method"],
            "threshold": c["threshold"],
            "optimizer": c["optimizer"],
            "alpha": c["alpha"],
            "normalize_columns": c["normalize_columns"],
            "signal_normalization": c["signal_normalization"],
            "fit_status": c["fit_status"],
            "fit_failure_reason": c["fit_failure_reason"],
            "nonzero_terms": c["nonzero_terms"],
            "possible_terms": c["possible_terms"],
            "term_utilization_percent": c["term_utilization_percent"],
            "status_label": c["status_label"],
            "diagnostic_flags": c["diagnostic_flags"],
            "sim_n_trials": c["sim_n_trials"],
            "sim_n_success": c["sim_n_success"],
            "sim_n_failed": c["sim_n_failed"],
            "sim_success_fraction": c["sim_success_fraction"],
            "sim_reached_fraction_mean": c["sim_reached_fraction_mean"],
            "sim_x0_rmse_median": c["sim_x0_rmse_median"],
            "sim_trajectory_rmse_median": c["sim_trajectory_rmse_median"],
            "sim_failure_reasons": c["sim_failure_reasons"],
            "equations": c["equations"],
            "figures": c["figures"],
            "annotation": ann,
        })
    meta = {
        "n_configs": len(payload),
        "sweeps": sorted({c["sweep_id"] for c in configs}),
        "generated_from": "outputs/pysindy/global_analysis/_build/build_index.py",
    }
    text = ("// Generated by _build/build_index.py -- do not edit by hand.\n"
            "window.DASHBOARD_META = "
            + json.dumps(meta, separators=(",", ":")) + ";\n"
            "window.DASHBOARD_DATA = "
            + json.dumps(payload, separators=(",", ":")) + ";\n")
    path.write_text(text)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    """Run the full consolidation build.

    Returns:
        Process exit code (0 on success).
    """
    sweeps = discover_sweeps(GLOBAL_ANALYSIS_DIR)
    if not sweeps:
        print(f"No sweeps with raw_grid_merged.csv found under "
              f"{GLOBAL_ANALYSIS_DIR}", file=sys.stderr)
        return 1

    all_configs: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    for sweep_dir in sweeps:
        meta = read_sweep_meta(sweep_dir)
        configs = build_config_records(sweep_dir, meta)
        key_by_index = {c["configuration_index"]: c["config_key"] for c in configs}
        trials = build_trial_records(sweep_dir, meta, key_by_index)
        all_configs.extend(configs)
        all_trials.extend(trials)
        print(f"  {sweep_dir.name}: {len(configs)} configs, {len(trials)} trial rows")

    ensure_annotations_file()
    annotations = load_annotations()

    write_configs_csv(all_configs, BUILD_DIR / "configs.csv")
    write_trials_csv(all_trials, BUILD_DIR / "trials.csv")
    write_associations(all_configs, BUILD_DIR / "associations.md")
    write_dashboard_data(all_configs, annotations, BUILD_DIR / "dashboard_data.js")

    label_counts = Counter(c["status_label"] for c in all_configs)
    print(f"\nConsolidated {len(all_configs)} configs from {len(sweeps)} sweeps.")
    print(f"Status labels: {dict(label_counts)}")
    print(f"Annotations loaded: {len(annotations)}")
    print(f"Outputs written under: {BUILD_DIR}")
    if not DASHBOARD_HTML.exists():
        print(f"NOTE: dashboard {DASHBOARD_HTML.name} not present yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

