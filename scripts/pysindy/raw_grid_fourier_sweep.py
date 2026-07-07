"""Fitting-only raw-grid sweep using PySINDy's Fourier feature library.

Runs the same trial split and preprocessing as ``raw_grid_sweep.py`` but
replaces the polynomial library with a Fourier library parameterised by
``n_frequencies`` instead of ``degree``.

Grid (216 total): n_frequencies ∈ {1,2,3} × n_delays ∈ {2,4,6,8}
                  × delay ∈ {1,2,5} × smooth_window ∈ {0,5,9}
                  × lowpass ∈ {35,80}
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PYSINDY_SCRIPTS = SCRIPTS / "pysindy"
for path in (ROOT, SCRIPTS, PYSINDY_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sweep_io import parse_optional_float_list, prepare_lfp_trials
from load_data.convert import LFP_AMPLITUDE_UNIT, MAT_FILE
from load_data.preprocessing import (
    GlobalZScoreStats,
    apply_global_zscore,
    channel_traces,
    compute_global_zscore_stats,
)
from models.sindy import (
    count_terms,
    delay_embed_trajectories,
    equation_text,
    maximum_fourier_terms,
)
from pipeline_utils import parse_int_list

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pysindy" / "raw_grid_fourier"
DEFAULT_RESULTS = DEFAULT_OUTPUT_DIR / "raw_grid_fourier.csv"
DEFAULT_EQUATIONS = DEFAULT_OUTPUT_DIR / "raw_grid_fourier_equations.txt"
DEFAULT_METADATA = DEFAULT_OUTPUT_DIR / "run_metadata.json"

FIELDNAMES = [
    "configuration_index",
    "lowpass_hz",
    "global_zscore_mean",
    "global_zscore_std",
    "stlsq_threshold",
    "n_frequencies",
    "n_delays",
    "delay_samples",
    "delay_ms",
    "embedding_span_ms",
    "smooth_window_samples",
    "smooth_window_ms",
    "derivative_method",
    "fit_status",
    "fit_failure_reason",
    "nonzero_terms",
    "possible_terms",
    "term_utilization_percent",
    "fit_runtime_s",
    "feature_names_json",
    "coefficients_json",
    "equations",
]


def initialize_csv(path: Path) -> None:
    """Create an empty CSV with the Fourier-grid header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()


def append_result(path: Path, row: dict[str, object]) -> None:
    """Append one fitted configuration to the running CSV."""
    with path.open("a", newline="") as file:
        csv.DictWriter(file, fieldnames=FIELDNAMES).writerow(row)


def write_equations(path: Path, rows: list[dict[str, object]]) -> None:
    """Write human-readable equations for every successful fit."""
    sections = []
    for row in rows:
        if row["fit_status"] != "success":
            continue
        sections.append(
            f"Configuration {row['configuration_index']}: "
            f"lowpass={row['lowpass_hz']} Hz, n_frequencies={row['n_frequencies']}, "
            f"n_delays={row['n_delays']}, delay={row['delay_samples']} samples, "
            f"smooth={row['smooth_window_samples']} samples\n{row['equations']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + ("\n" if sections else ""))


def derivative_method_label(smooth_window: int) -> str:
    """Return a text description of the derivative method."""
    if smooth_window == 0:
        return "finite_difference"
    return "smoothed_finite_difference_savgol_order_3"


def build_metadata(
    args: argparse.Namespace,
    data,
    train_ids: list[int],
    test_ids: list[int],
    lowpass_values: list[float | None],
    n_freq_values: list[int],
    n_delay_values: list[int],
    delay_values: list[int],
    smooth_values: list[int],
) -> dict[str, object]:
    """Build the run manifest for one Fourier-grid Slurm task."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "library_type": "fourier",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "source": "lfp",
        "mat_file": str(args.mat_file),
        "session": str(data.sessname),
        "trial_type": args.trial_type,
        "validity_rule": (
            "is_fixation_trial & goodFix"
            if args.trial_type == "fixation"
            else "is_sequence_trial & goodFix_wholeseq"
        ),
        "channel": args.channel,
        "signal_units": LFP_AMPLITUDE_UNIT,
        "raw_sampling_hz": data.fs,
        "downsample_factor": args.downsample,
        "processed_sampling_hz": data.fs / args.downsample,
        "preprocessing": {
            "demean_each_trial": True,
            "lowpass_filter": "fourth-order Butterworth, zero-phase sosfiltfilt",
            "normalization": "global_zscore" if args.global_zscore else "none",
            "software_highpass": "none",
            "full_stored_trial": True,
        },
        "split": {
            "method": "random whole-trial split",
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "train_trial_ids": train_ids,
            "test_trial_ids": test_ids,
        },
        "fixed_model_settings": {
            "optimizer": "STLSQ",
            "threshold": args.threshold,
            "alpha": 0.05,
            "normalize_columns": args.normalize_columns,
            "savgol_polyorder": 3,
            "simulation_performed": False,
        },
        "grid": {
            "lowpass_hz": lowpass_values,
            "n_frequencies": n_freq_values,
            "n_delays": n_delay_values,
            "delay_samples": delay_values,
            "smooth_window_samples": smooth_values,
        },
        "expected_configurations": (
            len(lowpass_values)
            * len(n_freq_values)
            * len(n_delay_values)
            * len(delay_values)
            * len(smooth_values)
        ),
    }


def run_raw_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    """Fit the Fourier raw-grid without simulation."""
    try:
        import pysindy as ps
    except ImportError as exc:
        raise ImportError("PySINDy is required for the Fourier sweep.") from exc

    lowpass_values = parse_optional_float_list(args.lowpass_list)
    n_freq_values = parse_int_list(args.n_frequencies_list)
    n_delay_values = parse_int_list(args.n_delays_list)
    delay_values = parse_int_list(args.delay_list)
    smooth_values = parse_int_list(args.smooth_window_list)
    if any(w not in {0, 5, 9} for w in smooth_values):
        raise ValueError("Approved smoothing windows are 0, 5, and 9 samples.")

    data, train_ids, test_ids = prepare_lfp_trials(args)
    dt = args.downsample / data.fs

    train_raw_by_lowpass: dict[float | None, list] = {}
    global_stats: dict[float | None, GlobalZScoreStats] = {}
    for lowpass_hz in lowpass_values:
        raw = channel_traces(
            data,
            channel=args.channel,
            trials=train_ids,
            downsample=args.downsample,
            lowpass_hz=lowpass_hz,
            normalize="none",
        )
        if args.global_zscore:
            stats = compute_global_zscore_stats(raw, channel=args.channel)
            global_stats[lowpass_hz] = stats
            raw = apply_global_zscore(raw, stats)
        train_raw_by_lowpass[lowpass_hz] = raw

    metadata = build_metadata(
        args, data, train_ids, test_ids,
        lowpass_values, n_freq_values, n_delay_values, delay_values, smooth_values,
    )
    if args.global_zscore:
        metadata["global_zscore"] = {
            str(lp if lp is not None else "none"): {"mean": s.mean, "std": s.std}
            for lp, s in global_stats.items()
        }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
    initialize_csv(args.out_csv)

    rows = []
    total = int(metadata["expected_configurations"])
    configuration_index = 0

    for lowpass_hz in lowpass_values:
        train_raw = train_raw_by_lowpass[lowpass_hz]
        lp_stats = global_stats.get(lowpass_hz)
        for n_frequencies, n_delays, delay, smooth_window in itertools.product(
            n_freq_values, n_delay_values, delay_values, smooth_values,
        ):
            configuration_index += 1
            started = time.perf_counter()
            possible_terms = maximum_fourier_terms(n_delays, n_frequencies)
            row: dict[str, object] = {
                "configuration_index": configuration_index,
                "lowpass_hz": lowpass_hz if lowpass_hz is not None else "none",
                "global_zscore_mean": lp_stats.mean if lp_stats is not None else "",
                "global_zscore_std": lp_stats.std if lp_stats is not None else "",
                "stlsq_threshold": args.threshold,
                "n_frequencies": n_frequencies,
                "n_delays": n_delays,
                "delay_samples": delay,
                "delay_ms": 1000 * delay * dt,
                "embedding_span_ms": 1000 * (n_delays - 1) * delay * dt,
                "smooth_window_samples": smooth_window,
                "smooth_window_ms": 1000 * smooth_window * dt,
                "derivative_method": derivative_method_label(smooth_window),
                "fit_status": "failed",
                "fit_failure_reason": "",
                "nonzero_terms": 0,
                "possible_terms": possible_terms,
                "term_utilization_percent": "",
                "fit_runtime_s": float("nan"),
                "feature_names_json": "",
                "coefficients_json": "",
                "equations": "",
            }
            try:
                train = delay_embed_trajectories(train_raw, n_delays=n_delays, delay=delay)
                kwargs: dict = {}
                if smooth_window and smooth_window > 2:
                    window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
                    kwargs["differentiation_method"] = ps.SmoothedFiniteDifference(
                        smoother_kws={"window_length": window, "polyorder": 3}
                    )
                model = ps.SINDy(
                    optimizer=ps.STLSQ(
                        threshold=args.threshold,
                        alpha=0.05,
                        normalize_columns=args.normalize_columns,
                    ),
                    feature_library=ps.FourierLibrary(n_frequencies=n_frequencies),
                    **kwargs,
                )
                try:
                    model.fit(train, t=dt)
                except TypeError:
                    model.fit(train, t=dt, multiple_trajectories=True)

                row["fit_status"] = "success"
                row["nonzero_terms"] = count_terms(model)
                row["term_utilization_percent"] = (
                    100 * int(row["nonzero_terms"]) / possible_terms
                )
                row["feature_names_json"] = json.dumps(model.get_feature_names())
                row["coefficients_json"] = json.dumps(model.coefficients().tolist())
                row["equations"] = equation_text(model)
            except Exception as exc:
                row["fit_failure_reason"] = str(exc)
            row["fit_runtime_s"] = time.perf_counter() - started
            rows.append(row)
            append_result(args.out_csv, row)
            print(
                f"[{configuration_index}/{total}] status={row['fit_status']} "
                f"lowpass={row['lowpass_hz']} n_freq={n_frequencies} delays={n_delays} "
                f"delay={delay} smooth={smooth_window} terms={row['nonzero_terms']} "
                f"runtime={float(row['fit_runtime_s']):.1f}s",
                flush=True,
            )

    write_equations(args.equations_out, rows)
    return rows


def main() -> None:
    """CLI entry point for the Fourier raw-grid fitting sweep."""
    parser = argparse.ArgumentParser(
        description="Fit a Fourier PySINDy grid without simulation diagnostics."
    )
    parser.add_argument("--mat-file", type=Path, default=MAT_FILE)
    parser.add_argument("--trial-type", choices=("fixation", "non_fixation"), default="fixation")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--lowpass-list", default="35,80", help="Cutoffs in Hz.")
    parser.add_argument("--n-frequencies-list", default="1,2,3",
                        help="Fourier harmonic counts (replaces --degree-list).")
    parser.add_argument("--n-delays-list", default="2,4,6,8")
    parser.add_argument("--delay-list", default="1,2,5", help="Processed samples.")
    parser.add_argument("--smooth-window-list", default="0,5,9", help="Processed samples.")
    parser.add_argument("--threshold", type=float, default=1000.0,
                        help="STLSQ coefficient threshold.")
    parser.add_argument(
        "--normalize-columns",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--global-zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply global z-score (μ,σ from pooled training samples) before fitting. "
            "Required for well-conditioned Fourier basis functions. Default: True."
        ),
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--equations-out", type=Path, default=DEFAULT_EQUATIONS)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    if args.threshold < 0:
        parser.error("--threshold must be nonnegative.")

    rows = run_raw_grid(args)
    successful = sum(row["fit_status"] == "success" for row in rows)
    print(f"saved: {args.out_csv}")
    print(f"saved: {args.equations_out}")
    print(f"saved: {args.metadata_out}")
    print(f"successful fits: {successful}/{len(rows)}")


if __name__ == "__main__":
    main()
