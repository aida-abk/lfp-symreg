# Exploration Output Schema

The current fitting stage uses `scripts/pysindy/raw_grid_sweep.py`. The separate
`scripts/pysindy/exploration_sweep.py` diagnostic stage remains available for
later simulation work.

## Raw-Grid Table

The current Oscar fitting stage uses `scripts/pysindy/raw_grid_sweep.py` and
writes `outputs/pysindy/raw_grid/raw_grid.csv`. It performs no simulation or
diagnostic calculation.

| Column | Definition | Unit or range |
|---|---|---|
| `configuration_index` | Sequential identifier within one job | count |
| `lowpass_hz` | Low-pass cutoff | Hz |
| `degree` | Polynomial-library degree | count |
| `n_delays` | Number of delay coordinates | count |
| `delay_samples` | Coordinate spacing | processed samples |
| `delay_ms` | Coordinate spacing | milliseconds |
| `embedding_span_ms` | Oldest-to-current coordinate span | milliseconds |
| `smooth_window_samples` | Derivative smoothing window; zero means finite difference | processed samples |
| `smooth_window_ms` | Derivative smoothing duration | milliseconds |
| `derivative_method` | Finite or Savitzky-Golay smoothed finite difference | category |
| `fit_status` | `success` or `failed` | category |
| `fit_failure_reason` | Fitting exception text | text |
| `nonzero_terms` | Nonzero coefficients across all equations | count |
| `fit_runtime_s` | Configuration fitting wall time | seconds |
| `feature_names_json` | Full feature-library names | JSON text |
| `coefficients_json` | Full-precision coefficient matrix | JSON text |
| `equations` | Human-readable equations | text |

Fixed settings and trial IDs are stored once in `run_metadata.json`. Slurm part
files are merged with `scripts/pysindy/merge_raw_grid.py`, which verifies 216
unique configurations.

## Model Table

Default: `outputs/pysindy/exploration_sweep.csv`

One row represents one fitted parameter configuration. Cross-trial simulation
metrics correspond to the maximum evaluation horizon.

| Column | Definition | Unit or range |
|---|---|---|
| `configuration_index` | Sequential configuration identifier within the job | count |
| `source` | `lfp` or `lorenz` | category |
| `trial_type` | `fixation`, `non_fixation`, or `synthetic` | category |
| `channel` | Zero-based LFP channel index | count |
| `random_seed` | NumPy random-split seed | integer |
| `train_trials` | Number of training trajectories | count |
| `test_trials` | Number of held-out trajectories | count |
| `train_trial_ids` | Space-separated original trial identifiers | identifiers |
| `test_trial_ids` | Space-separated original trial identifiers | identifiers |
| `downsample_factor` | Keep every Nth sample after filtering | integer factor |
| `processed_sampling_hz` | Sampling frequency after downsampling | Hz |
| `sample_interval_s` | Time between processed samples | seconds |
| `signal_units` | Text description of the current amplitude scale | category |
| `lowpass_hz` | Butterworth low-pass knee, or `none` | Hz |
| `normalization` | `none`, `center`, or `zscore` | category |
| `degree` | Maximum polynomial-library degree | count |
| `n_delays` | Number of delay coordinates | count |
| `delay_samples` | Spacing between coordinates | processed samples |
| `delay_s` | Spacing between coordinates | seconds |
| `embedding_span_s` | Oldest-to-current coordinate span | seconds |
| `stlsq_threshold` | STLSQ coefficient-removal threshold | coefficient-scale dependent |
| `stlsq_alpha` | STLSQ ridge regularization strength | feature-scale dependent |
| `derivative_method` | Method used to estimate derivatives | category |
| `simulation_solver` | Numerical ODE integrator | category |
| `simulation_horizon_s` | Maximum requested simulation duration | seconds |
| `evaluation_horizons_s` | Space-separated evaluation checkpoints | seconds |
| `divergence_threshold_std` | Error threshold or `disabled` | measured-signal SD |
| `divergence_persistence_s` | Required exceedance duration or `disabled` | seconds |
| `fit_status` | `success` or `failed` | category |
| `fit_failure_reason` | Fitting exception text | text |
| `train_derivative_r2` | Derivative fit on training trajectories | unitless, normally <= 1 |
| `test_derivative_r2` | Derivative fit on held-out trajectories | unitless, normally <= 1 |
| `test_derivative_rmse` | Held-out derivative error | signal units/second |
| `nonzero_terms` | Number of nonzero coefficients across equations | count |
| `simulation_status` | `all_success`, `partial_failure`, `all_failed`, or `not_run` | category |
| `simulation_success_fraction` | Successful held-out simulations divided by all held-out trials | 0 to 1 |
| `successful_test_trials` | Held-out simulations reaching the horizon | count |
| `failed_test_trials` | Held-out simulations not reaching the horizon | count |
| `simulation_failure_reasons` | Unique numerical failure messages | text |
| `rhs_evaluations_mean` | Mean LSODA equation evaluations across held-out trials | evaluations |
| `rhs_evaluations_median` | Median LSODA equation evaluations across held-out trials | evaluations |
| `diverged_fraction` | Fraction meeting the configured divergence definition | 0 to 1 or NaN |
| `fit_runtime_s` | Model fitting wall time | seconds |
| `simulation_runtime_s` | Held-out simulation and metric wall time | seconds |
| `configuration_runtime_s` | Total configuration wall time | seconds |
| `equations` | All fitted differential equations | text |

For each metric below, the model table contains both `<metric>_mean` and
`<metric>_median`, computed over successful held-out simulations:

| Metric prefix | Definition | Unit or range |
|---|---|---|
| `trajectory_rmse` | RMSE over every delay coordinate and time point | signal units |
| `x0_rmse` | RMSE of the current-LFP coordinate | signal units |
| `x0_correlation` | Zero-lag Pearson correlation of measured and simulated `x0` | unitless, -1 to 1 |
| `max_amplitude_ratio` | Maximum absolute simulated state divided by measured state | unitless ratio |
| `collapse_std_ratio` | Simulated tail SD divided by measured `x0` SD | unitless ratio |
| `psd_similarity` | Pearson correlation of normalized Welch PSD curves | unitless, -1 to 1 |
| `distribution_ks` | Two-sample Kolmogorov-Smirnov statistic for `x0` | unitless, 0 to 1 |
| `divergence_time_s` | First sustained configured error exceedance | seconds or NaN |

`signal units` means microvolts (`uV`) when normalization is `none` or `center`,
and per-trial standard deviations when normalization is `zscore`.

## Checkpoint Table

Default: `outputs/pysindy/exploration_sweep_checkpoints.csv`

One row represents one configuration at one evaluation horizon. It contains:

- All configuration identifiers: `configuration_index`, `source`, `trial_type`,
  `channel`, `random_seed`, `lowpass_hz`, `degree`, `n_delays`, and
  `delay_samples`.
- `evaluation_horizon_s`, measured in seconds.
- `simulation_status`, `simulation_success_fraction`,
  `successful_test_trials`, `failed_test_trials`, and
  `simulation_failure_reasons`, defined as in the model table.
- Mean and median RHS evaluation counts.
- Mean and median columns for every metric listed above.
- `diverged_fraction`, defined as in the model table.

This table shows how diagnostics change with forecast duration without fitting
the model again.

## Trial-Metric Table

Default: `outputs/pysindy/exploration_sweep_trial_metrics.csv`

One row represents one held-out trial at one evaluation horizon.

| Column | Definition | Unit or range |
|---|---|---|
| configuration identifier columns | Same identifiers as the checkpoint table | mixed |
| `test_trial_index` | Position within the held-out list | zero-based count |
| `test_trial_id` | Original MATLAB or synthetic trial identifier | identifier |
| `evaluation_horizon_s` | Requested metric checkpoint | seconds |
| `reached_horizon_s` | Last time returned by the integrator | seconds |
| `rhs_evaluations` | Learned-equation evaluations requested by LSODA | count |
| `simulation_status` | `success` or `failed` at this checkpoint | category |
| `simulation_failure_reason` | Numerical failure message | text |
| `trajectory_rmse` | All-coordinate trajectory error | signal units |
| `x0_rmse` | Current-LFP trajectory error | signal units |
| `x0_correlation` | Time-aligned Pearson correlation | unitless, -1 to 1 |
| `max_amplitude_ratio` | Simulated/measured maximum amplitude | unitless ratio |
| `collapse_std_ratio` | Simulated-tail/measured SD | unitless ratio |
| `psd_similarity` | Normalized PSD correlation | unitless, -1 to 1 |
| `distribution_ks` | Amplitude-distribution KS statistic | unitless, 0 to 1 |
| `divergence_time_s` | Configured practical divergence time | seconds or NaN |
| `diverged` | Whether configured practical divergence occurred | Boolean or blank |

## Equation Text

Default: `outputs/pysindy/exploration_sweep_equations.txt`

This file contains every successfully fitted equation with its configuration
identifier. Equations are not ranked or labeled acceptable.
