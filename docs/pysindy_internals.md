# PySINDy Internals

Reverse-engineered reference for how PySINDy actually behaves, derived by reading
the library source rather than by inferring behavior from sweep outcomes. Every
claim below carries a `file:line` citation and is pinned by an executable probe
in `scripts/pysindy/internals_probes/`.

## Scope and provenance

| Item | Value |
|---|---|
| PySINDy version | 2.1.0 |
| Upstream tag | `v2.1.0`, commit `1edf3126` (`git describe`: "feat: remove Pipeline (#668)") |
| Reference clone | a local clone of `dynamicslab/pysindy` (outside this repo, read-only) |
| Installed package | `.venv/lib/python3.14/site-packages/pysindy` |
| Drift check | Clone and installed package are byte-identical on every `.py` file except build-generated `version.py` |

Line numbers refer to paths relative to the package root, e.g. `_core.py` means
`pysindy/_core.py`. **Verify against the installed package before trusting any
citation after a version change** — the probes exist to make that cheap.

A note on locations: the `SINDy` class is defined in `_core.py`, not in a file
named `pysindy.py`. The critical path for one `fit()` call spans roughly 3,500
lines across seven files.

---

## Stage 1 — Input handling and trajectory isolation

### The call path

`SINDy.fit()` (`_core.py:333`) executes these steps in order:

| Line | Operation | Data shape at this point |
|---|---|---|
| `_core.py:386-390` | Validate inputs; wrap a lone array as a one-element list | list of trajectories |
| `_core.py:393` | `_process_trajectories` — **differentiate, per trajectory** | list of trajectories |
| `_core.py:405` | `concat_sample_axis(x_dot)` — concatenate targets | single array |
| `_core.py:406` | `feature_library.fit_transform(x)` — build library, still a list | list of trajectories |
| `_core.py:407` | `SampleConcatter()` — concatenate features | single array |
| `_core.py:408` | `optimizer.fit(x, x_dot)` — one regression over all trials | single array |

The essential ordering: **differentiation and the library transform both happen
per trajectory; concatenation happens only afterwards**, at lines 405 and 407.

### Claim 1.1 — Derivatives are computed per trajectory

`_process_trajectories` (`_core.py:531-538`) is a list comprehension that passes
`calc_trajectory` exactly one trajectory at a time:

```python
x, x_dot = zip(*[
    self.feature_library.calc_trajectory(self.differentiation_method, xi, ti)
    for xi, ti in _zip_like_sequence(x, t)
])
```

**Verified.** With two trajectories offset by +1000 and `dt=0.01`, a derivative
taken across the seam would register ~100,000. Observed max `|ẋ|` is **1.0**, and
each trajectory's derivative is bit-identical whether fit alone or alongside the
other.

### Claim 1.2 — The library transform is also per trajectory

`PolynomialLibrary.transform` (`polynomial_library.py:217`) iterates
`for x in x_full:`, so library features never span a trajectory boundary either.

**Verified.** Fitting `[A, B]` versus `np.concatenate([A, B])` yields coefficients
differing by up to **299**, confirming the list form is not silently a
concatenation.

### Consequence for this project

The assumption recorded in `CLAUDE.md` — *"Trials are preserved as separate
trajectories, never stacked into one matrix"* — **holds in pysindy 2.1.0**. There
is no derivative contamination at the 27 seams of a 28-trial training set. This is
now verified against source rather than asserted, and `probe01` will detect it if
a future upgrade changes the behavior.

### Claim 1.3 — The feature library is built from the *smoothed* signal

`calc_trajectory` (`feature_library/base.py:69-72`) replaces the input `x` with
the differentiation method's smoothed output before returning it:

```python
def calc_trajectory(self, diff_method, x, t):
    x_dot = diff_method(x, t=t)
    x = AxesArray(diff_method.smoothed_x_, x.axes)   # x is REPLACED
    return x, AxesArray(x_dot, x.axes)
```

The `x` returned here is what `fit()` line 406 feeds to the feature library. So
with `SmoothedFiniteDifference`, **the polynomial library is constructed from the
Savitzky-Golay-smoothed signal, not the signal passed to `fit()`**.

**Verified.** The returned array equals `diff_method.smoothed_x_` exactly. Its
deviation from the input is data-dependent: ~1e-8 on a clean sine (Savgol is
nearly an identity on smooth input) but **0.31** on noisy data with
`window_length=9, polyorder=3`.

Implications for this pipeline (which uses `smooth_window=9`):

- Fitted equations describe the dynamics of the **smoothed** state variable.
- The fit is internally consistent — both `ẋ` and `Θ(x)` derive from the same
  smoothed signal — so this is deliberate design, not a defect.
- However, simulations are compared against the **unsmoothed** measured trial, so
  model and target are not the same signal. Worth quantifying as a possible
  contributor to low trajectory correlation.
- This smoothing is applied *inside* PySINDy, on top of the 35 Hz Butterworth
  filter in `load_data/preprocessing.py`. Two smoothing stages are active, and
  only one is visible in this repository's own code.

### Probe

`scripts/pysindy/internals_probes/probe01_trial_boundaries.py` — covers claims
1.1, 1.2, and 1.3.

```bash
.venv/bin/python scripts/pysindy/internals_probes/probe01_trial_boundaries.py
```

---

## Stage 2 — Differentiation

### The call path

`SmoothedFiniteDifference` subclasses `FiniteDifference`. One call does:

```python
# smoothed_finite_difference.py:63-71
def _differentiate(self, x, t):
    x_smooth = self.smoother(x, **self.smoother_kws)      # savgol over whole trajectory
    x_dot = super()._differentiate(x_smooth, t)           # finite-difference the SMOOTHED array
    self.smoothed_x_ = x_smooth if self.save_smooth else x
    return x_dot
```

`FiniteDifference._differentiate` (`finite_difference.py:241-310`) then:

| Line | Operation |
|---|---|
| `247` | `x_dot = np.full_like(x, fill_value=np.nan)` — everything starts as NaN |
| `254-278` | Centered differences fill the interior |
| `281-308` | **If `drop_endpoints` is False**, one-sided forward/backward differences fill the boundaries |
| `309` | `self.smoothed_x_ = x` (so plain `FiniteDifference` reports its input unchanged) |

### Claim 2.1 — The `drop_endpoints` docstring is inverted

The class docstring (`finite_difference.py:38`) states: *"If False, endpoints will
be set to np.nan."* The code does the opposite — line 281 computes the boundaries
precisely when `drop_endpoints` is False.

**Verified.** With `order=2` (default, `n_stencil=3`) on a 500-sample trajectory:

| Setting | NaN count | NaN positions |
|---|---|---|
| `drop_endpoints=False` (**default**) | **0** | — |
| `drop_endpoints=True` | 2 | first and last sample |

**Trust the code, not the docstring.** By default `x_dot` contains no NaN, so the
`drop_nan_samples` call in `score()` normally removes nothing.

### Claim 2.2 — Only one sample at each end is one-sided

With `order=2`, `n_stencil=3`, so `(n_stencil-1)//2 = 1`. Comparing the output
against a pure centered difference, exactly samples `[0, 499]` differ — **1
one-sided sample at each trajectory end**, the rest centered.

### Claim 2.3 — `save_smooth=False` decouples the two sides of the regression

`save_smooth` does not change the derivative at all; it changes only what
`smoothed_x_` reports, and therefore what Stage 1 feeds to the feature library.

| `save_smooth` | `x_dot` from | library built from | consistent? |
|---|---|---|---|
| `True` (**default**) | smoothed x | smoothed x | yes |
| `False` | smoothed x | **raw x** | **no** |

**Verified.** Derivatives are identical either way; `smoothed_x_` equals the
smoothed signal under `True` and the raw input under `False`. Leave it at the
default — `False` fits `ẋ` from one signal against a library built from another.

### Claim 2.4 — Mutable default argument mutates the caller's dict

`SmoothedFiniteDifference.__init__` (`smoothed_finite_difference.py:48-61`) has
signature `smoother_kws={}` — a mutable default — and then writes into it:

```python
self.smoother_kws["window_length"] = 11   # when unspecified
self.smoother_kws["polyorder"] = 3        # when unspecified
self.smoother_kws["axis"] = 0             # always
```

**Verified.** Constructing with no arguments yields
`{'window_length': 11, 'polyorder': 3, 'axis': 0}`, and a caller-supplied dict
`{'window_length': 7, 'polyorder': 2}` comes back mutated to include `'axis': 0`.

Not currently harmful here: `models/sindy.py` builds a fresh dict literal per
call, so nothing is shared. It would bite if a `smoother_kws` dict were ever
constructed once and reused across configurations.

Note `axis=0` is correct — smoothing runs along the sample (time) axis.

### Claim 2.5 — Savitzky-Golay edge handling

`smoother` defaults to `scipy.signal.savgol_filter`, whose own default is
`mode='interp'`: rather than padding, it fits a degree-`polyorder` polynomial to
the first and last `window_length` samples and evaluates it there. So the first
and last `window_length // 2` samples of each trajectory are polynomial
extrapolations rather than centered smoothing results.

### Scale of edge effects for this pipeline

With `smooth_window=9`, `order=2`, and 28 training trials of roughly 3,500
samples each (14 s at the processed 250 Hz):

| Effect | Samples per trial | Total | Share of 98,000 |
|---|---|---|---|
| Savgol edge window | 8 (4 per end) | 224 | **0.23 %** |
| One-sided derivative | 2 (1 per end) | 56 | **0.06 %** |

Edge artifacts are negligible at this trial length. They would matter for short
trials — at 500 samples per trial the same effects reach 1.6 % and 0.4 %.

### Configuration guidance from Stage 2

- Keep `drop_endpoints` at its default (`False`): it computes endpoints and
  produces no NaN, and one-sided samples are a rounding error at this trial
  length.
- Keep `save_smooth=True`. `False` silently mismatches library input and
  derivative source.
- Remember Stage 1's consequence: with any `smooth_window > 2`, the fitted
  equations describe the **smoothed** signal. With `smooth_window` of 0 or 2,
  `models/sindy.py` falls back to plain `FiniteDifference`, whose
  `smoothed_x_ = x` (`finite_difference.py:309`) leaves the library on the
  unsmoothed signal.
- Build `smoother_kws` fresh per model rather than reusing one dict.

### Probe

`scripts/pysindy/internals_probes/probe02_differentiation.py` — covers claims
2.1 through 2.5.

```bash
.venv/bin/python scripts/pysindy/internals_probes/probe02_differentiation.py
```

## Stage 3 — Feature library

*(not yet started)*

## Stage 4 — Optimizer

### The call path

`BaseOptimizer.fit` (`optimizers/base.py`) runs five stages, in this order:

| Line | Operation | Notes |
|---|---|---|
| `229-231` | Normalize library columns if `normalize_columns` | `_normalize_features`: divide each column by its L2 norm |
| `234` | `coef_ = np.linalg.lstsq(x_normed, y)` | Plain OLS initial guess. **No ridge.** |
| `243` | `history_ = [coef_]` | Seeds history with that guess |
| `247` | `_reduce(x_normed, y)` | The STLSQ loop: ridge → threshold, iterated |
| `248` | `ind_ = np.abs(coef_) > 1e-14` | Support recomputed with a fixed epsilon, not the threshold |
| `250-251` | `if self.unbias: self._unbias(...)` | **Default `True`** |
| `254-259` | Divide `coef_`, `coef_full_`, and every `history_` entry by `feat_norms` | Return to original units |

### Claim 4.1 — The objective, exactly

`STLSQ` minimizes (per its own docstring, `optimizers/stlsq.py:18-21`, and matching
the verbose columns emitted in `_reduce`):

```
‖y − Xw‖²₂  +  α‖w‖²₂
```

where `y = Ẋ` (the estimated derivative) and `X = Θ(x)` (the library at
**measured** states). The ridge solve is
`sklearn.linear_model.ridge_regression(x, y, self.alpha)`.

**Nothing in this objective refers to a simulated trajectory, a power spectrum,
or any multi-step quantity.** SINDy performs one-step derivative regression; the
model is never evaluated on its own output during fitting. This is the source-level
reason why no hyperparameter setting optimizes trajectory or spectral agreement —
those quantities are absent from the loss.

### Claim 4.2 — `unbias=True` is the default, and it discards ridge from the final values

`STLSQ.__init__` sets `unbias=True` (`optimizers/stlsq.py:113`). `_unbias`
(`optimizers/base.py:265-274`) then refits, restricted to the selected support:

```python
coef[i, self.ind_[i]] = (
    LinearRegression(fit_intercept=False)      # unregularized least squares
    .fit(x[:, self.ind_[i]], y[:, i]).coef_
)
```

**Verified.** Toggling `unbias` leaves the support unchanged while altering
coefficient values by up to **5.37** on the probe data.

### Claim 4.3 — `alpha` selects the support; it does not set the coefficients

Because the reported coefficients come from an unregularized refit, `alpha`'s
only influence is on *which* terms survive to that refit.

**Verified, decisively.** Fitting at `alpha ∈ {0.0, 0.05, 0.5}` selected an
identical support and produced **bit-identical coefficients** (maximum pairwise
difference exactly `0`). Only `alpha=5.0` selected a different support (27 terms
versus 30) and so produced different values.

This corrects the natural reading of `alpha` as a shrinkage parameter. It is a
*selection* parameter. It also explains the earlier sweep results: `alpha=0.05`
versus `alpha=0` changed term counts dramatically under `normalize_columns=True`
(14 versus 1270) but changed nothing under `normalize_columns=False` — in the
latter case the penalty was too weak to alter selection, so the identical refit
followed.

### Claim 4.4 — Sub-threshold survivors come from `normalize_columns`, not `unbias`

A "sub-threshold survivor" is a nonzero coefficient whose reported magnitude is
below the threshold it was fit at. Isolating the two candidate causes:

| `normalize_columns` | `unbias` | nonzero | below threshold |
|---|---|---|---|
| True | True | 30 | **3** |
| True | False | 30 | **5** |
| False | True | 27 | **0** |
| False | False | 27 | **0** |

**Two distinct mechanisms exist, and they have different scope.** Replicated on
the real LFP data across **5 independent training-trial subsets** (20 of 28
trials each, threshold recalibrated per subset and regime to ~20 terms) ×
degrees {3,5,7} × alpha {0, 0.05} × `unbias` {True, False} — 120 fits, written to
`outputs/pysindy/shape_analysis/subthreshold_replication.csv`.

**Mechanism 1 — `normalize_columns=True`. General and deterministic.**
Thresholding happens in normalized-column space while `coefficients()` reports
original units (rescaled at `base.py:254-259`), so the two are not comparable.

| `normalize_columns` | degree | subsets showing survivors | median count |
|---|---|---|---|
| True | 3 | **20/20** | 52 |
| True | 5 | **20/20** | 223 |
| True | 7 | **20/20** | 239 |
| False | 3 | 0/20 | 0 |
| False | 5 | 0/20 | 0 |
| False | 7 | 5/20 | 0 |

**Mechanism 2 — `unbias=True` under severe ill-conditioning. Conditional.**
The unregularized refit (`base.py:265-274`) relocates coefficients after
selection, and they can land below the threshold they were selected at. With
`normalize_columns=False` so units agree:

| degree | alpha | `unbias` | subsets showing survivors | median count |
|---|---|---|---|---|
| 3, 5 | 0.0 and 0.05 | True or False | **0/5** | 0 |
| 7 | 0.05 | True or False | **0/5** | 0 |
| **7** | **0.0** | **True** | **5/5** | **867** |
| **7** | **0.0** | **False** | **0/5** | **0** |

Mechanism 2 reproduces perfectly (5/5 versus 0/5) but **only** at degree 7 with
`alpha=0`. It is absent at degrees 3 and 5, and absent at `alpha=0.05` even at
degree 7. It is therefore a symptom of severe ill-conditioning, not a general
property of `unbias`.

**Consequence for configuration:** `alpha=0.05` eliminates mechanism 2 entirely.
This is a concrete argument for a nonzero alpha that has nothing to do with
shrinkage — alpha keeps the selection well-conditioned enough that the unbias
refit does not move coefficients across the threshold.

This refines the note in `models/sindy.py:count_subthreshold_terms`, which
attributes the effect to support selection versus "the last refit's
coefficients". That describes mechanism 2, which is real but narrow; the
mechanism that fires under this project's default `normalize_columns=True` is
the unit mismatch.

**Scope:** all five subsets are drawn from the same 28 training trials of one
session, so this establishes stability across data samples, not across sessions
or subjects.

### Claim 4.5 — Thresholding is per-equation and uses `>=`

`_reduce` loops over targets, calling `_sparse_coefficients` separately for each
equation, so **each state equation gets its own support**. The comparison is
`np.abs(c) >= threshold`.

**Verified** (with `normalize_columns=False` so units agree): surviving terms per
equation were `[10, 7, 10]` — genuinely different supports — and the smallest
surviving coefficient was `0.534` against a threshold of `0.5`, with every
survivor at or above the threshold.

### Configuration guidance from Stage 4

- `alpha` is a **selection** knob, not a shrinkage knob. Tuning it finely is
  pointless: it does nothing until it is large enough to change which terms
  survive. Sweep it in multiplicative jumps, not increments.
- `normalize_columns=True` makes the threshold scale-fair across library columns
  but breaks the correspondence between the threshold and the reported
  coefficients. Use `normalize_columns=False` whenever you need to reason about
  coefficient magnitudes against the threshold you set.
- Threshold values are **not comparable** across `normalize_columns` settings, or
  across signal normalizations. Always record the regime alongside the value.
- `unbias=True` (the default) is generally desirable — it removes ridge bias from
  the final estimates — but it means the reported coefficients are not the ones
  that were thresholded.

### Probe

`scripts/pysindy/internals_probes/probe03_optimizer.py` — covers claims 4.2
through 4.5.

```bash
.venv/bin/python scripts/pysindy/internals_probes/probe03_optimizer.py
```

---

## Stage 4b — Every optimizer, and how its configuration defines the result

### What is shared by all optimizers

Everything in the `BaseOptimizer.fit` table above applies regardless of
optimizer: column normalization, the OLS initial guess, the `history_` seed, the
`_reduce` call, the optional unbias step, and the rescaling to original units.
Each optimizer differs **only** in its `_reduce` implementation — the search
strategy — plus its `unbias` default.

### The algorithms

| Optimizer | Strategy | What its sparsity knob does |
|---|---|---|
| `STLSQ` | Ridge solve on the current support, hard-threshold, repeat | `threshold`: direct cut, `|c| >= threshold` |
| `SR3` | Split into two vectors and alternate | `reg_weight_lam` (λ) feeds a proximal operator |
| `SSR` | Backward elimination — drop one term per iteration | `max_iter`: number of terms removed |
| `FROLS` | Forward selection by Error Reduction Ratio | `max_iter`: maximum terms admitted (default 10) |
| `StableLinearSR3` | SR3 constrained so the linear operator is negative-definite | `reg_weight_lam`, plus a stability constraint |
| `EnsembleOptimizer` | Wraps another optimizer, bagging over random subsets | delegates to the wrapped optimizer |

**`STLSQ`** minimizes `‖y − Xw‖² + α‖w‖²`. See Stage 4 above.

**`SR3`** (`optimizers/sr3.py`) minimizes

```
0.5‖y − Xw‖²  +  λ·R(u)  +  (0.5/ν)‖w − u‖²
```

It maintains **two** coefficient vectors and alternates (`_reduce`):

```python
coef_full   = cho_solve(cho, X'y + coef_sparse / nu)   # w: unregularized, pulled toward u
coef_sparse = self.prox(coef_full, reg_weight_lam * relax_coeff_nu)   # u: regularized
```

`ν` (`relax_coeff_nu`) sets how tightly `w` is tied to `u`: small `ν` couples them
strongly. Convergence is `‖u_k − u_{k−1}‖ / ν < tol` (default `tol=1e-5`), and the
**reported** `coef_` is `u` (the regularized vector); `coef_full_` holds `w`.

**`SSR`** starts from the full model and removes one term per iteration, choosing
by `criteria="coefficient_value"` (smallest coefficient) or `"model_residual"`
(the removal that least increases residual). **`FROLS`** does the opposite:
greedy forward selection, orthogonalizing candidates and taking the largest Error
Reduction Ratio. For both, `max_iter` — not a threshold — controls sparsity.

### Claim 4b.1 — `threshold` is not a shared unit across optimizers

The proximal operators (`utils/base.py`, `get_prox`) are:

| `regularizer` | operator | effective cut |
|---|---|---|
| `l0` | `x * (|x| > sqrt(2·weight))` | `sqrt(2·λ·ν)` — hard, no shrinkage |
| `l1` | `sign(x)·max(|x| − weight, 0)` | `λ·ν` — soft, **shrinks survivors** |
| `l2` | `x / (1 + 2·weight)` | **none — never produces a zero** |

where `weight = reg_weight_lam * relax_coeff_nu`.

So SR3 with L0 cuts at **`sqrt(2·λ·ν)`**, a *square root* of its λ, while STLSQ
cuts at `threshold` directly. Passing the same number to both is a unit error.

**Verified.** Matching requires `reg_weight_lam = threshold² / 2`:

| STLSQ `threshold` | STLSQ terms | SR3 `lam=threshold` | terms | SR3 `lam=threshold²/2` | terms |
|---|---|---|---|---|---|
| 0.05 | 28 | 0.05 | 27 | 0.00125 | **28** |
| 0.2 | 27 | 0.2 | 25 | 0.02 | **27** |
| 0.5 | 26 | 0.5 | 23 | 0.125 | **26** |
| 1.0 | 23 | 1.0 | 21 | 0.5 | **23** |

The matched column reproduces STLSQ exactly; `STLSQ(threshold=0.5)` and
`SR3(reg_weight_lam=0.125, regularizer="L0")` both give 26 terms and identical
R² (+0.1916). The naive column does not.

Concretely, `reg_weight_lam=20000` with `ν=1` cuts at `sqrt(2·20000) = 200`, not
20000 — a factor of 100.

**Relevance to this repository:** `models/sindy.py` passes
`reg_weight_lam=config.threshold` when `optimizer="sr3"`, so a threshold chosen
for STLSQ does not carry over. No archived sweep is affected — every row in
`outputs/pysindy/global_analysis` used STLSQ — but any future STLSQ/SR3
comparison must convert with `lam = threshold² / 2`.

### Claim 4b.2 — `regularizer="l2"` produces no sparsity at all

**Verified.** At `reg_weight_lam=0.5`: `L0` → 23 nonzero of 30, `l1` → 26, `l2` →
**30 of 30**. `_prox_l2` only rescales. Choosing `l2` for a sparsity sweep yields
a dense model at every λ.

### Claim 4b.3 — `unbias` defaults differ by optimizer

**Verified.**

| Optimizer | `unbias` default |
|---|---|
| `STLSQ` | **True** |
| `SSR` | **True** |
| `FROLS` | **True** |
| `SR3` and all variants | **False** |

Consequence: STLSQ reports coefficients refit by unregularized least squares on
the selected support, whereas SR3 reports the regularized vector `u` with its
bias intact. With `l1` that bias is real shrinkage. Swapping optimizers therefore
changes both the selection rule *and* whether the returned magnitudes are
debiased — pass `unbias=True` explicitly to SR3 for a like-for-like comparison.

### Claim 4b.4 — `StableLinearSR3` needs CVXPY, which is not installed

`stable_linear_sr3.py:145-147` routes through CVXPY whenever `reg_weight_lam` is
nonzero. **Verified:** `import cvxpy` fails in this environment. Using it with a
nonzero λ requires installing CVXPY first. Note also that it builds *stable
linear* models, so it constrains the result to linear dynamics by construction.

### Configuration guidance from Stage 4b

- Sparsity knobs are **not** interchangeable: STLSQ and SR3 use different units,
  and SSR/FROLS use a term *count* (`max_iter`) rather than a threshold.
- Convert with `lam = threshold² / 2` (L0, `ν=1`) for any STLSQ↔SR3 comparison,
  and set `unbias=True` on SR3 to match STLSQ's reporting.
- Never use `regularizer="l2"` when sparsity is the goal.
- `SSR` and `FROLS` are the natural choices when the target is "give me exactly N
  terms" — they control term count directly instead of via a scale-dependent
  threshold.

### Probe

`scripts/pysindy/internals_probes/probe04_optimizer_configs.py` — covers claims
4b.1 through 4b.4.

```bash
.venv/bin/python scripts/pysindy/internals_probes/probe04_optimizer_configs.py
```

## Stage 5 — Output and scoring

### Claim 5.1 — `score` reports derivative R², not trajectory accuracy

`SINDy.score` (`_core.py:442-500`) computes:

```python
x_dot_predict = self.predict(x, u)                       # Theta(x) . Xi
if x_dot is None:
    x, x_dot = self._process_trajectories(x, t, x_dot)   # numerical derivative
return metric(x_dot, x_dot_predict)                      # default sklearn r2_score
```

It is a **one-step** measure evaluated at measured states — the same quantity
STLSQ minimizes. It says nothing about whether integrating the model reproduces
the signal. A model can score respectably here and still diverge or decorrelate
under free-running simulation.

### Claim 5.2 — `score` and `fit` disagree about smoothing

At `_core.py:491`, `predict(x)` is called on the **raw** input, *before* line 494
reassigns `x` to the smoothed version. `predict` (`_core.py:120`) applies
`feature_library.transform(x)` directly and never calls `calc_trajectory`, so no
smoothing occurs.

The result, when a smoothing differentiation method is used:

| | library built from | target `ẋ` from |
|---|---|---|
| `fit` | **smoothed** x (Stage 1, claim 1.3) | smoothed x |
| `score` | **raw** x | smoothed x |

So `score` evaluates the model on slightly different inputs than it was fitted
on, and mildly understates the fit. Small for this project's data — smoothed and
unsmoothed traces correlate at 0.9987 (Savgol removes 3.2 % of variance from an
already 35 Hz low-passed signal) — but it is a real inconsistency, not a rounding
artifact.

### Measured result on this project's data

Twenty configurations (degrees 2–9 × thresholds 1,000–50,000) at `n_delays=2`,
`delay=5`, `smooth=9`, 35 Hz, raw signal, `normalize_columns=True`, `alpha=0.05`,
fitted on 28 trials and scored on 9 held out:

| quantity | value |
|---|---|
| derivative R², train | 0.3469 – 0.3578 |
| derivative R², held-out (median) | 0.3332 – 0.3430 |
| train − test gap | ≈ 0.015 |
| PSD similarity, same models | 0.129 – 0.134 |

Three observations:

1. **The derivative fit is modest, not strong.** R² ≈ 0.35 means roughly two
   thirds of the derivative's variance is unexplained. SINDy is capturing real
   structure, but not most of it.
2. **There is essentially no overfitting.** Train and held-out scores differ by
   about 0.015 even for a 100-term degree-9 model, so the shortfall is not a
   generalization failure.
3. **Library size buys almost nothing.** A 4-term model scores 0.3469; a 100-term
   degree-9 model scores 0.3578. **25× more terms yields ~0.011 more explained
   variance**, so the nonlinear terms contribute marginally over the linear delay
   model.

The practical reading: the free-running simulation failure is not solely an
objective/evaluation mismatch. The one-step fit is itself limited, and
integrating a model that leaves ~65 % of the derivative unexplained accumulates
error quickly. Both stages are weak, and hyperparameters move neither (R² varies
by 0.011 across a 25× range in term count, mirroring the flat 0.13 PSD).

An open question this raises: how much of the unexplained 65 % is *noise in the
derivative estimate* rather than missing dynamics. If `Ẋ` has a low
signal-to-noise ratio, R² is capped regardless of the model, and no library
change would help. Testing that requires estimating the derivative's noise floor
— for example by comparing derivative estimates across differentiation methods,
or by fitting on progressively smoothed data.

### Probe

Measured by `scripts/pysindy/score_vs_simulation.py`; outputs
`outputs/pysindy/shape_analysis/derivative_scores.csv`.

```bash
.venv/bin/python scripts/pysindy/score_vs_simulation.py
```
