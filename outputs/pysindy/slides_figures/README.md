# Slide figures — the embedding-dimension progression

Per-trial simulation figures pulled unmodified from
`outputs/pysindy/global_analysis/_build/dist/<sweep>/simulations_pertrial/figures/`.
Nothing was re-run; these are the original sweep outputs.

In every figure: **grey** = raw µV, **blue** = low-pass filtered measurement,
**orange dashed** = simulated trajectory.

Shared across all configurations: `alpha=0.05`, `normalize_columns=True`,
`optimizer=STLSQ`, `signal_normalization=none` (raw µV), channel 0, fixation
trials, all held-out trials simulated successfully.

---

## Two properties of the fitted linear operator explain every figure

For each fitted model, take the coefficients of the linear features
(`x0, x1, ...`). That square matrix **A** is the linear part of the learned
dynamics, and its eigenvalues determine the qualitative behaviour:

* **Number of oscillation frequencies** = number of complex-conjugate pairs =
  at most `n_delays / 2`. A real N×N matrix has N eigenvalues in conjugate
  pairs, so the embedding dimension caps the spectral richness *by
  construction* — no polynomial degree or threshold can add a frequency.
* **max Re(λ)** decides whether the simulation survives. Negative → the
  trajectory decays with time constant `τ = −1/Re(λ)`. Near zero or positive →
  sustained (or slowly growing) oscillation for the whole trial.

| file prefix | sweep | idx | n_delays | delay | deg | smooth | thr | LP | terms | nonlin | RMSE | **max Re(λ)** | **behaviour** | **frequencies (Hz)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nd2_1freq_DECAYS` | deg2357_t50000 | 45 | 2 | 5 | 3 | 9 | 50k | 35 | 6 | 1 | 39.3 | **−1.142** | decays, τ=0.88 s | **5.26** |
| `nd4_2freq_alive` | deg2357_t20000 | 49 | 4 | 2 | 3 | 0 | 20k | 35 | 74 | — | 57.5 | **+0.808** | sustained/growing | **5.66, 20.67** |
| `nd6_3freq_alive` | deg2357_t20000 | 55 | 6 | 1 | 3 | 0 | 20k | 35 | 152 | — | 58.3 | **+0.375** | sustained | **5.02, 18.04, 29.65** |
| `nd8_4freq_alive_lp35` | deg2357_t50000 | 67 | 8 | 2 | 3 | 0 | 50k | 35 | 62 | 14 | 51.0 | **+0.110** | sustained | **3.48, 11.95, 20.30, 26.29** |
| `nd8_4freq_alive_lp80` | deg2357_t50000 | 212 | 8 | 2 | 3 | 5 | 50k | 80 | 47 | 11 | 57.4 | **+0.279** | sustained | **2.80, 11.08, 16.29, 21.31** |
| `nd8_4freq_alive_deg2` | threshold10000 | 180 | 8 | 5 | 2 | 9 | 10k | 80 | 150 | 93 | 50.9 | **+0.124** | sustained | **1.22, 6.37, 10.56, 13.08** |

Each row has four figures, one per held-out trial (`_trial0004`, `_trial0620`,
`_trial0622`, `_trial1088`). Trial 4 is the common reference.

---

## Reading RMSE correctly

The zero-prediction baseline — a model outputting a flat line — scores
**36.3 µV** on these trials, which is the *lowest achievable* RMSE. A model
producing the right amplitude but the wrong phase scores about
`sqrt(2) × 36.3 ≈ 51 µV`.

So on this data **RMSE ranks silence above signal**, and selecting configurations
by minimum RMSE selects the ones that decayed to zero. The band 50–62 µV
identifies models that are alive with roughly correct amplitude; that is the band
every "alive" row above occupies, while the decaying `nd2` row sits at 39.3.

Among the 699 configurations that simulated successfully, **none** fell below
38 µV — so no configuration in the archive beat the flat-line baseline on RMSE.
This is a reason to judge on spectral agreement rather than RMSE, not a reason to
prefer the decaying models.

---

## The progression this supports

1. `n_delays=2` → one frequency, and here `Re(λ) = −1.14` so it decays to nothing
   within ~2.6 s. Its 6-term equation contains a single nonlinear coefficient of
   `0.001` against linear terms of 21.9 and 40.9 — four orders of magnitude
   smaller.
2. `n_delays=4` → two frequencies, sustained.
3. `n_delays=6` → three frequencies, sustained.
4. `n_delays=8` → four frequencies, sustained.

Measured consequences of the same progression, from held-out data:

| n_delays | frequencies | derivative R² (held-out) | simulation completion |
|---|---|---|---|
| 2 | 1 | 0.33 | 95 % |
| 4 | 2 | 0.56 | 48 % |
| 6 | 3 | 0.63 | 30 % |
| 8 | 4 | 0.66 | 28 % |
| 16 | 8 | 0.70 | — |

By contrast, holding `n_delays` fixed and sweeping polynomial degree 2→9 changes
R² by 0.011, and sweeping the threshold over a 50× range changes PSD similarity
by 0.005. **Embedding dimension is the parameter that governs the result; degree
and threshold are not.**

The tension to state plainly: richer embeddings explain far more of the
derivative but simulate reliably far less often. The configurations that always
complete are the ones too simple to reproduce the signal.
