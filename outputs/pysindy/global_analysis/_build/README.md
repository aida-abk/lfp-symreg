# Global sweep consolidation

Consolidates every PySINDy sweep under `outputs/pysindy/global_analysis/` into one
reviewable representation for manual scientific judgment. It does **not** decide
whether a fit is scientifically meaningful — it makes visual comparison,
filtering, and annotation easy and leaves the call to you.

## What it produces

Run the build (stdlib only, no dependencies):

```bash
.venv/bin/python outputs/pysindy/global_analysis/_build/build_index.py
```

Outputs (all regenerated in place; originals never touched):

| File | Purpose |
|---|---|
| `_build/configs.csv` | One row per attempted configuration, canonical schema, every original column preserved in `raw_extra_json`. |
| `_build/trials.csv` | One row per configuration × held-out trial (simulation status + metrics). |
| `_build/associations.md` | Descriptive, non-causal observed associations. |
| `_build/dashboard_data.js` | Embedded payload for the dashboard (bulky coefficient columns excluded). |
| `../index.html` | Self-contained dashboard: faceted filter, sortable multi-metric table, figure gallery, per-config detail drawer. |
| `../annotations.csv` | **Your** manual annotations. Created empty once, then read-only to the build. |

## Viewing the dashboard

Because browsers block `file://` cross-file loads inconsistently, serve the
folder over HTTP and open it:

```bash
cd outputs/pysindy/global_analysis && python -m http.server 8777
# then open http://localhost:8777/index.html
```

Table and Gallery views both respect the sidebar filters. Click any row or card
to open the detail drawer (full parameters, equations, all simulation figures,
and any manual annotation).

## Publishing as a public static site

The full folder is ~1.7 GB and includes files the site never loads. Package a
minimal, self-contained `dist/` (only `index.html`, the payload, and the exact
figures the dashboard references) with:

```bash
.venv/bin/python outputs/pysindy/global_analysis/_build/build_index.py   # if data changed
.venv/bin/python outputs/pysindy/global_analysis/_build/make_site.py     # -> _build/dist/
```

`dist/` is ~4.8k files, largest file 3.8 MB (under Cloudflare's 25 MB/file and
20k-file limits). Static hosts do not run Jekyll, so no `.nojekyll` is needed.

Deploy the `dist/` folder to any static host (each needs a free account you log
into yourself):

- **Cloudflare Pages (CLI, most reliable for this many files):**
  ```bash
  cd outputs/pysindy/global_analysis/_build
  npx wrangler login
  npx wrangler pages deploy dist --project-name=lfp-sweeps
  ```
  Gives a `https://lfp-sweeps.pages.dev` URL.
- **Netlify (CLI):**
  ```bash
  cd outputs/pysindy/global_analysis/_build
  npx netlify deploy --dir=dist --prod
  ```
- **Drag-and-drop:** upload the `dist/` folder at `app.netlify.com/drop` or the
  Cloudflare Pages "Upload assets" flow. Simplest, but slower/flakier with
  thousands of files than the CLI.

The published site is public and read-only. To add manual annotations, edit
`annotations.csv` locally, re-run `build_index.py` then `make_site.py`, and
redeploy — viewers cannot write annotations from the site (by design).

## Status labels (computational, not scientific)

Derived from actual pipeline behavior. They flag obvious computational outcomes
and never assert scientific quality.

| Label | Meaning |
|---|---|
| `fit_failed` | `fit_status != success`; see `fit_failure_reason`. |
| `not_simulated` | Sweep/config produced no simulation. |
| `sim_all_failed` | Simulated, but no held-out trial reached the requested horizon. |
| `sim_partial` | Some held-out trials reached the horizon, some did not. |
| `sim_ok` | All held-out trials reached the horizon (computationally only). |

Diagnostic flags (warnings, non-judgmental): `derivative_nonfinite`,
`wallclock_timeout`, `lsoda_istate`, `zero_terms`, `full_utilization`,
`short_reached_horizon`.

## Annotations — how they survive rebuilds

Annotations live in `../annotations.csv`, keyed by `config_key` — a stable hash
of the configuration's parameters (sweep + lowpass + degree + delays + spacing +
smoothing + threshold + optimizer + alpha + normalization). Because the key is
content-based, re-merging or re-ordering a sweep cannot orphan an annotation.

To annotate: copy a config's `config_key` (shown in the detail drawer), add a row
to `annotations.csv` (edit by hand or in a spreadsheet), and re-run the build.
The build **only reads** this file and joins it in — it never rewrites it.

Columns: `config_key, verdict, tags, notes` (all free text except the key).

## Scope and guarantees

- Covers the 9 sweeps under `global_analysis/`. Every attempted configuration is
  included — failures are labeled, never hidden or discarded.
- No single aggregate score or ranking is computed.
- The scientific pipeline is not modified. The build reads existing
  `raw_grid_merged.csv`, `parts/*_metadata.json`, and `simulations*/status/*.csv`
  files and references existing figures by relative path.
- Re-running is idempotent and writes only into `_build/` and `../index.html`
  (plus creating `../annotations.csv` if absent).

## Adding a new sweep

Drop a sweep folder containing `raw_grid_merged.csv` (and optionally
`parts/*_metadata.json` and a `simulations*/` directory) under
`global_analysis/`, then re-run the build. Per-sweep schema differences
(threshold as a column vs. fixed in metadata; `optimizer`/`alpha` present or not;
`global_zscore_*` vs. `training_lfp_rms_uv`) are handled by fallbacks in
`build_config_records`.
