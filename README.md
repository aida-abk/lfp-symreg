# LFP Symbolic Regression

This project analyzes local field potential (LFP) recordings and builds
dynamical models with delay embeddings, PySINDy, and PySR.


## Project Structure

```text
lfp-symreg/
├── AGENTS.md
├── README.md
├── archive/
│   /unused scripts
├── docs/
│   ├── project_context.md
│   └── repo_walkthrough.md
├── load_data/
│   ├── convert.py
│   ├── preprocessing.py
│   ├── synthetic.py
│   └── trial_selection.py
├── models/
│   ├── pysr.py
│   ├── sindy.py
│   └── validation.py
├── outputs/
│   ├── channel_analysis/
│   ├── filter/
│   ├── pysindy/
│   └── pysr/
├── preprocessing/
│   └── get_good_channels.py
├── raw_data/
│   └── trialdata_v03_buzz_20231106_pre-0.100_post0.100.mat
├── scripts/
│   ├── channel_analysis/
│   │   ├── analyze_all_trials.py
│   │   ├── analyze_single_trial.py
│   │   └── compare_fixation_trials.py
│   ├── filter/
│   │   ├── fixation_filter.py
│   │   └── visualize_80hz.py
│   ├── pysindy/
│   │   ├── exploration_sweep.py
│   │   ├── lfp_sindy.py
│   │   ├── pipeline_utils.py
│   │   └── simulate_fixation_sweep_model.py
│   └── pysr/
└── README.md
```

`raw_data/` contains local data files and is ignored by Git.

