# LFP Symbolic Regression

This project analyzes local field potential (LFP) recordings and builds
dynamical models with delay embeddings, PySINDy, and PySR.


## Project Structure

```text
lfp-symreg/
├── load_data/
│   └── convert.py
├── raw_data/
│   └── trialdata_v03_buzz_20231106_pre-0.100_post0.100.mat
├── scripts/
│   ├── channel_analysis/
│   │   ├── analyze_single_trial.py
│   │   ├── analyze_all_trials.py
│   │   └── compare_fixation_trials.py
│   ├── filter/
│   │   └── get_fixed_trials.py
│   ├── pysindy/
│   │   ├── lfp_sindy.py
│   │   ├── experiment_delays.py
│   │   ├── lfp_sindy_fixation_only.py
│   │   ├── lfp_sindy_exclude_fixation.py
│   │   └── tune_fixation_sindy.py
│   ├── pysr/
│   │   └── lfp_sr.py
│   └── graph_lfp.m
├── outputs/
│   ├── channel_analysis/
│   ├── pysindy/
│   └── old outputs/
└── README.md
```

`raw_data/` and `outputs/` are local working directories and are ignored by
Git.


