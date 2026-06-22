# LFP Delay Embedding for PySINDy

This repo has a Python loader for `raw_data/trialdata_v03_buzz_20231106_pre-0.100_post0.100.mat`.

The MATLAB file contains:

- `1550` trials
- `32` LFP channels
- `500 Hz` sampling rate
- `7100` samples per trial for the checked channel

## Basic Conversion

`convert.py` loads the MATLAB struct into a Python `TrialData` object:

```bash
python3 convert.py
```

Use it from Python like this:

```python
from convert import TrialData

data = TrialData.load()
x = data.lfp_trace(trial=0, channel=0)  # one 1D LFP trace
fs = data.fs
```

## Delay Embedding

With only one channel, SINDy sees a one-dimensional time series. A delay embedding creates synthetic state variables from past values of the same signal:

```text
X(t) = [x(t), x(t - tau), x(t - 2 tau), ..., x(t - (m - 1) tau)]
```

Here `m` is the embedding dimension and `tau` is the delay.

Run:

```bash
python3 lfp_sindy.py --channel 0 --max-trials 20 --n-delays 8 --delay 5 --downsample 5 --save-npz embedded_lfp_ch0.npz
```

That creates:

- `traces`: a list-like object array with one LFP trace per trial
- `embedded_trials`: a list-like object array with one delay-embedded trajectory per trial

Trials in this file are not all the same length. That is normal for event- or behavior-aligned data, and PySINDy can use a list of trajectories with different sample counts.

## Fit PySINDy

Install PySINDy first:

```bash
python3 -m pip install pysindy scikit-learn
```

Then run:

```bash
python3 lfp_sindy.py --channel 0 --max-trials 20 --n-delays 8 --delay 5 --downsample 5 --fit
```

Recent PySINDy versions can fit multiple trajectories directly from a list of arrays, so each trial is passed as one embedded trajectory.

## Is Delay Embedding Built Into PySINDy?

PySINDy supports fitting multiple trajectories and lets you provide any state matrix you want, but delay embedding is usually prepared before calling `model.fit(...)`. The helper functions in `lfp_sindy.py` do that preprocessing explicitly.

## Practical Starting Values

Start small:

```bash
python3 lfp_sindy.py --channel 0 --max-trials 5 --n-delays 6 --delay 5 --downsample 10 --fit
```

Then increase trials or reduce downsampling only after the model runs. Using all `1550` trials at full sampling can create a very large regression problem.
