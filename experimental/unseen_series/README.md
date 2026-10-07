# Unseen series

The train/test split of the main pipeline is temporal, not by series: every series contributes
sliding windows to training and its last window to the test. The test horizon is never a training
target, but every one of the 194,442 test series was seen during training in earlier windows. This
experiment evaluates the trained models, without retraining, on FRED series that never entered
training in any window.

It is kept apart from the main pipeline (its own `data/`, `scripts/` and `results/`) so its
artefacts do not mix with the main ones. The reusable code it needs lives in the package:
`fred_forecast.data.oos_dataset` and `fred_forecast.data.subperiod_resample`.

## The sets

| Set | Series | What they are | Evaluated on |
|---|---|---|---|
| `monthly` | 17,302 | Monthly series the cleaning phase of the main dataset discarded, put through the same quality bar | Each series' own last 18 months (ending between 2012 and 2026) |
| `weekly` | 2,197 (L=8), 2,190 (L=30) | Series FRED publishes at Weekly frequency, brought to monthly | July 2020 - December 2021, the horizon of the main test |
| `daily` | 8,129 (L=8), 7,940 (L=30) | Series FRED publishes at Daily frequency, brought to monthly | July 2020 - December 2021 |

Sub-monthly series are brought to monthly by taking, for every month, the last value observed on or
before its first day: always a real observation, never an average or an interpolation.

The monthly set mixes calendar periods and the other two change the native frequency, so neither
isolates transfer to unseen series on its own.

## Results

Mean scaled MAE (same metric as the main comparison). The paper reports the monthly set and the
weekly and daily sets pooled into one group; `results/unseen_series.csv` also has them separately.

| Model | Monthly, L=8 | Monthly, L=30 | Weekly + daily, L=8 | Weekly + daily, L=30 |
|---|---|---|---|---|
| NeuroSym-CBF | **0.9530** | **0.3249** | **0.7279** | **0.6109** |
| Unimodal-ablation | 0.9864 | 0.3331 | 0.8918 | 0.6629 |
| TFT (in-domain) | 0.9846 | 0.3287 | 0.8786 | 0.6520 |
| Chronos-2 | 1.0617 | 0.3540 | 0.9488 | 0.7050 |
| Chronos-Bolt | 1.1067 | 0.3691 | 0.9998 | 0.7105 |
| Moirai | 1.0260 | 0.3425 | 0.8193 | 0.6547 |
| TimesFM | 1.1148 | 0.3984 | 0.9636 | 0.7011 |
| AutoETS | 1.0418 | 0.3519 | 1.0233 | 0.7332 |

About four fifths of the pooled group are daily series, and that is where the advantage comes from:
on the weekly series alone the eight models are close and NeuroSym-CBF is not ahead (0.9409 against
0.9307 for AutoETS at L=8, and 0.5581 against 0.5339 for the TFT at L=30).

`results/metrics/ctx{L}/json/` has the aggregated metrics of every model and set, and
`results/metrics/ctx{L}/npz/` the per-series errors the table is built from.

## Reproducing it

From the repo root, with the trained models in `models/imported_from_kaggle/`:

```
# monthly: no download, only the raw monthly parquets of Phase 1 and the list of candidate series
python experimental/unseen_series/scripts/01_build_monthly_discarded.py --raw-dir <raw parquets> --pool-file <candidate_series_ids.json>

# weekly and daily: download, bring to monthly, build the databases (repeat with --set daily)
python experimental/unseen_series/scripts/02_download_subperiod.py --set weekly
python experimental/unseen_series/scripts/03_resample_to_monthly.py --set weekly
python experimental/unseen_series/scripts/04_build_subperiod_database.py --set weekly

# every set: tags and metadata embeddings (repeat with --set monthly and --set daily)
python experimental/unseen_series/scripts/05_download_tags.py --set weekly
python experimental/unseen_series/scripts/06_build_embeddings.py --set weekly

# evaluate the eight models and build the table
python experimental/unseen_series/scripts/07_evaluate_all.py
python experimental/unseen_series/scripts/08_build_table.py
```

Steps 02 and 05 need `FRED_API_KEY` in `.env`. Moirai runs in its own environment (see
`scripts/evaluate_moirai.py`).

## What is not published

`data/` under this folder is ignored by git, for the same reason as the main `data/`: it holds
series downloaded from FRED and artefacts derived from them. The code, the aggregated metrics and
the per-series errors are published.
