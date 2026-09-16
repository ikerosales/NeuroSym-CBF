# Multimodal-FinanceIndicators-Agents

This repository accompanies my undergraduate thesis. The starting point is that every time series carries meaning beyond its numbers. A name that says what is being measured, the units, the sampling frequency, the categorical attributes, the descriptive tags: all of that is context a human analyst would use to read the data, and it is right there at inference time in most real-world databases.

Yet current foundation models for time series forecasting are trained almost exclusively on numerical history and throw that textual and categorical metadata away. What this work asks is simple to state: does closing that gap actually buy you measurable forecasting gains?

To answer it, I build a multimodal forecasting pipeline on top of FRED, the St. Louis Federal Reserve database and the largest public repository of U.S. economic indicators, where every series comes paired with its full metadata. The subset I keep amounts to nearly 200,000 monthly series,
which is already larger, in number of series, than most of the corpora these foundation models are pretrained on at the monthly frequency, so the usual scale-favours-pretraining argument does not really work against a specialised model here.

The model I propose is called **NeuroSym-CBF**. It takes both the numerical context and the T5-encoded metadata embedding, fuses them, and routes that fused representation through a Top-K
Sparse Autoencoder that plays the role of a concept bottleneck. A FiLM modulation block then uses those concepts to condition a closed-form symbolic decoder of intercept, trend and monthly seasonal components. The design choice that matters is that the metadata is only allowed to act through the
symbolic channel: it can amplify a trend, dampen a seasonal pattern or shift a baseline, but it cannot bypass the decomposition and write the forecast directly. That restriction is deliberate, because it keeps whatever the metadata changes readable in the same terms an analyst already uses. 

The interpretability is meant to be acted on, not just looked at: forecasts can be steered by editing individual concept activations at inference time, and silencing a concept produces a coherent, structurally readable change in the symbolic coefficients.

To find out whether the multimodal pathway actually pays off, I compare **NeuroSym-CBF** against a classical statistical baseline (AutoETS), against its own ablation that shares the entire neuro-symbolic backbone but drops the metadata (**Unimodal-ablation**), and against three zero-shot foundation models (Chronos-Bolt-Base, Chronos-2 and TimesFM-200M), across 194,442 series and five context lengths. NeuroSym-CBF gets the lowest mean MAE and the lowest 90th-percentile MAE at every one of the five context lengths. The gains are largest exactly where you would expect the metadata
to carry the most weight, in the short-context regimes where the numerical history is too thin to pin the series down: over the unimodal ablation they reach 7.5% in mean MAE and 7.1% in MASE at a context of 8 months, and 7.6% and 7.3% at 10 months. At the sample level it wins more often than it loses against every baseline, with a win rate of 0.61 against the unimodal ablation and 0.52 to 0.53 against the strongest zero-shot models. The takeaway is that for domains rich in structured textual context, integrating metadata looks like a more efficient path to better forecasts than just
scaling a unimodal model further.

On the practical side, the whole pipeline (downloading the data, cleaning it, generating the text embeddings, training and evaluating) runs from here with a `pip install` and a handful of scripts.
In the original thesis all of this was spread across my own machine, Kaggle and Google Colab; this repo unifies it so it can be run end to end anywhere, without depending on those platforms.

## The thesis is the source of truth

The full write-up of the work lives in `docs/paper.pdf`. If something about the design, the
terminology or the reasoning behind a particular decision is not clear from the code, the answer is
in there. It is the central document this repository accompanies, and it covers the architecture, the experimental protocol and the results in far more depth than a README can. When in doubt, read it before assuming anything.

## What is published and what is not

I have to be upfront here because of FRED's terms of use. FRED data comes with redistribution restrictions, so **this repository does not include the dataset itself** in any of its forms: neither the series values, nor the already-built databases, nor the metadata embeddings, nor the downloaded tags. Trained model weights are not published either.

Everything else is published, which is what really matters to understand and reproduce the work:

The complete code, from start to finish. It is the "how": how the dataset is downloaded from the FRED API, how it is cleaned, how the database is built, how the embeddings are generated, and how everything is trained and evaluated. That code is entirely my own authorship.

The list of series identifiers I used, in `series_ids/series_ids.json`. This is what makes the work reproducible on the data side: with that list and a FRED API key, anyone can re-download the same set of series and rebuild the dataset with the scripts here. The 194,442 series are all the monthly-frequency series that FRED had available at the time of my extraction, which I ran on **March 11, 2026**. That date matters more than it looks. FRED is a living
database: series get revised, backfilled, and occasionally added or discontinued, so a download run
on a different day will not return byte-for-byte the same values, and even the exact set of
available monthly series can drift a little. The series_ids file pins down which series to use, but
the actual values my results are built on are anchored to that March 2026 snapshot. This is also the path FRED's terms allow anyway: I can share which series I used and the code to fetch them, not the
series already fetched.

The evaluation metrics, in `results/`. These are derived numbers (per-sample and aggregate errors),
not series values, so publishing them is fine.

## Repository structure

The layout follows that of an installable Python package. The reusable code lives under `src/`, and
`scripts/` holds the entry points, numbered according to the real order of the pipeline.

```
.
├── src/fred_forecast/       # the installable package (pip install -e .)
│   ├── data/                # FRED download, cleaning and database construction
│   ├── embeddings/          # T5 encoder that turns the metadata into vectors
│   ├── datasets/            # TimeSeriesDataset, series and embedding loaders
│   ├── metrics/             # forecasting metrics (masked MAE, sMAPE, MASE)
│   ├── models/              # patch encoder, Top-K SAE, FiLM, symbolic head, NeuroSym-CBF, ablation
│   ├── training/            # training loops (they only train and save a checkpoint)
│   └── evaluation/          # evaluation of the models and baselines, plus aggregation
├── scripts/                 # CLI entry points, numbered in pipeline order (01 download ... 11 aggregate)
├── configs/                 # per-phase configuration in YAML, nothing hardcoded in the code
├── experimental/            # preliminary pinball-loss variant, kept outside the core pipeline
├── series_ids/              # the series I used, the only piece of the dataset that is published
├── results/                 # the computed thesis metrics and the comparison tables
├── docs/                    # paper.pdf, the thesis write-up
├── data/                    # local runtime data (gitignored, empty in the repo)
├── models/                  # local trained checkpoints (gitignored, empty in the repo)
├── pyproject.toml           # package metadata and dependencies
├── LICENSE
└── README.md
```

The `data/` and `models/` folders are intentionally empty: that is where the pipeline drops the
downloaded data, the database, the embeddings and the checkpoints when you run it on your machine.
None of that gets committed to the repository.

A note on `experimental/`. It holds a variant of NeuroSym-CBF with a pinball loss for probabilistic
quantile forecasting, together with its comparison against Chronos-2 in that same mode. It is
preliminary work that I did not evaluate in depth in the thesis, so I kept it separate from the main
pipeline to make it clear what is validated work and what is an open line. It has its own README
inside.

## Getting started

You need Python 3.10 or newer. To train or evaluate the deep learning models you will want a GPU
with CUDA; the rest of the pipeline runs fine on CPU.

```
git clone xxxxxxx
cd Multimodal-FinanceIndicators-Agents
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e .
```

Then create a `.env` file in the root. The only variable you actually need is the FRED API key,
which you can get for free at https://fred.stlouisfed.org/docs/api/api_key.html:

```
FRED_API_KEY=your_api_key_here
```

That single line is enough to run everything. The paths for data, models and results are optional:
if you leave them out they fall back to `./data`, `./models` and `./results`. You only add them if
you want those folders somewhere else, in which case each one you set overrides its default:

```
DATA_DIR=/somewhere/else/data
MODELS_DIR=/somewhere/else/models
RESULTS_DIR=/somewhere/else/results
```

All the per-phase configuration (frequencies, cleaning thresholds, model hyperparameters, the
context length of each baseline) lives in the YAML files under `configs/`. There are no magic values
hidden in the code: if you want to change something, you change it there.

One heads-up about the TimesFM baseline. It uses the 1.x API of the `timesfm` package with the
`google/timesfm-1.0-200m-pytorch` checkpoint, which is the one I evaluated. That version is no
longer on PyPI (the 2.x line changed both the API and the checkpoint entirely), so to run that
particular baseline you have to install it by hand from GitHub:

```
git clone --branch v1.2.6 --depth 1 https://github.com/google-research/timesfm.git /tmp/timesfm-1.2.6
pip install --no-deps --ignore-requires-python /tmp/timesfm-1.2.6
pip install einshape utilsforecast jax
```

None of the other baselines need any of this, `pip install -e .` is enough.

## Reproducing and verifying the results end to end

There are two ways to look at this, depending on how far you want to go.

If you just want to check the numbers, you do not need to run anything. In `results/metrics/` you
have the thesis results, one file per model and context, and in `results/comparison_point.csv` you
have the full comparison table. Those numbers are exactly the ones in Table 6.1 of the thesis (I
verified them one by one), so you can cross-check the CSV against the PDF and see that they match.

If you want to rebuild everything from scratch, the scripts are numbered in the order they run. You
need the FRED API key for the first phases and a GPU for training.

```
# 1. download the FRED observations, release by release
python scripts/01_download_releases.py

# 2. download the tags of each series (for the text side)
python scripts/02_download_tags.py

# 3. clean and homogenize: common context window and NaN filtering
python scripts/03_clean_releases.py

# 4. pack the cleaned parquets into the final database
python scripts/04_build_database.py

# 5. generate the T5 embeddings of the metadata
python scripts/05_build_metadata_embeddings.py

# 6. train the two own models (they only train and save a checkpoint)
python scripts/07_train_unimodal_ablation.py
python scripts/08_train_neurosym_cbf.py

# 7. evaluate any model, own or baseline, on the test set
python scripts/10_evaluate.py --model neurosym-cbf
python scripts/10_evaluate.py --model unimodal-ablation
python scripts/10_evaluate.py --model autoets-normafter
python scripts/10_evaluate.py --model chronos-bolt
python scripts/10_evaluate.py --model chronos-2
python scripts/10_evaluate.py --model timesfm

# 8. aggregate all the results into the comparison tables
python scripts/11_aggregate_results.py
```

Every script accepts `--help`. The context length lives in `configs/` (`model_protocol.yaml` for my
own models, `baselines.yaml` for the baselines), and the work sweeps five of them: 8, 10, 15, 24 and
30 months, always with a horizon of 18. For the baselines, which are zero-shot, sweeping a context
is just re-running the evaluation at that value. For my two own models it also means retraining at
that context first, because their architecture is tied to the input length: you run the training
script again with the new context and then evaluate the resulting checkpoint. One honest caveat
about reproduction: the baselines come out identical to the ones in the thesis because they are
deterministic given the frozen model, but my two own models, if you retrain them on a different
machine, will land within about one percent of the paper numbers. That is the normal variability of
retraining a network on different hardware with the same seeds, not a bug, and it does not change
any of the conclusions.

There are also a couple of unnumbered utilities that are not part of the mandatory pipeline:
`scripts/check_data_quality.py` to diagnose duplicates and NaNs over the raw parquets, and
`scripts/export_local_dataset.py` to consolidate the dataset into a single local working file (its
output is never published, and the script itself warns about that on startup).

## Authors and credits

Anonymous Authors.

## License

The code is released under the MIT license (see the `LICENSE` file) and is entirely my own
authorship. The license covers the code, not the data: the FRED series are not included in this
repository and are governed by their own terms of use, which you can read at
https://fred.stlouisfed.org/docs/api/terms_of_use.html.