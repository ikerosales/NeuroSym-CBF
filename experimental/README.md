# Experimental

Everything that is **not** part of the main reproducible pipeline (`src/fred_forecast/` +
`scripts/`): side experiments that are evaluated end to end but kept apart so their artefacts do not
mix with the main ones (`unseen_series/`), and preliminary lines of work that were not evaluated in
depth (`pinball_loss/`, `concept_naming/`). Each one is self-contained in its own subfolder.

## `unseen_series/`

Evaluates the trained models, without retraining, on FRED series that never entered training:
monthly series the main dataset discarded, and natively weekly and daily series brought to monthly.
It has its own `data/` (ignored by git), numbered `scripts/` and `results/`; see its `README.md` for
the sets, the results table and how to reproduce it.

## `pinball_loss/`

NeuroSym-CBF variant with pinball loss for probabilistic forecasting, plus its Chronos-2 baseline.

- **`neurosym_cbf_pinball/`** -- NeuroSym-CBF variant with pinball loss (0.5 and 0.9 quantiles).
  The architecture and training loop are migrated and verified with synthetic data (same fidelity
  level as the rest of the repo), but they were not evaluated in depth -- they were left as
  future work. It reuses the shared and already validated components from `src/fred_forecast/`
  (dataset, patch encoder, TopKSAE, base metrics) -- only the parts specific to the probabilistic
  variant live here (quantile-wise vectorized symbolic head and FiLM, pinball/coverage/
  interval-width metrics).
- **`chronos2_pinball/`** -- zero-shot counterpart to the above: evaluates Chronos-2 in
  probabilistic mode (0.5 and 0.9 quantiles via `torch.quantile` over the samples returned by the
  pipeline), with the same pinball/coverage/width metrics as `neurosym_cbf_pinball` (reused from
  there, not reimplemented). Intended specifically for comparison against NeuroSym-CBF (pinball).
- Run from the root of the repo: `python experimental/pinball_loss/train_neurosym_cbf_pinball.py`
  and `python experimental/pinball_loss/evaluate_chronos2_pinball.py`. Results are written to the
  shared `<RESULTS_DIR>/metrics/ctx{N}/` (same format as the core pipeline), labeled
  `neurosym-cbf-pinball`/`chronos2-pinball` so they never collide with the core models' results.

## `concept_naming/`

Names the atoms of NeuroSym-CBF's Top-K SAE dictionary via FRED tag enrichment: for each atom,
checks which tags appear disproportionately among the series that activate it (one-sided
hypergeometric test). Run once against the canonical Kaggle checkpoint's evaluation output.

- **`concept_naming.py`** -- the enrichment logic (tag loading, hypergeometric test, per-atom and
  per-tag reports).
- **`run_concept_naming.py`** -- reads the `z` (SAE activation) array already saved by
  `scripts/10_evaluate.py` and writes the per-atom naming results as JSON, under `results/`.
- **`results/`** -- self-contained output of this experiment (JSON only).
- Run from the root of the repo:
  ```
  python experimental/concept_naming/run_concept_naming.py \
      --npz results/metrics/ctx30/npz/neurosym-cbf.npz --context-len 30
  ```

## Why it is separate

So that anyone browsing the repo can immediately tell what is the main pipeline
(`src/fred_forecast/` + `scripts/`) and what is a side experiment or an open line. The code here is
not installed as part of the `fred-forecast` package (it is not under `src/`) and it has no numbered
script in `scripts/` -- each subfolder is run directly, as shown above.
