# Experimental

Preliminary/future work code -- **not** formally evaluated as part of the results reported in the
thesis, and **not** part of the main reproducible pipeline (`src/fred_forecast/` + `scripts/`).

## What is here

- **`neurosym_cbf_pinball/`** — NeuroSym-CBF variant with pinball loss for probabilistic
  forecasting (0.5 and 0.9 quantiles). The architecture and training loop are
  migrated and verified with synthetic data (same fidelity level as the rest of the
  repo), but they were not evaluated in depth -- they were left as future work.
  It reuses the shared and already validated components from `src/fred_forecast/` (dataset,
  patch encoder, TopKSAE, base metrics) -- only the parts specific to the probabilistic
  variant live here (quantile-wise vectorized symbolic head and FiLM, pinball/
  coverage/interval-width metrics).
- **`chronos2_pinball/`** — zero-shot counterpart to the above: evaluates Chronos-2 in
  probabilistic mode (0.5 and 0.9 quantiles via `torch.quantile` over the samples returned by the
  pipeline), with the same pinball/coverage/width metrics as `neurosym_cbf_pinball`
  (reused from there, not reimplemented). Intended specifically for comparison against
  NeuroSym-CBF (pinball) -- see `experimental/evaluate_chronos2_pinball.py`.

## Why it is separate

So that anyone browsing the repo can immediately tell what is validated thesis work
(`src/fred_forecast/`) and what is an open/unfinished line. The code here is not
installed as part of the `fred-forecast` package (it is not under `src/`) and it has no numbered
script in `scripts/` -- it is run separately, see `experimental/train_neurosym_cbf_pinball.py`.
