"""Interventions on the metadata branch of NeuroSym-CBF, without retraining.

Every intervention replaces the metadata embedding a series receives and measures what happens to
the forecast and to the SAE concepts. The temporal context is never touched, so `hidden_ts` and
`coeffs_ns` are identical in every variant: the runner computes them **once per batch** and only
repeats the tail of the model (fusion -> SAE -> FiLM -> symbolic head), which is the cheap part.
That is why evaluating N interventions does not cost N full passes.

Levels, from least to most informative (each one controls for the previous one):

1. **Ablation** (`zero`, `mean`): upper bound on how much the model uses the text.
2. **Random permutation** (`permute`): the NULL. Without it a targeted swap says nothing -- you
   could not tell semantic damage from "the vector simply is not this series' own".
3. **Targeted swap by tag** (`tag_swap`): swaps metadata between series of two semantically
   different tags. Only interpretable against the null of level 2.
4. **Negative control** (`within_tagset_swap`): swaps metadata between series with the *same* set
   of tags. The text still changes (title and units differ), but the semantic content is almost
   the same, so the effect should be ~0.

Design note: the largest achievable effect is bounded by the architecture. `coeffs_ns` comes from
the series encoder and FiLM only modulates it (`coeffs_mod = γ⊙coeffs_ns + β`), so the metadata
cannot change the forecast arbitrarily. That is a property of the model, not an artefact of the
measurement.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from fred_forecast.metrics.forecasting_metrics import masked_mae_per_sample, masked_smape_per_sample

logger = logging.getLogger(__name__)

IDENTITY = "identity"
ZERO = "zero"
MEAN = "mean"
PERMUTE = "permute"

BASELINE_NAME = "baseline"


@dataclass
class Intervention:
    """One intervention on the metadata matrix.

    `perm[i] = j` means "series i receives the embedding of series j". `affected` marks the series
    whose embedding actually changes; the aggregates are computed only over them (in a targeted
    swap the vast majority of series are not involved, and averaging over all of them would dilute
    the effect until it disappeared). `affected=None` means "all of them".
    """

    name: str
    kind: str
    perm: np.ndarray | None = None
    affected: np.ndarray | None = None
    notes: str = ""

    @property
    def n_affected(self) -> int:
        return int(self.affected.sum()) if self.affected is not None else -1


# --------------------------------------------------------------------------------------------
# Intervention builders
# --------------------------------------------------------------------------------------------


def baseline_intervention() -> Intervention:
    """No intervention: the reference every other one is compared against."""
    return Intervention(name=BASELINE_NAME, kind=IDENTITY, notes="original metadata")


def ablation_interventions() -> list[Intervention]:
    """Level 1: metadata set to zero and metadata set to the corpus mean."""
    return [
        Intervention(name="meta-zero", kind=ZERO, notes="metadata embedding = 0"),
        Intervention(name="meta-mean", kind=MEAN, notes="embedding = corpus mean"),
    ]


def random_permutation_intervention(n_series: int, seed: int) -> Intervention:
    """Level 2: global random permutation of embeddings across series (the null)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_series)
    affected = perm != np.arange(n_series)
    return Intervention(
        name=f"permute-seed{seed}",
        kind=PERMUTE,
        perm=perm,
        affected=affected,
        notes=f"global random permutation (seed={seed}, {int((~affected).sum())} fixed points)",
    )


def tag_sets_for(series_ids: list[str], tags_df: pd.DataFrame) -> list[frozenset[str]]:
    """Set of tags of each series, in the same order as `series_ids`.

    Series with no row in `tags_df` (or with empty tags) get an empty set: they take part in no
    targeted swap and in no group of the negative control.
    """
    tag_map = dict(zip(tags_df["series_id"].astype(str), tags_df["tags"].fillna("")))
    return [frozenset(t for t in tag_map.get(sid, "").split("|") if t) for sid in series_ids]


def _rotate_within_groups(groups: list[list[int]], n_series: int, rng) -> tuple[np.ndarray, np.ndarray, int]:
    """Permutation that rotates the (already shuffled) members within each group.

    The rotation guarantees that no series keeps its own embedding, and that the embedding it
    receives comes from another series of the same group.
    """
    perm = np.arange(n_series)
    affected = np.zeros(n_series, dtype=bool)
    n_used = 0
    for members in groups:
        if len(members) < 2:
            continue
        members = list(members)
        rng.shuffle(members)
        perm[members] = members[1:] + members[:1]
        affected[members] = True
        n_used += 1
    return perm, affected, n_used


def tag_swap_with_matched_null(
    tag_a: str,
    tag_b: str,
    tag_sets: list[frozenset[str]],
    seed: int,
    max_pairs: int | None = None,
) -> list[Intervention]:
    """Level 3 plus its matched control. Returns the two interventions, in that order.

    - **Targeted swap**: each series with `tag_a` receives the embedding of a series with `tag_b`,
      and vice versa. Series carrying both tags are excluded (the change would not be contrastive).
    - **Matched null**: exactly the same series, exactly the same number of embeddings changed,
      but each series receives the embedding of another series **of its own group** (a CPI series
      gets the metadata of another CPI series).

    Comparing the swap against the **global null** is not valid: the series of one particular tag
    may be more sensitive to metadata than the corpus average, and the effect of the swap would
    then be confounded with that sensitivity. The matched null fixes the population and the number
    of changes, so the difference between the two isolates the only thing that varies: whether the
    new embedding comes from the opposite semantic group or from the series' own.
    """
    rng = np.random.default_rng(seed)
    group_a = [i for i, ts in enumerate(tag_sets) if tag_a in ts and tag_b not in ts]
    group_b = [i for i, ts in enumerate(tag_sets) if tag_b in ts and tag_a not in ts]

    rng.shuffle(group_a)
    rng.shuffle(group_b)
    n_pairs = min(len(group_a), len(group_b))
    if max_pairs is not None:
        n_pairs = min(n_pairs, max_pairs)
    paired_a, paired_b = group_a[:n_pairs], group_b[:n_pairs]

    n_series = len(tag_sets)
    perm = np.arange(n_series)
    affected = np.zeros(n_series, dtype=bool)
    for i, j in zip(paired_a, paired_b):
        perm[i], perm[j] = j, i
        affected[i] = affected[j] = True

    label = f"{tag_a.replace(' ', '_')}-vs-{tag_b.replace(' ', '_')}"
    swap = Intervention(
        name=f"swap-{label}",
        kind=PERMUTE,
        perm=perm,
        affected=affected,
        notes=(
            f"targeted swap {tag_a!r} <-> {tag_b!r}: {n_pairs} pairs "
            f"(available: {len(group_a)} and {len(group_b)})"
        ),
    )

    null_perm, null_affected, _ = _rotate_within_groups([paired_a, paired_b], n_series, rng)
    matched_null = Intervention(
        name=f"null-{label}",
        kind=PERMUTE,
        perm=null_perm,
        affected=affected,  # evaluated on the SAME series as the swap
        notes=f"matched null of swap-{label}: shuffled within each group",
    )
    if not np.array_equal(null_affected, affected):
        logger.warning("The matched null of %s does not cover exactly the same series", label)

    return [swap, matched_null]


def within_tagset_swap_intervention(
    tag_sets: list[frozenset[str]], seed: int, min_group_size: int = 2
) -> Intervention:
    """Level 4 (negative control): swaps embeddings between series with identical tags.

    The encoded text still changes (title and units differ between members of the group), but the
    semantic load is almost the same, so the expected effect is ~0.
    """
    rng = np.random.default_rng(seed)
    groups: dict[frozenset[str], list[int]] = defaultdict(list)
    for i, ts in enumerate(tag_sets):
        if ts:
            groups[ts].append(i)

    usable = [m for m in groups.values() if len(m) >= min_group_size]
    perm, affected, n_groups = _rotate_within_groups(usable, len(tag_sets), rng)
    return Intervention(
        name="swap-within-tagset",
        kind=PERMUTE,
        perm=perm,
        affected=affected,
        notes=f"negative control: swap within {n_groups} groups of identical tags",
    )


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


class _IndexedDataset(Dataset):
    """Wraps the test dataset so it also returns the series index of each sample.

    Needed to know which series each row of the batch belongs to, and so apply the matching
    metadata permutation to it.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        return (*self.base[i], int(self.base._ts_idx[i]))


@torch.no_grad()
def _forward_tail(model, hidden, coeffs_ns, meta):
    """Tail of the model from already computed `hidden`/`coeffs_ns`: fusion -> SAE -> FiLM -> head.

    Exact replica of `NeuroSymCBFModel.forward` from the point where the metadata comes in (see
    `models/neurosym_cbf.py`), so the encoder is not repeated for every intervention.
    """
    fused = model.fusion_proj(torch.cat([hidden, meta], dim=-1))
    z = model.sae.encode(fused)
    gamma, beta = model.film(z)
    coeffs_mod = gamma * coeffs_ns + beta
    forecast = model.symbolic_head(coeffs_mod)
    return forecast, z, gamma, beta


def _build_meta_batch(intervention: Intervention, idx: torch.Tensor, meta_all: torch.Tensor,
                      meta_mean: torch.Tensor, perm_cache: dict[str, torch.Tensor]) -> torch.Tensor:
    if intervention.kind == IDENTITY:
        return meta_all[idx]
    if intervention.kind == ZERO:
        return torch.zeros(len(idx), meta_all.shape[1], dtype=meta_all.dtype)
    if intervention.kind == MEAN:
        return meta_mean.expand(len(idx), -1)
    if intervention.kind == PERMUTE:
        return meta_all[perm_cache[intervention.name][idx]]
    raise ValueError(f"Unknown intervention kind: {intervention.kind!r}")


@torch.no_grad()
def run_metadata_interventions(
    model,
    dataset,
    interventions: list[Intervention],
    device: torch.device,
    batch_size: int = 2048,
    max_batches: int | None = None,
) -> dict:
    """Runs every intervention over the test set and returns metrics paired by series.

    `interventions[0]` must be the baseline (`baseline_intervention()`): it is the reference all
    the deltas are computed against.
    """
    if not interventions or interventions[0].kind != IDENTITY:
        raise ValueError("The first intervention must be the baseline (kind='identity')")

    model.eval()
    meta_all = torch.from_numpy(np.asarray(dataset.metadata, dtype=np.float32))
    meta_mean = meta_all.mean(dim=0, keepdim=True)
    n_series = meta_all.shape[0]
    for iv in interventions:
        if iv.perm is not None and len(iv.perm) != n_series:
            raise ValueError(f"The permutation of {iv.name!r} does not match {n_series} series")

    perm_cache = {
        iv.name: torch.from_numpy(iv.perm.astype(np.int64))
        for iv in interventions
        if iv.perm is not None
    }

    loader = DataLoader(_IndexedDataset(dataset), batch_size=batch_size, shuffle=False)
    names = [iv.name for iv in interventions]
    per_sample: dict[str, dict[str, list]] = {n: defaultdict(list) for n in names}
    sums: dict[str, dict[str, np.ndarray]] = {n: {} for n in names}
    series_index: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        context, horizon, mask, _meta, _stats, idx = batch
        context = context.to(device)
        horizon = horizon.to(device)
        mask_f = mask.to(device, dtype=torch.float32)
        series_index.append(idx.numpy())

        hidden = model.encode_ts(context)
        coeffs_ns = model.coeff_layer(hidden)

        base_forecast = base_active = None
        for iv in interventions:
            meta_b = _build_meta_batch(iv, idx, meta_all, meta_mean, perm_cache).to(device)
            forecast, z, gamma, beta = _forward_tail(model, hidden, coeffs_ns, meta_b)
            active = z > 0

            store = per_sample[iv.name]
            store["mae"].append(masked_mae_per_sample(forecast, horizon, mask_f).cpu().numpy())
            store["smape"].append(masked_smape_per_sample(forecast, horizon, mask_f).cpu().numpy())
            store["n_active"].append(active.sum(dim=1).cpu().numpy().astype(np.float32))

            if iv.kind == IDENTITY:
                base_forecast, base_active = forecast, active
                base_gamma, base_beta = gamma, beta
                jaccard = np.ones(len(idx), dtype=np.float32)
                turnover = np.zeros(len(idx), dtype=np.float32)
                d_forecast = np.zeros(len(idx), dtype=np.float32)
            else:
                inter = (active & base_active).sum(dim=1).float()
                union = (active | base_active).sum(dim=1).float().clamp(min=1.0)
                jaccard = (inter / union).cpu().numpy()
                turnover = (active ^ base_active).sum(dim=1).float().cpu().numpy()
                diff = (forecast - base_forecast).abs()
                d_forecast = (
                    (diff * mask_f).sum(dim=(1, 2)) / (mask_f.sum(dim=(1, 2)) + 1e-8)
                ).cpu().numpy()

            store["jaccard"].append(jaccard)
            store["concept_turnover"].append(turnover)
            store["d_forecast"].append(d_forecast)

            acc = sums[iv.name]
            batch_sums = {
                "fire_counts": active.sum(dim=0).cpu().numpy().astype(np.float64),
                "abs_d_gamma": (gamma - base_gamma).abs().sum(dim=0).cpu().numpy().astype(np.float64),
                "abs_d_beta": (beta - base_beta).abs().sum(dim=0).cpu().numpy().astype(np.float64),
            }
            for key, value in batch_sums.items():
                acc[key] = acc.get(key, 0.0) + value

        if batch_idx % 20 == 0:
            logger.info("Batch %d/%d", batch_idx + 1, len(loader))

    series_index_arr = np.concatenate(series_index)
    n_samples = len(series_index_arr)
    stacked = {
        name: {key: np.concatenate(chunks) for key, chunks in store.items()}
        for name, store in per_sample.items()
    }

    return _aggregate(interventions, stacked, sums, series_index_arr, n_samples)


def _sign_test_z(n_worse: int, n_total: int) -> float:
    """z statistic of the sign test (H0: the intervention makes things worse as often as better)."""
    if n_total == 0:
        return float("nan")
    return float((n_worse - n_total / 2) / np.sqrt(n_total / 4))


def _aggregate(interventions, stacked, sums, series_index_arr, n_samples) -> dict:
    """Aggregates into per-intervention scalars (JSON) and per-sample arrays (npz)."""
    base = stacked[BASELINE_NAME]
    summary: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {"series_index": series_index_arr}

    for iv in interventions:
        cur = stacked[iv.name]
        d_mae = cur["mae"] - base["mae"]

        if iv.affected is None:
            sel = np.ones(n_samples, dtype=bool)
        else:
            sel = iv.affected[series_index_arr]
        n_sel = int(sel.sum())

        n_worse = int((d_mae[sel] > 0).sum())
        d_sel = d_mae[sel]
        sem = float(d_sel.std(ddof=1) / np.sqrt(n_sel)) if n_sel > 1 else float("nan")
        mean_d = float(d_sel.mean()) if n_sel else float("nan")

        summary[iv.name] = {
            "kind": iv.kind,
            "notes": iv.notes,
            "n_evaluated": n_sel,
            "mae_mean": float(cur["mae"][sel].mean()) if n_sel else float("nan"),
            "mae_median": float(np.median(cur["mae"][sel])) if n_sel else float("nan"),
            "smape_mean": float(cur["smape"][sel].mean()) if n_sel else float("nan"),
            "delta_mae_mean": mean_d,
            "delta_mae_sem": sem,
            "delta_mae_ci95": [mean_d - 1.96 * sem, mean_d + 1.96 * sem] if n_sel > 1 else None,
            "delta_mae_median": float(np.median(d_sel)) if n_sel else float("nan"),
            "delta_mae_p90": float(np.percentile(d_sel, 90)) if n_sel else float("nan"),
            "delta_mae_rel_mean": (
                float(mean_d / base["mae"][sel].mean()) if n_sel and base["mae"][sel].mean() else float("nan")
            ),
            "frac_worse": float(n_worse / n_sel) if n_sel else float("nan"),
            "sign_test_z": _sign_test_z(n_worse, n_sel),
            "jaccard_mean": float(cur["jaccard"][sel].mean()) if n_sel else float("nan"),
            "jaccard_median": float(np.median(cur["jaccard"][sel])) if n_sel else float("nan"),
            "concept_turnover_mean": float(cur["concept_turnover"][sel].mean()) if n_sel else float("nan"),
            "d_forecast_mean": float(cur["d_forecast"][sel].mean()) if n_sel else float("nan"),
            "n_active_mean": float(cur["n_active"][sel].mean()) if n_sel else float("nan"),
            "dead_concepts": int((sums[iv.name]["fire_counts"] == 0).sum()),
            "abs_d_gamma_mean_per_coeff": (sums[iv.name]["abs_d_gamma"] / n_samples).tolist(),
            "abs_d_beta_mean_per_coeff": (sums[iv.name]["abs_d_beta"] / n_samples).tolist(),
        }

        prefix = iv.name
        arrays[f"{prefix}__mae"] = cur["mae"].astype(np.float32)
        arrays[f"{prefix}__delta_mae"] = d_mae.astype(np.float32)
        arrays[f"{prefix}__jaccard"] = cur["jaccard"].astype(np.float32)
        arrays[f"{prefix}__concept_turnover"] = cur["concept_turnover"].astype(np.float32)
        arrays[f"{prefix}__d_forecast"] = cur["d_forecast"].astype(np.float32)
        arrays[f"{prefix}__fire_counts"] = sums[iv.name]["fire_counts"].astype(np.float64)
        arrays[f"{prefix}__selected"] = sel

    return {"interventions": summary, "n_samples": int(n_samples), **arrays}
