"""Automatic naming of NeuroSym-CBF's Top-K SAE concepts, via tag enrichment.

Idea: each of the `dict_size` atoms in the SAE dictionary (`TopKSAE`, see `models/topk_sae.py`)
activates (enters the top-k) for a subset of series. If that subset shares a FRED tag much more
often than the general population, that tag is a naming candidate for the atom -- the same logic
as "concept naming" in the SAE-for-LLMs literature, adapted to economic series tags instead of
text tokens.

The enrichment test is an exact hypergeometric test (equivalent to a one-tailed Fisher test): of
the `n_active` series where the atom is active, how many have the tag, compared to what would be
expected if the atom's activation were independent of having that tag?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def reconstruct_test_series_order(database_json_path, context_len: int, horizon_len: int) -> list[str]:
    """Reconstructs the exact `series_id` order of the test set, without touching the model.

    `split_train_test_obs` is deterministic and does not shuffle anything, so this order matches
    the rows of the arrays saved in `scripts/10_evaluate.py`'s `.npz` for that same
    `context_len`/`horizon_len` -- this lets `z` (SAE activations) be aligned with `series_id`
    without having to re-evaluate the model.
    """
    from fred_forecast.datasets.load_series import load_series_observations
    from fred_forecast.datasets.timeseries_dataset import split_train_test_obs

    df = load_series_observations(database_json_path)
    _df_train, df_test = split_train_test_obs(df, context_len, horizon_len)
    return df_test["series_id"].tolist()


def load_tag_sets(tags_parquet_path) -> dict[str, frozenset[str]]:
    """`series_id` -> set of short (unexpanded) tags, from `download_tags.py`'s parquet."""
    df = pd.read_parquet(tags_parquet_path, columns=["series_id", "tags"])
    df = df.dropna(subset=["tags"])
    return {
        str(row.series_id): frozenset(t for t in row.tags.split("|") if t)
        for row in df.itertuples(index=False)
    }


def atom_membership(z: np.ndarray) -> np.ndarray:
    """Boolean matrix (n_samples, dict_size): whether the atom is active (nonzero) in that sample.

    With hard Top-K activation, `z > 0` is exactly "among the k active atoms" -- there is no need
    to pass `k` separately.
    """
    return z > 0


def enrich_atom(
    active_mask: np.ndarray,
    series_ids: list[str],
    tag_sets: dict[str, frozenset[str]],
    tag_base_counts: pd.Series,
    n_total: int,
    min_tag_count_in_active: int = 5,
    top_n: int = 5,
) -> pd.DataFrame:
    """Tag enrichment table for ONE already-activated atom (`active_mask`, length n_samples).

    Per tag: count within the active group, lift (rate in the group / base rate), and one-tailed
    hypergeometric p-value (P(at least that count) under independence). Tags with fewer than
    `min_tag_count_in_active` occurrences in the active group are filtered out, so an extremely
    rare tag with 1-2 chance matches does not win on an artificially high lift.
    """
    active_ids = [series_ids[i] for i in np.flatnonzero(active_mask)]
    n_active = len(active_ids)
    if n_active == 0:
        return pd.DataFrame(columns=["tag", "n_in_active", "n_active", "base_rate", "active_rate", "lift", "p_value"])

    tag_counts_active: dict[str, int] = {}
    for sid in active_ids:
        for tag in tag_sets.get(sid, ()):
            tag_counts_active[tag] = tag_counts_active.get(tag, 0) + 1

    rows = []
    for tag, n_in_active in tag_counts_active.items():
        if n_in_active < min_tag_count_in_active:
            continue
        n_tag_total = int(tag_base_counts.get(tag, 0))
        if n_tag_total == 0:
            continue
        base_rate = n_tag_total / n_total
        active_rate = n_in_active / n_active
        lift = active_rate / base_rate if base_rate > 0 else np.inf
        # P(X >= n_in_active) with X ~ Hypergeom(N=n_total, K=n_tag_total, n=n_active)
        p_value = hypergeom.sf(n_in_active - 1, n_total, n_tag_total, n_active)
        rows.append(
            {
                "tag": tag,
                "n_in_active": n_in_active,
                "n_active": n_active,
                "base_rate": base_rate,
                "active_rate": active_rate,
                "lift": lift,
                "p_value": p_value,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["p_value", "lift"], ascending=[True, False]).head(top_n).reset_index(drop=True)


def enrich_tag_across_atoms(
    tag: str,
    z: np.ndarray,
    series_ids: list[str],
    tag_sets: dict[str, frozenset[str]],
) -> pd.DataFrame:
    """The inverse of `enrich_atom`: fixes ONE tag and looks across all `dict_size` atoms.

    `enrich_atom`/`name_all_concepts` only keep the top-`n` tags per atom, so a common tag (e.g.
    "usa", present in 88% of series) might not show up in any atom's top-5 even if it is related
    to several, simply because rarer, more "surprising" tags win the slot. For specific tags that
    are worth checking explicitly (known macro concepts: "usa", "unemployment", "gdp", ...) this
    second, full pass over all `dict_size` atoms is needed.

    Besides `lift` (how much more frequent the tag is WITHIN the active group than at the base
    rate, a precision measure), this adds `recall` (what fraction of ALL series with that tag fall
    inside that atom's active group) -- for a ubiquitous tag like "usa" the lift can barely rise
    (the base rate is already very high), so without `recall` there is no way to tell an atom that
    truly concentrates the concept apart from one that just shares it with nearly the whole
    dictionary.
    """
    ids_with_tag = {sid for sid, tags in tag_sets.items() if tag in tags}
    n_tag_total = len(ids_with_tag)
    n_total = len(series_ids)
    base_rate = n_tag_total / n_total if n_total else 0.0

    membership = atom_membership(z)
    series_ids_arr = np.asarray(series_ids)
    has_tag = np.isin(series_ids_arr, list(ids_with_tag)) if ids_with_tag else np.zeros(n_total, dtype=bool)

    rows = []
    for atom_idx in range(z.shape[1]):
        mask = membership[:, atom_idx]
        n_active = int(mask.sum())
        n_in_active = int((mask & has_tag).sum())
        active_rate = n_in_active / n_active if n_active else 0.0
        lift = active_rate / base_rate if base_rate > 0 else np.nan
        recall = n_in_active / n_tag_total if n_tag_total else 0.0
        p_value = (
            hypergeom.sf(n_in_active - 1, n_total, n_tag_total, n_active)
            if n_active and n_tag_total else 1.0
        )
        rows.append(
            {
                "atom": atom_idx, "n_active": n_active, "n_in_active": n_in_active,
                "base_rate": base_rate, "active_rate": active_rate, "lift": lift,
                "recall": recall, "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def name_all_concepts(
    z: np.ndarray,
    series_ids: list[str],
    tag_sets: dict[str, frozenset[str]],
    min_tag_count_in_active: int = 5,
    top_n: int = 5,
) -> list[dict]:
    """Tag enrichment for all `dict_size` atoms of the SAE dictionary.

    Returns a list of dicts (one per atom), with: index, number of series where it activates
    (`n_active`), and its table of naming-candidate tags (`top_tags`, can be empty if the atom has
    no tag that clears the enrichment bar).
    """
    n_total = len(series_ids)
    all_tags = set()
    for tags in tag_sets.values():
        all_tags.update(tags)
    tag_base_counts = pd.Series(0, index=sorted(all_tags), dtype=int)
    for tags in tag_sets.values():
        for tag in tags:
            tag_base_counts[tag] += 1

    membership = atom_membership(z)
    dict_size = z.shape[1]

    results = []
    for atom_idx in range(dict_size):
        mask = membership[:, atom_idx]
        top_tags = enrich_atom(
            mask, series_ids, tag_sets, tag_base_counts, n_total,
            min_tag_count_in_active=min_tag_count_in_active, top_n=top_n,
        )
        results.append(
            {
                "atom": atom_idx,
                "n_active": int(mask.sum()),
                "top_tags": top_tags.to_dict(orient="records"),
            }
        )
    return results
