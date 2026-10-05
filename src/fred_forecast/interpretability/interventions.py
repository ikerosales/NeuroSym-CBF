"""Intervenciones sobre la rama de metadatos de NeuroSym-CBF, sin re-entrenar.

Todas las intervenciones sustituyen el embedding de metadatos que recibe una serie y miden qué
le pasa al forecast y a los conceptos del SAE. El contexto temporal nunca se toca, así que
`hidden_ts` y `coeffs_ns` son idénticos en todas las variantes: el runner los calcula **una vez
por batch** y solo repite la cola del modelo (fusion -> SAE -> FiLM -> cabeza simbólica), que es
la parte barata. Por eso evaluar N intervenciones no cuesta N pases completos.

Niveles, de menos a más informativo (cada uno controla al anterior):

1. **Ablación** (`zero`, `mean`): cota superior de cuánto usa el modelo el texto.
2. **Permutación aleatoria** (`permute`): el NULO. Sin esto, un swap dirigido no dice nada —
   no distinguirías daño semántico de "el vector simplemente no corresponde".
3. **Swap dirigido por tag** (`tag_swap`): intercambia metadatos entre series de dos tags
   semánticamente distintos. Solo es interpretable comparado contra el nulo del nivel 2.
4. **Control negativo** (`within_tagset_swap`): intercambia metadatos entre series con el
   *mismo* conjunto de tags. El texto sigue cambiando (el título y las unidades son distintos),
   pero el contenido semántico es casi el mismo, así que el efecto debería ser ~0.

Nota de diseño: el efecto máximo alcanzable está acotado por la arquitectura. `coeffs_ns` sale
del encoder de la serie y FiLM solo modula (`coeffs_mod = γ⊙coeffs_ns + β`), así que los
metadatos no pueden cambiar el forecast arbitrariamente. Es una propiedad del modelo, no un
artefacto de la medición.
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
    """Una intervención sobre la matriz de metadatos.

    `perm[i] = j` significa "la serie i recibe el embedding de la serie j". `affected` marca las
    series donde el embedding realmente cambia; los agregados se calculan solo sobre ellas
    (en un swap dirigido, la inmensa mayoría de las series no está implicada y promediar sobre
    todas diluiría el efecto hasta hacerlo invisible). `affected=None` significa "todas".
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
# Constructores de intervenciones
# --------------------------------------------------------------------------------------------


def baseline_intervention() -> Intervention:
    """Sin intervención: la referencia contra la que se comparan las demás."""
    return Intervention(name=BASELINE_NAME, kind=IDENTITY, notes="metadatos originales")


def ablation_interventions() -> list[Intervention]:
    """Nivel 1: metadatos a cero y metadatos a la media global del corpus."""
    return [
        Intervention(name="meta-zero", kind=ZERO, notes="embedding de metadatos = 0"),
        Intervention(name="meta-mean", kind=MEAN, notes="embedding = media global del corpus"),
    ]


def random_permutation_intervention(n_series: int, seed: int) -> Intervention:
    """Nivel 2: permutación aleatoria global de embeddings entre series (el nulo)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_series)
    affected = perm != np.arange(n_series)
    return Intervention(
        name=f"permute-seed{seed}",
        kind=PERMUTE,
        perm=perm,
        affected=affected,
        notes=f"permutación global aleatoria (seed={seed}, {int((~affected).sum())} puntos fijos)",
    )


def tag_sets_for(series_ids: list[str], tags_df: pd.DataFrame) -> list[frozenset[str]]:
    """Conjunto de tags de cada serie, en el mismo orden que `series_ids`.

    Las series sin fila en `tags_df` (o con tags vacíos) quedan como conjunto vacío: no entran
    en ningún swap dirigido ni en ningún grupo del control negativo.
    """
    tag_map = dict(zip(tags_df["series_id"].astype(str), tags_df["tags"].fillna("")))
    return [frozenset(t for t in tag_map.get(sid, "").split("|") if t) for sid in series_ids]


def _rotate_within_groups(groups: list[list[int]], n_series: int, rng) -> tuple[np.ndarray, np.ndarray, int]:
    """Permutación que rota los miembros (ya barajados) dentro de cada grupo.

    La rotación garantiza que ninguna serie se quede con su propio embedding, y que el embedding
    que recibe venga de otra serie del mismo grupo.
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
    """Nivel 3 + su control emparejado. Devuelve las dos intervenciones, en ese orden.

    - **Swap dirigido**: cada serie con `tag_a` recibe el embedding de una serie con `tag_b`, y
      viceversa. Se excluyen las series que llevan ambos tags (el cambio no sería contrastivo).
    - **Nulo emparejado**: exactamente las mismas series, exactamente el mismo número de
      embeddings cambiados, pero cada serie recibe el embedding de otra serie **de su propio
      grupo** (una serie de CPI recibe metadatos de otra serie de CPI).

    Comparar el swap contra el **nulo global** no vale: las series de un tag concreto pueden ser
    más sensibles a los metadatos que la media del corpus, y entonces el efecto del swap se
    confundiría con esa sensibilidad. El nulo emparejado fija la población y el número de
    cambios, así que la diferencia entre ambos aísla lo único que varía: si el embedding nuevo
    viene del grupo semántico contrario o del propio.
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
            f"swap dirigido {tag_a!r} <-> {tag_b!r}: {n_pairs} pares "
            f"(disponibles: {len(group_a)} y {len(group_b)})"
        ),
    )

    null_perm, null_affected, _ = _rotate_within_groups([paired_a, paired_b], n_series, rng)
    matched_null = Intervention(
        name=f"null-{label}",
        kind=PERMUTE,
        perm=null_perm,
        affected=affected,  # se evalúa sobre las MISMAS series que el swap
        notes=f"nulo emparejado de swap-{label}: barajado dentro de cada grupo",
    )
    if not np.array_equal(null_affected, affected):
        logger.warning("El nulo emparejado de %s no cubre exactamente las mismas series", label)

    return [swap, matched_null]


def within_tagset_swap_intervention(
    tag_sets: list[frozenset[str]], seed: int, min_group_size: int = 2
) -> Intervention:
    """Nivel 4 (control negativo): intercambia embeddings entre series de tags idénticos.

    El texto codificado sigue cambiando (título y unidades difieren entre miembros del grupo),
    pero la carga semántica es casi la misma, así que el efecto esperado es ~0.
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
        notes=f"control negativo: swap dentro de {n_groups} grupos de tags idénticos",
    )


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


class _IndexedDataset(Dataset):
    """Envuelve el dataset de test para devolver también el índice de serie de cada muestra.

    Hace falta para saber a qué serie pertenece cada fila del batch y poder aplicarle la
    permutación de metadatos correspondiente.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        return (*self.base[i], int(self.base._ts_idx[i]))


@torch.no_grad()
def _forward_tail(model, hidden, coeffs_ns, meta):
    """Cola del modelo a partir de `hidden`/`coeffs_ns` ya calculados: fusion -> SAE -> FiLM -> head.

    Réplica exacta de `NeuroSymCBFModel.forward` a partir del punto donde entran los metadatos
    (ver `models/neurosym_cbf.py`), para no repetir el encoder en cada intervención.
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
    raise ValueError(f"Tipo de intervención desconocido: {intervention.kind!r}")


@torch.no_grad()
def run_metadata_interventions(
    model,
    dataset,
    interventions: list[Intervention],
    device: torch.device,
    batch_size: int = 2048,
    max_batches: int | None = None,
) -> dict:
    """Ejecuta todas las intervenciones sobre el test set y devuelve métricas pareadas por serie.

    `interventions[0]` debe ser la línea base (`baseline_intervention()`): es la referencia
    contra la que se calculan todos los deltas.
    """
    if not interventions or interventions[0].kind != IDENTITY:
        raise ValueError("La primera intervención debe ser la línea base (kind='identity')")

    model.eval()
    meta_all = torch.from_numpy(np.asarray(dataset.metadata, dtype=np.float32))
    meta_mean = meta_all.mean(dim=0, keepdim=True)
    n_series = meta_all.shape[0]
    for iv in interventions:
        if iv.perm is not None and len(iv.perm) != n_series:
            raise ValueError(f"La permutación de {iv.name!r} no cuadra con {n_series} series")

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
    """Estadístico z del test de signos (H0: la intervención empeora tanto como mejora)."""
    if n_total == 0:
        return float("nan")
    return float((n_worse - n_total / 2) / np.sqrt(n_total / 4))


def _aggregate(interventions, stacked, sums, series_index_arr, n_samples) -> dict:
    """Agrega a escalares por intervención (JSON) y arrays por muestra (npz)."""
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
