"""Fase 2 del pipeline: limpieza y homogeneización temporal de los releases descargados.

A partir de los parquets crudos por release (Fase 1), determina la ventana de contexto
temporal más larga que deja el mayor número de series utilizables, descarta series con
demasiado NaN o con NaN justo en el borde de la ventana, y rellena (forward-fill) el resto de
huecos. El resultado es un parquet por release recortado a esa ventana, listo para ensamblar
en la base de datos final (ver `build_database.py`). El TFG trabaja solo con series mensuales.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import tqdm
from fastparquet import ParquetFile

logger = logging.getLogger(__name__)


def compute_valid_bounds(parquet_dir: Path) -> pd.DataFrame:
    """Primera y última fecha con valor no-NaN, por serie, agregando todos los releases.

    Un `series_id` normalmente vive en un único release, pero se agrega con min/max por si
    apareciera repetido en más de uno.
    """
    series_bounds: dict[str, list[pd.Timestamp]] = {}

    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="bounds válidos"):
        df = pd.read_parquet(file, columns=["series_id", "date", "value"], engine="fastparquet")
        g = df.groupby("series_id")
        first_valid = g.apply(lambda x: x.loc[x["value"].notna(), "date"].min())
        last_valid = g.apply(lambda x: x.loc[x["value"].notna(), "date"].max())
        stats = pd.DataFrame({"first_valid": first_valid, "last_valid": last_valid})

        for sid, row in stats.iterrows():
            if sid not in series_bounds:
                series_bounds[sid] = [row["first_valid"], row["last_valid"]]
            else:
                series_bounds[sid][0] = min(series_bounds[sid][0], row["first_valid"])
                series_bounds[sid][1] = max(series_bounds[sid][1], row["last_valid"])

    bounds_valid = pd.DataFrame(series_bounds).T
    bounds_valid.columns = ["first_valid", "last_valid"]
    return bounds_valid


def find_best_cutoff(
    bounds_valid: pd.DataFrame, context_months: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Series]:
    """Fecha de corte que maximiza el nº de series con >= `context_months` de contexto continuo.

    Devuelve `(best_cutoff, window_start, counts_por_fecha_candidata)`.
    """
    candidate_dates = pd.date_range(
        bounds_valid["first_valid"].min() + pd.DateOffset(months=context_months),
        bounds_valid["last_valid"].max(),
        freq="MS",
    )

    counts = []
    for d in candidate_dates:
        active = (
            (bounds_valid["first_valid"] <= d - pd.DateOffset(months=context_months))
            & (bounds_valid["last_valid"] >= d)
        ).sum()
        counts.append(active)

    counts_series = pd.Series(counts, index=candidate_dates)
    best_cutoff = counts_series.idxmax()
    window_start = best_cutoff - pd.DateOffset(months=context_months)

    logger.info("Mejor cutoff: %s (%d series activas)", best_cutoff, counts_series.max())
    logger.info("Ventana: %s -> %s", window_start, best_cutoff)
    return best_cutoff, window_start, counts_series


def select_series_with_context(
    bounds_valid: pd.DataFrame, window_start: pd.Timestamp, best_cutoff: pd.Timestamp
) -> set[str]:
    """Series cuyo rango válido cubre completamente la ventana [window_start, best_cutoff]."""
    valid = set(
        bounds_valid[
            (bounds_valid["first_valid"] <= window_start) & (bounds_valid["last_valid"] >= best_cutoff)
        ].index
    )
    logger.info("Series con contexto válido: %d", len(valid))
    return valid


def filter_by_nan_ratio(
    parquet_dir: Path,
    candidate_series: set[str],
    bounds_valid: pd.DataFrame,
    max_nan_ratio: float = 0.25,
) -> set[str]:
    """Descarta series cuyo ratio de NaN (dentro de su propio rango [first_valid, last_valid])
    supera `max_nan_ratio`.
    """
    nan_ratios = []
    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="ratio de NaN"):
        df = pd.read_parquet(file, columns=["series_id", "date", "value"], engine="fastparquet")
        df = df[df["series_id"].isin(candidate_series)]
        if df.empty:
            continue

        bounds_subset = bounds_valid.loc[df["series_id"].unique()]
        df = df.merge(bounds_subset, left_on="series_id", right_index=True)
        df = df[(df["date"] >= df["first_valid"]) & (df["date"] <= df["last_valid"])]

        g = df.groupby("series_id")["value"].apply(lambda x: x.isna().mean())
        nan_ratios.append(g)

    nan_ratios_combined = pd.concat(nan_ratios).groupby(level=0).mean()
    removed = set(nan_ratios_combined[nan_ratios_combined > max_nan_ratio].index)
    kept = candidate_series - removed

    logger.info("Series descartadas por >%.0f%% NaN: %d", max_nan_ratio * 100, len(removed))
    logger.info("Series tras filtro de NaN: %d", len(kept))
    return kept


def filter_by_edge_nan(parquet_dir: Path, candidate_series: set[str], best_cutoff: pd.Timestamp) -> set[str]:
    """Descarta series con NaN justo en la fecha de corte (`best_cutoff`)."""
    invalid: set[str] = set()
    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="NaN en el borde"):
        df = pd.read_parquet(file, columns=["series_id", "date", "value"], engine="fastparquet")
        df = df[df["series_id"].isin(candidate_series)]
        if df.empty:
            continue

        df_cutoff = df[df["date"] == best_cutoff]
        if df_cutoff.empty:
            continue

        bad = df_cutoff[df_cutoff["value"].isna()]["series_id"].unique()
        invalid.update(bad)

    kept = candidate_series - invalid
    logger.info("Series descartadas por NaN en el borde de la ventana: %d", len(invalid))
    logger.info("Series finales utilizables: %d", len(kept))
    return kept


def write_cleaned_releases(
    parquet_dir: Path,
    output_dir: Path,
    valid_series: set[str],
    bounds_valid: pd.DataFrame,
    best_cutoff: pd.Timestamp,
) -> None:
    """Recorta cada release a [first_valid_de_la_serie, best_cutoff], hace forward-fill de los
    NaN restantes, y escribe el resultado (por row-group, para no cargar todo en memoria) en
    `output_dir`.

    Nota: el forward-fill se aplica dentro de cada row-group leído del parquet original, no
    sobre la serie completa reensamblada — fiel al comportamiento del código original. Si el
    historial de una serie estuviera repartido en varios row-groups no contiguos, un hueco al
    inicio de un row-group no heredaría el último valor válido del row-group anterior. En la
    práctica cada release cabe en pocas páginas de la API (límite de 500k filas), así que esto
    no se observó como problema, pero queda documentado por si se cambia el tamaño de página.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    first_valid_map = bounds_valid["first_valid"].to_dict()

    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="ffill + escritura"):
        pf = ParquetFile(file)
        output_file = output_dir / file.name
        first_write = True

        for df in pf.iter_row_groups():
            df = df[df["series_id"].isin(valid_series)]
            if df.empty:
                continue

            df["first_valid"] = df["series_id"].map(first_valid_map)
            df = df[(df["date"] >= df["first_valid"]) & (df["date"] <= best_cutoff)]
            df = df.drop(columns="first_valid")
            df = df.sort_values(["series_id", "date"])
            df["value"] = df.groupby("series_id")["value"].transform(lambda s: s.ffill())

            if first_write:
                df.to_parquet(output_file, engine="fastparquet", index=False)
                first_write = False
            else:
                df.to_parquet(output_file, engine="fastparquet", append=True)


def check_context_coverage(
    output_dir: Path, window_start: pd.Timestamp, best_cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Series del dataset limpio cuyo rango de fechas NO cubre [window_start, best_cutoff]
    (debería devolver un DataFrame vacío).
    """
    bounds: dict[str, list[pd.Timestamp]] = {}
    for file in tqdm.tqdm(sorted(output_dir.glob("release_*.parquet")), desc="check cobertura"):
        df = pd.read_parquet(file, columns=["series_id", "date"])
        g = df.groupby("series_id")["date"].agg(["min", "max"])
        for sid, row in g.iterrows():
            if sid not in bounds:
                bounds[sid] = [row["min"], row["max"]]
            else:
                bounds[sid][0] = min(bounds[sid][0], row["min"])
                bounds[sid][1] = max(bounds[sid][1], row["max"])

    bounds_df = pd.DataFrame(bounds).T
    bounds_df.columns = ["first_date", "last_date"]
    return bounds_df[(bounds_df["first_date"] > window_start) | (bounds_df["last_date"] < best_cutoff)]


def count_remaining_nans(output_dir: Path) -> tuple[int, int]:
    """(nº de valores NaN, nº total de valores) en el dataset limpio — el primero debería ser 0."""
    total_nans = 0
    total_values = 0
    for file in tqdm.tqdm(sorted(output_dir.glob("release_*.parquet")), desc="check NaN restantes"):
        df = pd.read_parquet(file, columns=["value"])
        total_nans += int(df["value"].isna().sum())
        total_values += len(df)
    return total_nans, total_values


def clean_releases(
    parquet_dir: Path,
    output_dir: Path,
    context_months: int = 48,
    max_nan_ratio: float = 0.25,
) -> dict:
    """Orquesta la Fase 2 completa: bounds -> mejor cutoff -> filtros de NaN -> ffill + write.

    Devuelve un resumen con las cifras clave del proceso (útil para logging y tests).
    """
    bounds_valid = compute_valid_bounds(parquet_dir)
    best_cutoff, window_start, _counts = find_best_cutoff(bounds_valid, context_months)

    candidate_series = select_series_with_context(bounds_valid, window_start, best_cutoff)
    candidate_series = filter_by_nan_ratio(parquet_dir, candidate_series, bounds_valid, max_nan_ratio)
    valid_series = filter_by_edge_nan(parquet_dir, candidate_series, best_cutoff)

    write_cleaned_releases(parquet_dir, output_dir, valid_series, bounds_valid, best_cutoff)

    violations = check_context_coverage(output_dir, window_start, best_cutoff)
    total_nans, total_values = count_remaining_nans(output_dir)

    summary = {
        "best_cutoff": best_cutoff,
        "window_start": window_start,
        "n_series_final": len(valid_series),
        "n_context_violations": len(violations),
        "n_remaining_nans": total_nans,
        "n_total_values": total_values,
    }
    logger.info("Fase 2 completada: %s", summary)
    if len(violations) > 0:
        logger.warning("%d series no cubren la ventana de contexto esperada", len(violations))
    if total_nans > 0:
        logger.warning("Quedan %d valores NaN en el dataset limpio", total_nans)

    return summary
