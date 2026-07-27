"""Fase 4 del pipeline: genera embeddings de texto (T5) a partir de los metadatos de cada serie.

Combina título + tags (expandidos a texto legible vía el lookup de `download_tags.py`) +
unidades en una frase sintética por serie, y la codifica con un T5 encoder (mean pooling sobre
el hidden state) para dar contexto textual al modelo multimodal (Fase 6+). El resultado
(`metadata_embeddings.pkl`) es un artefacto derivado del dataset FRED, así que no se publica
(por las condiciones de uso de FRED, ver README).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer

logger = logging.getLogger(__name__)


class T5TextEncoder:
    """Encoder de texto basado en T5 (mean pooling sobre el hidden state del encoder)."""

    def __init__(self, model_name: str = "google-t5/t5-base", max_length: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        logger.info("Modelo %s cargado en %s", model_name, self.device)

    @staticmethod
    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                emb = self.mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)


def clean_text(value) -> str | None:
    """Normaliza un valor a texto usable, o `None` si está vacío/ausente."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in {"", "none", "nan", "na"}:
        return None
    return value


def build_metadata_text(row: pd.Series) -> str:
    """Frase sintética por serie: título + tags + unidades (los campos ausentes se omiten)."""
    title = clean_text(row.get("title"))
    tags = clean_text(row.get("tags_mapped"))
    units = clean_text(row.get("units"))

    parts = []
    if title:
        parts.append(f"Time series about {title}.")
    if tags:
        parts.append(f"It is associated with the tags: {tags}.")
    if units:
        parts.append(f"Measured in {units}.")
    return " ".join(parts)


def load_series_text_metadata(database_json_path: Path) -> pd.DataFrame:
    """Carga solo los campos de texto necesarios de la base de datos (sin `observations`, para
    no duplicar el dataset entero — que puede pesar >1GB — en memoria)."""
    with open(database_json_path) as f:
        data = json.load(f)

    series = [
        {"series_id": x["series_id"], "title": x["title"], "units": x["units"], "notes": x.get("notes")}
        for x in data
        if "series_id" in x
    ]
    return pd.DataFrame(series)


def load_tags_with_expanded_notes(tags_parquet_path: Path, notes_lookup_path: Path) -> pd.DataFrame:
    """Carga los tags por serie y sustituye cada tag corto por su nota descriptiva (p.ej.
    "GDP" -> "Gross Domestic Product") usando el lookup generado por `download_tags.py`.
    """
    df_tags = pd.read_parquet(tags_parquet_path)
    with open(notes_lookup_path) as f:
        tag_map = json.load(f)

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, tag_map.keys())) + r")\b")
    df_tags["tags_mapped"] = (
        df_tags["tags"]
        .str.replace(pattern, lambda m: tag_map[m.group(0)], regex=True)
        .str.replace("|", ", ", regex=False)
    )
    return df_tags


def build_metadata_embeddings(
    database_json_path: Path,
    tags_parquet_path: Path,
    notes_lookup_path: Path,
    model_name: str = "google-t5/t5-base",
    max_length: int = 128,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Orquesta la Fase 4 completa: metadatos + tags -> texto sintético -> embeddings T5.

    Devuelve un DataFrame con columnas `series_id`, `embeddings` (array de dim T5, 768) —
    el mismo nombre de columna que espera `load_series.load_metadata_embeddings`.
    """
    df = load_series_text_metadata(database_json_path)
    df_tags = load_tags_with_expanded_notes(tags_parquet_path, notes_lookup_path)

    missing = len(df) - len(df_tags)
    if missing:
        logger.warning("%d series sin tags encontrados (se quedan sin ese texto)", missing)

    df = df.merge(df_tags[["series_id", "tags_mapped"]], on="series_id", how="left")
    df["whole_text"] = df.apply(build_metadata_text, axis=1)

    encoder = T5TextEncoder(model_name=model_name, max_length=max_length)
    embeddings = encoder.encode(df["whole_text"].tolist(), batch_size=batch_size)
    logger.info("Embeddings generados: %s", tuple(embeddings.shape))

    return pd.DataFrame({"series_id": df["series_id"].values, "embeddings": list(embeddings.cpu().numpy())})


def save_embeddings(df_embeddings: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_embeddings.to_pickle(output_file)
    logger.info("Guardado en %s", output_file)
