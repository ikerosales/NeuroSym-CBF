"""Phase 4 of the pipeline: generate text embeddings (T5) from each series' metadata.

It combines title + tags (expanded into readable text via the `download_tags.py` lookup) +
units into one synthetic sentence per series, and encodes it with a T5 encoder (mean pooling over
the hidden state) to provide textual context to the multimodal model (Phase 6+). The result
(`metadata_embeddings.pkl`) is a derived artifact of the FRED dataset, so it is not published
(because of FRED's terms of use, see README).
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer

logger = logging.getLogger(__name__)


class T5TextEncoder:
    """T5-based text encoder (mean pooling over the encoder hidden state)."""

    def __init__(self, model_name: str = "google-t5/t5-base", max_length: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        logger.info("Model %s loaded on %s", model_name, self.device)

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
    """Normalize a value into usable text, or `None` if it is empty/missing."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in {"", "none", "nan", "na"}:
        return None
    return value


def build_metadata_text(row: pd.Series) -> str:
    """Synthetic sentence per series: title + tags + units (missing fields are omitted)."""
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
    """Load only the text fields needed from the database (without `observations`, to avoid
    duplicating the entire dataset -- which can weigh >1GB -- in memory)."""
    with open(database_json_path) as f:
        data = json.load(f)

    series = [
        {"series_id": x["series_id"], "title": x["title"], "units": x["units"], "notes": x.get("notes")}
        for x in data
        if "series_id" in x
    ]
    return pd.DataFrame(series)


def load_notes_lookup(notes_lookup_path: Path | Sequence[Path]) -> dict[str, str]:
    """Tag -> descriptive note lookup, merging several files when more than one is given.

    Needed to encode series that were not in the training corpus: their own tag download produces
    its own lookup, which brings new tags but also repeats known ones. The files are merged with
    priority to the first one (the training corpus), so a shared tag is expanded exactly as it was
    then and a new tag is not left unexpanded just because it never appeared in the corpus.
    """
    paths = [notes_lookup_path] if isinstance(notes_lookup_path, (str, Path)) else list(notes_lookup_path)
    merged: dict[str, str] = {}
    for path in reversed(paths):  # the first one in the list overrides the rest
        with open(path) as f:
            merged.update(json.load(f))
    if len(paths) > 1:
        logger.info("Notes lookup merged from %d files: %d tags", len(paths), len(merged))
    return merged


def load_tags_with_expanded_notes(
    tags_parquet_path: Path | Sequence[Path], notes_lookup_path: Path | Sequence[Path]
) -> pd.DataFrame:
    """Load the tags per series and replace each short tag with its descriptive note (e.g.
    "GDP" -> "Gross Domestic Product") using the lookup generated by `download_tags.py`.

    Both arguments accept one path or several; several tag files are concatenated and
    de-duplicated by `series_id`.
    """
    parquets = [tags_parquet_path] if isinstance(tags_parquet_path, (str, Path)) else list(tags_parquet_path)
    df_tags = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    df_tags = df_tags.drop_duplicates("series_id")
    tag_map = load_notes_lookup(notes_lookup_path)

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, tag_map.keys())) + r")\b")
    df_tags["tags_mapped"] = (
        df_tags["tags"]
        .str.replace(pattern, lambda m: tag_map[m.group(0)], regex=True)
        .str.replace("|", ", ", regex=False)
    )
    return df_tags


def build_metadata_embeddings(
    database_json_path: Path,
    tags_parquet_path: Path | Sequence[Path],
    notes_lookup_path: Path | Sequence[Path],
    model_name: str = "google-t5/t5-base",
    max_length: int = 128,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Orchestrate the full Phase 4: metadata + tags -> synthetic text -> T5 embeddings.

    Returns a DataFrame with columns `series_id`, `embeddings` (T5-dim array, 768) --
    the same column name expected by `load_series.load_metadata_embeddings`.
    """
    df = load_series_text_metadata(database_json_path)
    df_tags = load_tags_with_expanded_notes(tags_parquet_path, notes_lookup_path)

    missing = len(df) - len(df_tags)
    if missing:
        logger.warning("%d series without tags found (they will have no text for that field)", missing)

    df = df.merge(df_tags[["series_id", "tags_mapped"]], on="series_id", how="left")
    df["whole_text"] = df.apply(build_metadata_text, axis=1)

    encoder = T5TextEncoder(model_name=model_name, max_length=max_length)
    embeddings = encoder.encode(df["whole_text"].tolist(), batch_size=batch_size)
    logger.info("Embeddings generated: %s", tuple(embeddings.shape))

    return pd.DataFrame({"series_id": df["series_id"].values, "embeddings": list(embeddings.cpu().numpy())})


def save_embeddings(df_embeddings: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_embeddings.to_pickle(output_file)
    logger.info("Saved to %s", output_file)
