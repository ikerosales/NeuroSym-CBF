"""Phase 1 of the pipeline (metadata): download the tags for each series_id from FRED.

Async, with sliding-window rate limiting, exponential backoff retries, and
disk checkpointing so it can resume if interrupted halfway through the bulk download
(~194k series). It also builds a tag -> descriptive note lookup, used later to
expand short tags (e.g. "GDP") into full text before passing them to the T5 encoder -- see
`src/fred_forecast/embeddings/text_encoder.py`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

TAGS_URL = "https://api.stlouisfed.org/fred/series/tags"


class RateLimiter:
    """Sliding window of `period` seconds -- respects FRED's requests/min limit."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.period:
                self._calls.popleft()

            if len(self._calls) >= self.max_calls:
                sleep_for = self.period - (now - self._calls[0])
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()

            self._calls.append(time.monotonic())


async def fetch_tags_for_series(
    session: aiohttp.ClientSession,
    limiter: RateLimiter,
    series_id: str,
    semaphore: asyncio.Semaphore,
    api_key: str,
    max_retries: int,
) -> dict:
    """Fetch the tags for a series_id, with retries and exponential backoff."""
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}

    async with semaphore:
        for attempt in range(max_retries):
            await limiter.acquire()
            try:
                async with session.get(
                    TAGS_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tags = data.get("tags", [])
                        return {
                            "series_id": series_id,
                            "tags": [t["name"] for t in tags],
                            "tags_detail": [t.get("notes") for t in tags],
                            "error": None,
                        }
                    if resp.status == 429:
                        wait = 60 * (attempt + 1)
                        logger.warning("[%s] 429 received, waiting %ds", series_id, wait)
                        await asyncio.sleep(wait)
                    elif resp.status in (500, 502, 503, 504):
                        wait = 2**attempt
                        logger.warning(
                            "[%s] HTTP %d, retry %d in %ds", series_id, resp.status, attempt + 1, wait
                        )
                        await asyncio.sleep(wait)
                    else:
                        body = await resp.text()
                        return {
                            "series_id": series_id,
                            "tags": [],
                            "tags_detail": [],
                            "error": f"HTTP {resp.status}: {body[:200]}",
                        }
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = 2**attempt
                logger.warning("[%s] network error (%s), retry %d in %ds", series_id, exc, attempt + 1, wait)
                await asyncio.sleep(wait)

    return {
        "series_id": series_id,
        "tags": [],
        "tags_detail": [],
        "error": f"Failed after {max_retries} attempts",
    }


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _update_notes_lookup(results: list[dict], lookup: dict) -> dict:
    """Add only new tag-note pairs with a non-empty note to the lookup."""
    for r in results:
        for tag, note in zip(r["tags"], r["tags_detail"]):
            if tag not in lookup and note and note.strip():
                lookup[tag] = note.strip()
    return lookup


def _save_results(results: list[dict], output_file: Path, append: bool) -> None:
    df = pd.DataFrame(
        [
            {
                "series_id": r["series_id"],
                "tags": "|".join(r["tags"]),
                "n_tags": len(r["tags"]),
                "error": r["error"],
            }
            for r in results
        ]
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if append and output_file.exists():
        existing = pd.read_parquet(output_file)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates("series_id")
    df.to_parquet(output_file, index=False)


async def download_all_tags(
    series_ids: list[str],
    api_key: str,
    output_file: Path,
    checkpoint_file: Path,
    notes_file: Path,
    max_requests_per_minute: int = 120,
    concurrent_requests: int = 20,
    max_retries: int = 5,
    batch_size: int = 5000,
) -> None:
    """Download the tags for `series_ids`, resuming from `checkpoint_file` if it already exists."""
    done_ids = set(_load_json(checkpoint_file, []))
    notes_lookup = _load_json(notes_file, {})
    pending = [s for s in series_ids if s not in done_ids]
    logger.info("Total: %d | already processed: %d | pending: %d", len(series_ids), len(done_ids), len(pending))

    if not pending:
        logger.info("Everything was already processed. Nothing to do.")
        return

    limiter = RateLimiter(max_calls=max_requests_per_minute)
    semaphore = asyncio.Semaphore(concurrent_requests)
    connector = aiohttp.TCPConnector(limit=concurrent_requests, ttl_dns_cache=300)

    all_results: list[dict] = []
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            logger.info("Processing chunk %d (%d series)...", i // batch_size + 1, len(chunk))

            tasks = [
                fetch_tags_for_series(session, limiter, sid, semaphore, api_key, max_retries) for sid in chunk
            ]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            notes_lookup = _update_notes_lookup(results, notes_lookup)
            _save_json(notes_file, notes_lookup)

            done_ids.update(r["series_id"] for r in results)
            _save_json(checkpoint_file, sorted(done_ids))
            _save_results(all_results, output_file, append=(i > 0))
            logger.info("Chunk saved. Total accumulated: %d", len(all_results))

    ok = sum(1 for r in all_results if not r["error"])
    logger.info("Extraction complete. Successes: %d | Errors: %d", ok, len(all_results) - ok)
