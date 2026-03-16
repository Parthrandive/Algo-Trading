from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Protocol

from src.agents.sentiment.schemas import SentimentCacheEntry, SentimentLane


def build_cache_key(*, lane: SentimentLane, text: str) -> tuple[str, str]:
    normalized = " ".join(text.lower().strip().split())
    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sentiment:{lane.value}:{text_hash}", text_hash


class SentimentCacheBackend(Protocol):
    def get(
        self,
        cache_key: str,
        *,
        as_of_utc: datetime | None = None,
        include_expired: bool = False,
    ) -> SentimentCacheEntry | None:
        ...

    def set(self, cache_key: str, entry: SentimentCacheEntry) -> None:
        ...


class InMemorySentimentCache:
    def __init__(self):
        self._records: dict[str, SentimentCacheEntry] = {}

    def get(
        self,
        cache_key: str,
        *,
        as_of_utc: datetime | None = None,
        include_expired: bool = False,
    ) -> SentimentCacheEntry | None:
        entry = self._records.get(cache_key)
        if entry is None:
            return None

        check_time = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        if entry.expires_at_utc <= check_time:
            if include_expired:
                return entry
            self._records.pop(cache_key, None)
            return None
        return entry

    def set(self, cache_key: str, entry: SentimentCacheEntry) -> None:
        self._records[cache_key] = entry

    def size(self) -> int:
        return len(self._records)
