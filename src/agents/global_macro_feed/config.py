from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "configs" / "global_macro_feed_runtime_v1.json"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_backoff_seconds: float


@dataclass(frozen=True)
class GlobalMacroSource:
    name: str
    enabled: bool
    priority: int
    url: str
    source_type: str
    publisher: str
    timeout_seconds: float
    retry: RetryPolicy


@dataclass(frozen=True)
class GlobalMacroFeedConfig:
    version: str
    polling_interval_seconds: int
    max_items_per_source: int
    velocity_threshold: float
    bronze_base_dir: str
    silver_jsonl_dir: str
    sources: tuple[GlobalMacroSource, ...]


def load_global_macro_feed_config(config_path: str | Path) -> GlobalMacroFeedConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    version = str(raw.get("version", "global-macro-feed-runtime-v1"))
    polling_interval_seconds = int(raw.get("polling_interval_seconds", 900))
    max_items_per_source = int(raw.get("max_items_per_source", 40))

    risk_signal = raw.get("risk_signal", {}) or {}
    velocity_threshold = float(risk_signal.get("velocity_threshold", 0.6))

    storage = raw.get("storage", {}) or {}
    bronze_base_dir = str(storage.get("bronze_base_dir", "data/bronze"))
    silver_jsonl_dir = str(storage.get("silver_jsonl_dir", "data/silver/text/global_macro_feed"))

    sources_raw = raw.get("sources", [])
    sources: list[GlobalMacroSource] = []
    for item in sources_raw:
        if not isinstance(item, dict):
            continue
        retry_raw: dict[str, Any] = item.get("retry", {}) or {}
        retry = RetryPolicy(
            max_attempts=max(1, int(retry_raw.get("max_attempts", 2))),
            base_backoff_seconds=max(0.0, float(retry_raw.get("base_backoff_seconds", 1.0))),
        )
        source = GlobalMacroSource(
            name=str(item.get("name", "unknown_source")),
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 999)),
            url=str(item.get("url", "")),
            source_type=str(item.get("source_type", "rss_feed")),
            publisher=str(item.get("publisher", "WorldMonitor")),
            timeout_seconds=max(1.0, float(item.get("timeout_seconds", 12))),
            retry=retry,
        )
        if source.url:
            sources.append(source)

    return GlobalMacroFeedConfig(
        version=version,
        polling_interval_seconds=polling_interval_seconds,
        max_items_per_source=max_items_per_source,
        velocity_threshold=velocity_threshold,
        bronze_base_dir=bronze_base_dir,
        silver_jsonl_dir=silver_jsonl_dir,
        sources=tuple(sources),
    )


def load_default_global_macro_feed_config() -> GlobalMacroFeedConfig:
    return load_global_macro_feed_config(DEFAULT_RUNTIME_CONFIG_PATH)
