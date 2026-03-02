"""
Latency instrumentation for Phase 1 pipeline profiling.

Usage:
    from src.utils.latency import timed, LatencyStore

    @timed("sentinel", "ingest_quote")
    def ingest_quote(self, symbol):
        ...

    # Later:
    stats = LatencyStore.instance().stats("sentinel", "ingest_quote")
    # => {"count": 42, "p50": 0.12, "p95": 0.34, "p99": 0.51, "mean": 0.15}
"""

from __future__ import annotations

import functools
import statistics
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class LatencyStore:
    """Thread-safe singleton that stores timing samples keyed by (agent, stage)."""

    _instance: Optional["LatencyStore"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._samples: Dict[Tuple[str, str], List[float]] = {}
        self._sample_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LatencyStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — useful in tests."""
        with cls._lock:
            cls._instance = None

    def record(self, agent: str, stage: str, duration_s: float) -> None:
        key = (agent, stage)
        with self._sample_lock:
            self._samples.setdefault(key, []).append(duration_s)

    def stats(self, agent: str, stage: str) -> Dict[str, float]:
        key = (agent, stage)
        with self._sample_lock:
            samples = list(self._samples.get(key, []))
        if not samples:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        sorted_s = sorted(samples)
        n = len(sorted_s)
        return {
            "count": n,
            "p50": _percentile(sorted_s, 50),
            "p95": _percentile(sorted_s, 95),
            "p99": _percentile(sorted_s, 99),
            "mean": round(statistics.mean(sorted_s), 6),
        }

    def all_keys(self) -> List[Tuple[str, str]]:
        with self._sample_lock:
            return list(self._samples.keys())


def _percentile(sorted_data: List[float], pct: int) -> float:
    """Simple nearest-rank percentile on pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = max(0, min(int(len(sorted_data) * pct / 100), len(sorted_data) - 1))
    return round(sorted_data[k], 6)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def timed(agent: str, stage: str) -> Callable:
    """Decorator that records wall-clock duration of each call."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                LatencyStore.instance().record(agent, stage, elapsed)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------

class StageStats(BaseModel):
    agent: str
    stage: str
    count: int
    p50: float
    p95: float
    p99: float
    mean: float


class LatencyReport(BaseModel):
    stages: List[StageStats]

    @classmethod
    def from_store(cls, store: Optional[LatencyStore] = None) -> "LatencyReport":
        store = store or LatencyStore.instance()
        stages = []
        for agent, stage in sorted(store.all_keys()):
            s = store.stats(agent, stage)
            stages.append(StageStats(
                agent=agent,
                stage=stage,
                count=int(s["count"]),
                p50=s["p50"],
                p95=s["p95"],
                p99=s["p99"],
                mean=s["mean"],
            ))
        return cls(stages=stages)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Latency Report",
            "",
            "| Agent | Stage | Count | P50 (s) | P95 (s) | P99 (s) | Mean (s) |",
            "|-------|-------|------:|--------:|--------:|--------:|---------:|",
        ]
        for s in self.stages:
            lines.append(
                f"| {s.agent} | {s.stage} | {s.count} "
                f"| {s.p50:.4f} | {s.p95:.4f} | {s.p99:.4f} | {s.mean:.4f} |"
            )
        return "\n".join(lines) + "\n"
