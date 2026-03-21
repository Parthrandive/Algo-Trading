from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.agents.sentiment.schemas import CacheFreshnessState

_HOUR = 60 * 60
_DAY = 24 * _HOUR

DEFAULT_SOURCE_TTLS: dict[str, int] = {
    "news": 4 * _HOUR,
    "social": 1 * _HOUR,
    "transcript": 7 * _DAY,
    "rbi_report": 30 * _DAY,
}


@dataclass(frozen=True)
class CachePolicyDecision:
    freshness_state: CacheFreshnessState
    ttl_seconds: int
    age_seconds: float
    weight_multiplier: float
    reduced_risk_mode: bool


def resolve_source_bucket(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return "news"

    publisher = str(payload.get("publisher") or payload.get("source_name") or payload.get("author") or "").lower()
    item_type = str(payload.get("item_type") or "").lower()
    source_type = str(payload.get("source_type") or "").lower()
    platform = str(payload.get("platform") or "").lower()

    if "rbi" in publisher or "reserve bank of india" in publisher:
        return "rbi_report"
    if item_type == "transcript" or bool(payload.get("quarter")) or bool(payload.get("year")):
        return "transcript"
    if source_type == "social_media" or item_type == "social" or platform:
        return "social"
    return "news"


def resolve_ttl_seconds(
    payload: Mapping[str, Any] | None,
    *,
    fallback_seconds: int | None = None,
    source_ttls: Mapping[str, int] | None = None,
) -> int:
    resolved_source_ttls = dict(DEFAULT_SOURCE_TTLS)
    if source_ttls:
        resolved_source_ttls.update({str(key): int(value) for key, value in source_ttls.items()})

    bucket = resolve_source_bucket(payload)
    if bucket in resolved_source_ttls:
        return int(resolved_source_ttls[bucket])
    if fallback_seconds is not None:
        return int(fallback_seconds)
    return int(DEFAULT_SOURCE_TTLS["news"])


def classify_freshness(*, age_seconds: float, ttl_seconds: int) -> CacheFreshnessState:
    safe_age = max(0.0, float(age_seconds))
    safe_ttl = max(1, int(ttl_seconds))
    if safe_age <= safe_ttl:
        return CacheFreshnessState.FRESH
    if safe_age <= (2 * safe_ttl):
        return CacheFreshnessState.STALE
    return CacheFreshnessState.EXPIRED


def weight_multiplier_for_state(state: CacheFreshnessState) -> float:
    if state == CacheFreshnessState.FRESH:
        return 1.0
    if state == CacheFreshnessState.STALE:
        return 0.5
    return 0.0


def evaluate_cache_policy(*, age_seconds: float, ttl_seconds: int) -> CachePolicyDecision:
    freshness_state = classify_freshness(age_seconds=age_seconds, ttl_seconds=ttl_seconds)
    weight_multiplier = weight_multiplier_for_state(freshness_state)
    return CachePolicyDecision(
        freshness_state=freshness_state,
        ttl_seconds=max(1, int(ttl_seconds)),
        age_seconds=max(0.0, float(age_seconds)),
        weight_multiplier=weight_multiplier,
        reduced_risk_mode=freshness_state == CacheFreshnessState.EXPIRED,
    )
