from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from src.agents.strategic.config import StrategicAssemblerConfig
from src.agents.strategic.schemas import StrategicObservation
from src.db.phase3_recorder import Phase3Recorder
from src.db.queries import get_phase2_signal_bundle


@dataclass
class MaterializationSummary:
    symbols: list[str]
    rows_built: int
    rows_materialized: int
    rows_with_warn_quality: int
    per_symbol_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationAssembler:
    """Builds Phase 3 observations from live Phase 2 DB tables."""

    def __init__(
        self,
        *,
        config: StrategicAssemblerConfig | None = None,
        database_url: str | None = None,
        recorder: Phase3Recorder | None = None,
    ) -> None:
        self.config = config or StrategicAssemblerConfig()
        self.database_url = database_url
        self.recorder = recorder or Phase3Recorder(database_url=database_url)

    def build_symbol_observations(
        self,
        *,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[StrategicObservation]:
        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)
        if end_utc < start_utc:
            raise ValueError("end must be greater than or equal to start")

        bundle = get_phase2_signal_bundle(
            symbol=symbol,
            start=start_utc,
            end=end_utc,
            sentiment_lane=self.config.sentiment_lane,
        )
        merged = self._merge_bundle(bundle)
        if merged.empty:
            return []

        observations: list[StrategicObservation] = []
        for row in merged.itertuples(index=False):
            ts = _ensure_utc(getattr(row, "timestamp"))
            quality_status = "pass"
            if pd.isna(getattr(row, "price_forecast")) or pd.isna(getattr(row, "consensus_confidence")):
                quality_status = "warn"
            snapshot_id = f"{symbol}:{ts.isoformat()}:{self.config.observation_schema_version}"
            observation = StrategicObservation(
                timestamp=ts,
                symbol=symbol,
                snapshot_id=snapshot_id,
                technical_direction=_safe_str(getattr(row, "technical_direction"), default="neutral"),
                technical_confidence=_safe_float(getattr(row, "technical_confidence"), default=0.0),
                price_forecast=_safe_float(getattr(row, "price_forecast"), default=0.0),
                var_95=_safe_float(getattr(row, "var_95"), default=0.0),
                es_95=_safe_float(getattr(row, "es_95"), default=0.0),
                regime_state=_safe_str(getattr(row, "regime_state"), default="UNKNOWN"),
                regime_transition_prob=_safe_float(getattr(row, "regime_transition_prob"), default=0.0),
                sentiment_score=_safe_optional_float(getattr(row, "sentiment_score")),
                sentiment_z_t=_safe_optional_float(getattr(row, "sentiment_z_t")),
                consensus_direction=_safe_str(getattr(row, "consensus_direction"), default="HOLD"),
                consensus_confidence=_safe_float(getattr(row, "consensus_confidence"), default=0.0),
                crisis_mode=_safe_bool(getattr(row, "crisis_mode"), default=False),
                agent_divergence=_safe_bool(getattr(row, "agent_divergence"), default=False),
                orderbook_imbalance=_safe_optional_float(getattr(row, "orderbook_imbalance", None)),
                queue_pressure=_safe_optional_float(getattr(row, "queue_pressure", None)),
                current_position=0.0,
                unrealized_pnl=0.0,
                notional_exposure=0.0,
                portfolio_features={},
                observation_schema_version=self.config.observation_schema_version,
                quality_status=quality_status,
            )
            observations.append(observation)
        return observations

    def materialize_observation_batches(
        self,
        *,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
        batch_size: int = 500,
        dry_run: bool = False,
    ) -> MaterializationSummary:
        symbol_list = [s.strip() for s in symbols if s and s.strip()]
        built_total = 0
        materialized_total = 0
        warn_total = 0
        per_symbol: dict[str, int] = {}

        for symbol in symbol_list:
            observations = self.build_symbol_observations(symbol=symbol, start=start, end=end)
            per_symbol[symbol] = len(observations)
            built_total += len(observations)
            warn_total += sum(1 for obs in observations if obs.quality_status != "pass")

            if dry_run or not observations:
                continue
            for chunk in _chunked(observations, size=batch_size):
                payload = [item.model_dump(mode="python") for item in chunk]
                self.recorder.save_observation_batch(payload)
                materialized_total += len(chunk)

        return MaterializationSummary(
            symbols=symbol_list,
            rows_built=built_total,
            rows_materialized=materialized_total,
            rows_with_warn_quality=warn_total,
            per_symbol_counts=per_symbol,
        )

    def _merge_bundle(self, bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for key in ("technical", "regime", "sentiment", "consensus"):
            frame = bundle.get(key)
            if frame is None or frame.empty or "timestamp" not in frame.columns:
                continue
            sample = frame[["timestamp"]].copy()
            sample["timestamp"] = pd.to_datetime(sample["timestamp"], utc=True, errors="coerce")
            sample = sample.dropna(subset=["timestamp"])
            if sample.empty:
                continue
            frames.append(sample)
        if not frames:
            return pd.DataFrame()

        base = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        technical = _prepare_frame(
            bundle.get("technical"),
            {
                "direction": "technical_direction",
                "confidence": "technical_confidence",
                "price_forecast": "price_forecast",
                "var_95": "var_95",
                "es_95": "es_95",
            },
        )
        regime = _prepare_frame(
            bundle.get("regime"),
            {
                "regime_state": "regime_state",
                "transition_probability": "regime_transition_prob",
            },
        )
        sentiment = _prepare_frame(
            bundle.get("sentiment"),
            {
                "sentiment_score": "sentiment_score",
                "z_t": "sentiment_z_t",
            },
        )
        consensus = _prepare_frame(
            bundle.get("consensus"),
            {
                "final_direction": "consensus_direction",
                "final_confidence": "consensus_confidence",
                "crisis_mode": "crisis_mode",
                "agent_divergence": "agent_divergence",
            },
        )

        merged = base
        tolerance = self.config.max_signal_staleness
        for frame in (technical, regime, sentiment, consensus):
            if frame.empty:
                continue
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                frame.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=tolerance,
            )

        for name, default in (
            ("technical_direction", "neutral"),
            ("technical_confidence", 0.0),
            ("price_forecast", 0.0),
            ("var_95", 0.0),
            ("es_95", 0.0),
            ("regime_state", "UNKNOWN"),
            ("regime_transition_prob", 0.0),
            ("consensus_direction", "HOLD"),
            ("consensus_confidence", 0.0),
            ("crisis_mode", False),
            ("agent_divergence", False),
        ):
            if name not in merged.columns:
                merged[name] = default
        return merged


def _prepare_frame(df: pd.DataFrame | None, mapping: dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    available = ["timestamp"] + [src for src in mapping if src in df.columns]
    if len(available) <= 1:
        return pd.DataFrame()
    out = df[available].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.rename(columns={src: dst for src, dst in mapping.items() if src in out.columns})


def _chunked(values: list[StrategicObservation], *, size: int) -> Iterable[list[StrategicObservation]]:
    size = max(1, int(size))
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def _ensure_utc(value: datetime | Any) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime, got {type(value)!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_str(value: Any, *, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return str(value)


def _safe_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return bool(value)
