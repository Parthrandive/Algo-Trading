from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.agents.strategic.config import ObservationAssemblerConfig
from src.agents.strategic.schemas import OBSERVATION_FEATURE_NAMES, StrategicObservation

_DIRECTION_MAP = {"up": 1.0, "down": -1.0, "neutral": 0.0, "buy": 1.0, "sell": -1.0, "hold": 0.0}
_REGIME_TO_KEY = {
    "bull": "regime_bull",
    "bear": "regime_bear",
    "sideways": "regime_sideways",
    "crisis": "regime_crisis",
    "rbi-band transition": "regime_rbi_band_transition",
    "rbi_band_transition": "regime_rbi_band_transition",
    "alien": "regime_alien",
}


@dataclass
class ObservationAssembler:
    config: ObservationAssemblerConfig = ObservationAssemblerConfig()

    def assemble_from_frames(
        self,
        *,
        technical: pd.DataFrame,
        regime: pd.DataFrame,
        sentiment: pd.DataFrame,
        consensus: pd.DataFrame,
        portfolio: pd.DataFrame | None = None,
    ) -> list[StrategicObservation]:
        technical_df = self._prepare_frame(technical, "technical")
        regime_df = self._prepare_frame(regime, "regime")
        sentiment_df = self._prepare_frame(sentiment, "sentiment")
        consensus_df = self._prepare_frame(consensus, "consensus")
        portfolio_df = self._prepare_frame(
            portfolio if portfolio is not None else self._default_portfolio_frame(technical_df),
            "portfolio",
        )

        base = technical_df[["symbol", "timestamp"]].drop_duplicates().sort_values(["symbol", "timestamp"])
        merged = self._asof_merge(base, technical_df, "technical")
        merged = self._asof_merge(merged, regime_df, "regime")
        merged = self._asof_merge(merged, sentiment_df, "sentiment")
        merged = self._asof_merge(merged, consensus_df, "consensus")
        merged = self._asof_merge(merged, portfolio_df, "portfolio")

        observations: list[StrategicObservation] = []
        for row in merged.to_dict(orient="records"):
            observations.append(self._build_observation(row))
        return observations

    def assemble_from_records(
        self,
        *,
        technical: list[dict[str, Any]],
        regime: list[dict[str, Any]],
        sentiment: list[dict[str, Any]],
        consensus: list[dict[str, Any]],
        portfolio: list[dict[str, Any]] | None = None,
    ) -> list[StrategicObservation]:
        return self.assemble_from_frames(
            technical=pd.DataFrame(technical),
            regime=pd.DataFrame(regime),
            sentiment=pd.DataFrame(sentiment),
            consensus=pd.DataFrame(consensus),
            portfolio=pd.DataFrame(portfolio or []),
        )

    def _build_observation(self, row: dict[str, Any]) -> StrategicObservation:
        feature_map = {name: self.config.fill_value for name in OBSERVATION_FEATURE_NAMES}
        feature_map["price_forecast"] = self._float(row.get("technical_price_forecast"))
        feature_map["direction_score"] = self._direction_to_score(row.get("technical_direction"))
        feature_map["var_95"] = self._float(row.get("technical_var_95"))
        feature_map["es_95"] = self._float(row.get("technical_es_95"))
        feature_map["transition_probability"] = self._float(row.get("regime_transition_probability"))
        feature_map["sentiment_score"] = self._float(row.get("sentiment_sentiment_score"))
        feature_map["z_t"] = self._float(row.get("sentiment_z_t"))
        feature_map["final_direction_score"] = self._direction_to_score(row.get("consensus_final_direction"))
        feature_map["final_confidence"] = self._float(row.get("consensus_final_confidence"))
        feature_map["crisis_mode_flag"] = 1.0 if bool(row.get("consensus_crisis_mode")) else 0.0
        feature_map["current_position"] = self._float(row.get("portfolio_current_position"))
        feature_map["unrealized_pnl"] = self._float(row.get("portfolio_unrealized_pnl"))
        feature_map.update(self._encode_regime(row.get("regime_regime_state")))

        source_timestamps = {
            "technical": self._coerce_timestamp(row.get("technical_timestamp") or row.get("timestamp")),
            "regime": self._coerce_timestamp(row.get("regime_timestamp")),
            "sentiment": self._coerce_timestamp(row.get("sentiment_timestamp")),
            "consensus": self._coerce_timestamp(row.get("consensus_timestamp")),
            "portfolio": self._coerce_timestamp(row.get("portfolio_timestamp")),
        }

        return StrategicObservation(
            symbol=str(row["symbol"]),
            timestamp=self._coerce_timestamp(row["timestamp"]),
            observation_vector=tuple(feature_map[name] for name in OBSERVATION_FEATURE_NAMES),
            technical_model_id=self._optional_str(row.get("technical_model_id")),
            regime_model_id=self._optional_str(row.get("regime_model_id")),
            sentiment_model_id=self._optional_str(row.get("sentiment_model_id")),
            consensus_model_id=self._optional_str(row.get("consensus_model_id")),
            alignment_tolerance_seconds=float(self.config.alignment_tolerance_seconds),
            source_timestamps={k: v for k, v in source_timestamps.items() if v is not None},
            metadata={
                "missing_flags": {
                    "regime": row.get("regime_regime_state") is None,
                    "sentiment": row.get("sentiment_sentiment_score") is None,
                    "consensus": row.get("consensus_final_direction") is None,
                    "portfolio": row.get("portfolio_current_position") is None,
                }
            },
        )

    def _asof_merge(self, left: pd.DataFrame, right: pd.DataFrame, prefix: str) -> pd.DataFrame:
        right_df = right.sort_values(["symbol", "timestamp"]).copy()
        renamed = {col: f"{prefix}_{col}" for col in right_df.columns if col not in {"symbol", "timestamp"}}
        right_df = right_df.rename(columns=renamed)
        right_df[f"{prefix}_timestamp"] = right_df["timestamp"]
        return pd.merge_asof(
            left.sort_values(["symbol", "timestamp"]),
            right_df.sort_values(["symbol", "timestamp"]),
            on="timestamp",
            by="symbol",
            direction="backward",
            tolerance=pd.Timedelta(seconds=self.config.alignment_tolerance_seconds),
        )

    def _prepare_frame(self, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return self._default_empty_frame(prefix)
        prepared = frame.copy()
        if "timestamp" not in prepared.columns or "symbol" not in prepared.columns:
            raise ValueError(f"{prefix} frame must include symbol and timestamp columns")
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
        return prepared.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    def _default_portfolio_frame(self, technical_df: pd.DataFrame) -> pd.DataFrame:
        return technical_df[["symbol", "timestamp"]].assign(current_position=0.0, unrealized_pnl=0.0)

    @staticmethod
    def _default_empty_frame(prefix: str) -> pd.DataFrame:
        if prefix == "portfolio":
            return pd.DataFrame(columns=["symbol", "timestamp", "current_position", "unrealized_pnl"])
        return pd.DataFrame(columns=["symbol", "timestamp"])

    @staticmethod
    def _direction_to_score(value: Any) -> float:
        if value is None:
            return 0.0
        return _DIRECTION_MAP.get(str(value).strip().lower(), 0.0)

    @staticmethod
    def _encode_regime(value: Any) -> dict[str, float]:
        encoded = {name: 0.0 for name in OBSERVATION_FEATURE_NAMES if name.startswith("regime_")}
        if value is None:
            encoded["regime_sideways"] = 1.0
            return encoded
        key = _REGIME_TO_KEY.get(str(value).strip().lower(), "regime_sideways")
        encoded[key] = 1.0
        return encoded

    @staticmethod
    def _float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime().astimezone(UTC)
