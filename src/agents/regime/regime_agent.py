from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.db.phase2_recorder import Phase2Recorder
from src.agents.regime.data_loader import RegimeDataLoader
from src.agents.regime.models import HMMRegimeModel, OODDetector, PearlMetaModel
from src.agents.regime.schemas import RegimePrediction, RegimeState, RiskLevel

logger = logging.getLogger(__name__)


class RegimeAgent:
    """
    Unified Regime Agent orchestrator.

    Ensemble:
    - HMM baseline for regime-state estimation and transitions.
    - PEARL-like adaptive model for fast context shift adaptation.
    - OOD detector for warning/alien overrides and staged de-risking.
    """

    def __init__(
        self,
        loader: RegimeDataLoader | None = None,
        model_id: str = "ensemble_hmm_pearl_ood_v1.0",
        warmup_rows: int = 120,
        hmm_model: HMMRegimeModel | None = None,
        pearl_model: PearlMetaModel | None = None,
        ood_detector: OODDetector | None = None,
        database_url: str | None = None,
        persist_predictions: bool = True,
    ) -> None:
        self.loader = loader or RegimeDataLoader()
        self.model_id = model_id
        self.warmup_rows = int(max(40, warmup_rows))
        self.hmm = hmm_model or HMMRegimeModel(n_components=4, random_state=42)
        self.pearl = pearl_model or PearlMetaModel(adaptation_window=60, max_adaptation_weight=0.7)
        self.ood = ood_detector or OODDetector(
            warning_mahalanobis=3.0,
            alien_mahalanobis=5.0,
            warning_kl=1.0,
            alien_kl=6.0,
        )
        self.phase2_recorder = Phase2Recorder(database_url) if persist_predictions else None

    def _persist_prediction(self, prediction: RegimePrediction, *, data_snapshot_id: str | None = None) -> None:
        if self.phase2_recorder is None:
            return
        try:
            self.phase2_recorder.save_regime_prediction(prediction, data_snapshot_id=data_snapshot_id)
        except Exception as exc:
            logger.warning("Failed to persist regime prediction for %s: %s", prediction.symbol, exc)

    def detect_regime(
        self,
        symbol: str,
        limit: int = 500,
        *,
        data_snapshot_id: str | None = None,
    ) -> RegimePrediction:
        raw = self.loader.load_features(symbol=symbol, limit=limit)
        now = datetime.now(timezone.utc)
        if raw.empty:
            prediction = RegimePrediction(
                symbol=symbol,
                timestamp=now,
                regime_state=RegimeState.SIDEWAYS,
                transition_probability=0.0,
                confidence=0.0,
                risk_level=RiskLevel.REDUCED_RISK,
                model_id=self.model_id,
                details={"reason": "no_gold_features_available"},
            )
            self._persist_prediction(prediction, data_snapshot_id=data_snapshot_id)
            return prediction

        prepared = self._prepare_features(raw)
        if len(prepared) < self.warmup_rows:
            fallback = self._heuristic_prediction(prepared)
            prediction = RegimePrediction(
                symbol=symbol,
                timestamp=now,
                regime_state=fallback["regime_state"],
                transition_probability=fallback["transition_probability"],
                confidence=fallback["confidence"],
                risk_level=self._risk_level_from_state(fallback["regime_state"], fallback["confidence"]),
                model_id=f"{self.model_id}_fallback",
                details={"reason": "insufficient_warmup_rows", "rows_used": int(len(prepared))},
            )
            self._persist_prediction(prediction, data_snapshot_id=data_snapshot_id)
            return prediction

        if not self.hmm.fitted:
            self.hmm.fit(prepared)
        if not self.pearl.fitted:
            self.pearl.fit(prepared)
        if not self.ood.fitted:
            self.ood.fit(prepared)

        hmm_pred = self.hmm.predict(prepared)
        pearl_pred = self.pearl.predict(prepared)
        ood_result = self.ood.detect(prepared.tail(min(120, len(prepared))))
        latest = prepared.iloc[-1]

        regime_state = self._ensemble_state(hmm_pred.regime_state, pearl_pred.regime_state, hmm_pred.confidence, pearl_pred.confidence)
        if ood_result.is_alien:
            regime_state = RegimeState.ALIEN

        transition_probability = self._blend_transition(hmm_pred.transition_probability, pearl_pred.transition_probability, ood_result.is_warning)
        confidence = self._blend_confidence(hmm_pred.confidence, pearl_pred.confidence, ood_result.is_warning, ood_result.is_alien)

        risk_level = ood_result.risk_level
        if risk_level == RiskLevel.FULL_RISK:
            risk_level = self._risk_level_from_state(regime_state, confidence)

        prediction = RegimePrediction(
            symbol=symbol,
            timestamp=now,
            regime_state=regime_state,
            transition_probability=transition_probability,
            confidence=confidence,
            risk_level=risk_level,
            model_id=self.model_id,
            details={
                "rows_used": int(len(prepared)),
                "hmm": {
                    "regime_state": hmm_pred.regime_state.value,
                    "transition_probability": hmm_pred.transition_probability,
                    "confidence": hmm_pred.confidence,
                    "hidden_state": hmm_pred.hidden_state,
                },
                "pearl": {
                    "regime_state": pearl_pred.regime_state.value,
                    "transition_probability": pearl_pred.transition_probability,
                    "confidence": pearl_pred.confidence,
                    "probabilities": pearl_pred.probabilities,
                },
                "ood": {
                    "warning": ood_result.is_warning,
                    "alien": ood_result.is_alien,
                    "mahalanobis_distance": ood_result.mahalanobis_distance,
                    "kl_divergence": ood_result.kl_divergence,
                },
                "macro_features": {
                    "macro_directional_flag": int(float(latest.get("macro_directional_flag", 0.0))),
                    "macro_regime_index": round(float(latest.get("macro_regime_index", 0.0)), 6),
                    "macro_regime_shock": round(float(latest.get("macro_regime_shock", 0.0)), 6),
                    "macro_coverage_ratio": round(float(latest.get("macro_coverage_ratio", 0.0)), 6),
                },
            },
        )
        self._persist_prediction(prediction, data_snapshot_id=data_snapshot_id)
        return prediction

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "close_log_return" not in out.columns and "close" in out.columns:
            close = pd.to_numeric(out["close"], errors="coerce")
            out["close_log_return"] = np.log(close / close.shift(1))

        out["close_log_return"] = pd.to_numeric(out.get("close_log_return"), errors="coerce")
        if "close_log_return_zscore" not in out.columns:
            ret = out["close_log_return"]
            rolling = ret.rolling(window=30, min_periods=5)
            mean = rolling.mean()
            std = rolling.std().replace(0, np.nan)
            out["close_log_return_zscore"] = (ret - mean) / std

        if "rolling_vol_20" not in out.columns:
            out["rolling_vol_20"] = out["close_log_return"].rolling(window=20, min_periods=5).std()

        out = RegimeAgent._build_macro_features(out)

        for col in ["close_log_return_zscore", "rolling_vol_20", "macro_directional_flag", "macro_regime_index", "macro_regime_shock"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "timestamp" in out.columns:
            out = out.sort_values("timestamp")
        return out.reset_index(drop=True)

    @staticmethod
    def _build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        macro_weights = {
            "CPI": -0.18,
            "WPI": -0.16,
            "IIP": 0.16,
            "FII_FLOW": 0.18,
            "DII_FLOW": 0.10,
            "FX_RESERVES": 0.12,
            "INDIA_US_10Y_SPREAD": -0.12,
            "REPO_RATE": -0.14,
            "US_10Y": -0.12,
            "RBI_BULLETIN": 0.04,
        }

        weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
        available_weight = pd.Series(0.0, index=out.index, dtype=float)
        total_abs_weight = float(sum(abs(weight) for weight in macro_weights.values()))

        for col, weight in macro_weights.items():
            if col not in out.columns:
                out[col] = np.nan

            raw = pd.to_numeric(out[col], errors="coerce")
            ffilled = raw.ffill()
            rolling_mean = ffilled.rolling(window=120, min_periods=12).mean()
            rolling_std = ffilled.rolling(window=120, min_periods=12).std().replace(0, np.nan)
            zscore = ((ffilled - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)

            presence = raw.notna().astype(float)
            weighted_sum += zscore * float(weight) * presence
            available_weight += abs(float(weight)) * presence

            out[f"{col}_z"] = zscore

        out["macro_coverage_ratio"] = (available_weight / max(total_abs_weight, 1e-9)).clip(0.0, 1.0)
        out["macro_regime_index"] = (
            weighted_sum / available_weight.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        out["macro_regime_shock"] = out["macro_regime_index"].diff().fillna(0.0).clip(-5.0, 5.0)

        # Preserve upstream directional flag if present and informative, otherwise derive from macro composite.
        upstream_flag = pd.to_numeric(out.get("macro_directional_flag", 0.0), errors="coerce").fillna(0.0)
        derived_flag = np.select(
            [out["macro_regime_shock"] >= 0.15, out["macro_regime_shock"] <= -0.15],
            [1, -1],
            default=0,
        )
        out["macro_directional_flag"] = np.where(upstream_flag != 0.0, upstream_flag, derived_flag)
        return out

    @staticmethod
    def _heuristic_prediction(df: pd.DataFrame) -> dict[str, RegimeState | float]:
        latest = df.iloc[-1]
        ret_z = float(latest.get("close_log_return_zscore", 0.0))
        vol = float(latest.get("rolling_vol_20", 0.0))
        macro = float(latest.get("macro_directional_flag", 0.0))

        if abs(ret_z) >= 4.5 or vol >= 0.06:
            state = RegimeState.ALIEN
        elif abs(ret_z) >= 3.0 or vol >= 0.03:
            state = RegimeState.CRISIS
        elif ret_z <= -1.0:
            state = RegimeState.BEAR
        elif ret_z >= 1.0:
            state = RegimeState.BULL
        elif abs(macro) >= 0.5:
            state = RegimeState.RBI_BAND_TRANSITION
        else:
            state = RegimeState.SIDEWAYS

        confidence = float(np.clip((abs(ret_z) / 3.0 + vol / 0.03) / 2.0, 0.1, 0.95))
        transition_probability = float(np.clip(abs(ret_z) / 3.0, 0.0, 1.0))
        return {
            "regime_state": state,
            "confidence": round(confidence, 4),
            "transition_probability": round(transition_probability, 4),
        }

    @staticmethod
    def _ensemble_state(
        hmm_state: RegimeState,
        pearl_state: RegimeState,
        hmm_conf: float,
        pearl_conf: float,
    ) -> RegimeState:
        scores = {
            hmm_state: float(np.clip(hmm_conf, 0.0, 1.0)),
            pearl_state: float(np.clip(pearl_conf, 0.0, 1.0)),
        }
        if hmm_state == pearl_state:
            scores[hmm_state] += 0.15
        return max(scores, key=scores.get)

    @staticmethod
    def _blend_transition(hmm_tp: float, pearl_tp: float, warning: bool) -> float:
        blended = 0.6 * float(hmm_tp) + 0.4 * float(pearl_tp)
        if warning:
            blended = max(blended, 0.7)
        return float(round(np.clip(blended, 0.0, 1.0), 4))

    @staticmethod
    def _blend_confidence(hmm_conf: float, pearl_conf: float, warning: bool, alien: bool) -> float:
        conf = 0.55 * float(hmm_conf) + 0.45 * float(pearl_conf)
        if warning:
            conf *= 0.9
        if alien:
            conf *= 0.8
        return float(round(np.clip(conf, 0.05, 0.99), 4))

    @staticmethod
    def _risk_level_from_state(regime_state: RegimeState, confidence: float) -> RiskLevel:
        if regime_state in {RegimeState.ALIEN, RegimeState.CRISIS}:
            return RiskLevel.NEUTRAL_CASH
        if regime_state in {RegimeState.BEAR, RegimeState.RBI_BAND_TRANSITION}:
            return RiskLevel.REDUCED_RISK
        if confidence < 0.25:
            return RiskLevel.REDUCED_RISK
        return RiskLevel.FULL_RISK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Regime Agent inference for one symbol.")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Target symbol (default: RELIANCE.NS)")
    parser.add_argument("--limit", type=int, default=500, help="Max Gold feature rows to load (default: 500)")
    parser.add_argument("--database-url", default=None, help="Optional override DATABASE_URL")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    loader = RegimeDataLoader(database_url=args.database_url)
    agent = RegimeAgent(loader=loader)
    prediction = agent.detect_regime(symbol=args.symbol, limit=args.limit)
    print(json.dumps(prediction.model_dump(mode="json"), indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
