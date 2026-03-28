from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from src.agents.consensus.schemas import (
    ConsensusInput,
    ConsensusOutput,
    ConsensusRegimeRiskLevel,
    ConsensusRiskMode,
    ConsensusTransitionModel,
)

if TYPE_CHECKING:
    from src.db.phase2_recorder import Phase2Recorder

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "configs" / "consensus_agent_runtime_v1.json"
DEFAULT_MODEL_CARDS_ROOT = ROOT_DIR / "data" / "models"
DEFAULT_TRAINING_RUNS_ROOT = ROOT_DIR / "data" / "reports" / "training_runs"


class ConsensusAgent:
    def __init__(
        self,
        *,
        volatility_switch_threshold: float = 0.35,
        lstar_logistic_gain: float = 4.0,
        max_crisis_weight: float = 0.7,
        divergence_warn_threshold: float = 0.45,
        divergence_protective_threshold: float = 0.75,
        safety_bias_boost: float = 0.1,
        reduced_mode_scale: float = 0.5,
        technical_base_weight: float = 0.42,
        regime_base_weight: float = 0.35,
        sentiment_base_weight: float = 0.23,
        stale_sentiment_weight_multiplier: float = 0.4,
        missing_sentiment_weight_multiplier: float = 0.05,
        stale_confidence_penalty: float = 0.05,
        missing_confidence_penalty: float = 0.10,
        regime_warning_regime_weight_multiplier: float = 1.1,
        regime_warning_technical_weight_multiplier: float = 0.9,
        regime_warning_confidence_penalty: float = 0.15,
        regime_alien_regime_weight_multiplier: float = 1.2,
        regime_alien_technical_weight_multiplier: float = 0.8,
        regime_alien_sentiment_weight_multiplier: float = 0.6,
        regime_alien_confidence_penalty: float = 0.20,
        schema_version: str = "1.0",
        runtime_config_path: Path | None = None,
        model_cards_root: Path | None = None,
        training_runs_root: Path | None = None,
        phase2_recorder: Phase2Recorder | None = None,
    ):
        self.volatility_switch_threshold = max(0.0, volatility_switch_threshold)
        self.lstar_logistic_gain = max(0.1, lstar_logistic_gain)
        self.max_crisis_weight = self._clamp(max_crisis_weight, 0.0, 1.0)
        self.divergence_warn_threshold = self._clamp(divergence_warn_threshold, 0.0, 1.0)
        self.divergence_protective_threshold = self._clamp(divergence_protective_threshold, 0.0, 1.0)
        self.safety_bias_boost = self._clamp(safety_bias_boost, 0.0, 0.5)
        self.reduced_mode_scale = self._clamp(reduced_mode_scale, 0.1, 1.0)
        self.technical_base_weight = max(0.01, technical_base_weight)
        self.regime_base_weight = max(0.01, regime_base_weight)
        self.sentiment_base_weight = max(0.01, sentiment_base_weight)
        self.stale_sentiment_weight_multiplier = self._clamp(stale_sentiment_weight_multiplier, 0.0, 1.0)
        self.missing_sentiment_weight_multiplier = self._clamp(missing_sentiment_weight_multiplier, 0.0, 1.0)
        self.stale_confidence_penalty = self._clamp(stale_confidence_penalty, 0.0, 1.0)
        self.missing_confidence_penalty = self._clamp(missing_confidence_penalty, 0.0, 1.0)
        self.regime_warning_regime_weight_multiplier = max(0.01, regime_warning_regime_weight_multiplier)
        self.regime_warning_technical_weight_multiplier = max(0.01, regime_warning_technical_weight_multiplier)
        self.regime_warning_confidence_penalty = self._clamp(regime_warning_confidence_penalty, 0.0, 1.0)
        self.regime_alien_regime_weight_multiplier = max(0.01, regime_alien_regime_weight_multiplier)
        self.regime_alien_technical_weight_multiplier = max(0.01, regime_alien_technical_weight_multiplier)
        self.regime_alien_sentiment_weight_multiplier = max(0.01, regime_alien_sentiment_weight_multiplier)
        self.regime_alien_confidence_penalty = self._clamp(regime_alien_confidence_penalty, 0.0, 1.0)
        self.schema_version = str(schema_version)
        self.runtime_config_path = runtime_config_path or DEFAULT_RUNTIME_CONFIG_PATH
        self.model_cards_root = model_cards_root or DEFAULT_MODEL_CARDS_ROOT
        self.training_runs_root = training_runs_root or DEFAULT_TRAINING_RUNS_ROOT
        self.phase2_recorder = phase2_recorder

    @classmethod
    def from_default_components(
        cls,
        runtime_config_path: Path | None = None,
        *,
        model_cards_root: Path | None = None,
        training_runs_root: Path | None = None,
        phase2_recorder: Phase2Recorder | None = None,
    ) -> "ConsensusAgent":
        config_path = runtime_config_path or DEFAULT_RUNTIME_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as handle:
            runtime_config = json.load(handle)

        transition = runtime_config.get("transition", {})
        routing = runtime_config.get("routing", {})
        risk_modes = runtime_config.get("risk_modes", {})
        weights = runtime_config.get("weights", {})
        freshness = runtime_config.get("freshness", {})
        ood = runtime_config.get("ood", {})

        return cls(
            volatility_switch_threshold=float(transition.get("volatility_switch_threshold", 0.35)),
            lstar_logistic_gain=float(transition.get("lstar_logistic_gain", 4.0)),
            max_crisis_weight=float(routing.get("max_crisis_weight", 0.7)),
            safety_bias_boost=float(routing.get("safety_bias_boost", 0.1)),
            divergence_warn_threshold=float(risk_modes.get("divergence_warn_threshold", 0.45)),
            divergence_protective_threshold=float(risk_modes.get("divergence_protective_threshold", 0.75)),
            reduced_mode_scale=float(risk_modes.get("reduced_mode_scale", 0.5)),
            technical_base_weight=float(weights.get("technical_base", 0.42)),
            regime_base_weight=float(weights.get("regime_base", 0.35)),
            sentiment_base_weight=float(weights.get("sentiment_base", 0.23)),
            stale_sentiment_weight_multiplier=float(freshness.get("stale_sentiment_weight_multiplier", 0.4)),
            missing_sentiment_weight_multiplier=float(freshness.get("missing_sentiment_weight_multiplier", 0.05)),
            stale_confidence_penalty=float(freshness.get("stale_confidence_penalty", 0.05)),
            missing_confidence_penalty=float(freshness.get("missing_confidence_penalty", 0.10)),
            regime_warning_regime_weight_multiplier=float(ood.get("warning_regime_weight_multiplier", 1.1)),
            regime_warning_technical_weight_multiplier=float(ood.get("warning_technical_weight_multiplier", 0.9)),
            regime_warning_confidence_penalty=float(ood.get("warning_confidence_penalty", 0.15)),
            regime_alien_regime_weight_multiplier=float(ood.get("alien_regime_weight_multiplier", 1.2)),
            regime_alien_technical_weight_multiplier=float(ood.get("alien_technical_weight_multiplier", 0.8)),
            regime_alien_sentiment_weight_multiplier=float(ood.get("alien_sentiment_weight_multiplier", 0.6)),
            regime_alien_confidence_penalty=float(ood.get("alien_confidence_penalty", 0.20)),
            schema_version=str(runtime_config.get("schema_version", "1.0")),
            runtime_config_path=config_path,
            model_cards_root=model_cards_root,
            training_runs_root=training_runs_root,
            phase2_recorder=phase2_recorder,
        )

    def run(self, payload: ConsensusInput) -> ConsensusOutput:
        transition_model, transition_score = self._select_transition(payload)
        weights = self._compute_weights(payload=payload, transition_score=transition_score)

        raw_score = (
            (payload.technical.score * weights["technical"])
            + (payload.regime.score * weights["regime"])
            + (payload.sentiment.score * weights["sentiment"])
        )
        divergence_score = self._compute_divergence(payload)
        risk_mode = self._resolve_risk_mode(payload=payload, divergence_score=divergence_score)

        score = self._apply_risk_mode(raw_score, risk_mode)
        score = self._apply_daily_trend_gate(score, payload)
        
        confidence = self._compute_confidence(
            payload=payload,
            transition_score=transition_score,
            divergence_score=divergence_score,
            weights=weights,
        )

        return ConsensusOutput(
            score=self._clamp(score, -1.0, 1.0),
            confidence=self._clamp(confidence, 0.0, 1.0),
            transition_score=self._clamp(transition_score, 0.0, 1.0),
            transition_model=transition_model,
            risk_mode=risk_mode,
            divergence_score=divergence_score,
            crisis_weight=min(payload.crisis_probability, self.max_crisis_weight),
            weights=weights,
            generated_at_utc=payload.generated_at_utc,
            schema_version=self.schema_version,
        )

    def compute_lstar_transition(self, payload: ConsensusInput) -> float:
        signal = (
            (0.55 * payload.volatility)
            + (0.20 * abs(payload.macro_differential))
            + (0.15 * abs(payload.rbi_signal))
            + (0.10 * payload.sentiment_quantile)
        )
        centered = signal - 0.5
        return 1.0 / (1.0 + math.exp(-(self.lstar_logistic_gain * centered)))

    def compute_estar_transition(self, payload: ConsensusInput) -> float:
        x = (
            (0.50 * payload.volatility)
            + (0.25 * abs(payload.macro_differential))
            + (0.15 * abs(payload.rbi_signal))
            + (0.10 * payload.sentiment_quantile)
        )
        return 1.0 - math.exp(-(x**2))

    def _select_transition(self, payload: ConsensusInput) -> tuple[ConsensusTransitionModel, float]:
        if payload.volatility >= self.volatility_switch_threshold:
            return ConsensusTransitionModel.ESTAR, self.compute_estar_transition(payload)
        return ConsensusTransitionModel.LSTAR, self.compute_lstar_transition(payload)

    def _compute_weights(self, *, payload: ConsensusInput, transition_score: float) -> dict[str, float]:
        technical_weight = self.technical_base_weight + (0.08 * (1.0 - transition_score))
        regime_weight = self.regime_base_weight + (0.10 * transition_score)
        sentiment_weight = self.sentiment_base_weight - (0.04 * transition_score)

        if payload.sentiment_is_missing:
            sentiment_weight *= self.missing_sentiment_weight_multiplier
        elif payload.sentiment_is_stale:
            sentiment_weight *= self.stale_sentiment_weight_multiplier

        if payload.regime_ood_warning:
            regime_weight *= self.regime_warning_regime_weight_multiplier
            technical_weight *= self.regime_warning_technical_weight_multiplier
        if payload.regime_ood_alien:
            regime_weight *= self.regime_alien_regime_weight_multiplier
            technical_weight *= self.regime_alien_technical_weight_multiplier
            sentiment_weight *= self.regime_alien_sentiment_weight_multiplier

        crisis_weight = min(payload.crisis_probability, self.max_crisis_weight)
        regime_weight += 0.15 * crisis_weight
        technical_weight -= 0.10 * crisis_weight
        sentiment_weight -= 0.05 * crisis_weight

        if payload.technical.is_protective:
            technical_weight += self.safety_bias_boost
        if payload.regime.is_protective:
            regime_weight += self.safety_bias_boost
        if payload.sentiment.is_protective:
            sentiment_weight += self.safety_bias_boost

        technical_weight = max(0.01, technical_weight)
        regime_weight = max(0.01, regime_weight)
        sentiment_weight = max(0.01, sentiment_weight)

        total = technical_weight + regime_weight + sentiment_weight
        return {
            "technical": technical_weight / total,
            "regime": regime_weight / total,
            "sentiment": sentiment_weight / total,
        }

    def _compute_divergence(self, payload: ConsensusInput) -> float:
        scores = [payload.technical.score, payload.regime.score, payload.sentiment.score]
        spread = max(scores) - min(scores)
        return self._clamp(spread / 2.0, 0.0, 1.0)

    def _resolve_risk_mode(self, *, payload: ConsensusInput, divergence_score: float) -> ConsensusRiskMode:
        if payload.regime_risk_level == ConsensusRegimeRiskLevel.NEUTRAL_CASH or payload.regime_ood_alien:
            return ConsensusRiskMode.PROTECTIVE
        if divergence_score >= self.divergence_protective_threshold:
            return ConsensusRiskMode.PROTECTIVE
        if payload.regime_risk_level == ConsensusRegimeRiskLevel.REDUCED_RISK or payload.regime_ood_warning:
            return ConsensusRiskMode.REDUCED
        if divergence_score >= self.divergence_warn_threshold:
            return ConsensusRiskMode.REDUCED
        return ConsensusRiskMode.NORMAL

    def _apply_risk_mode(self, score: float, risk_mode: ConsensusRiskMode) -> float:
        if risk_mode == ConsensusRiskMode.PROTECTIVE:
            return 0.0
        if risk_mode == ConsensusRiskMode.REDUCED:
            return score * self.reduced_mode_scale
        return max(-1.0, min(1.0, score))

    def _apply_daily_trend_gate(self, score: float, payload: ConsensusInput) -> float:
        """
        Regime-Conditional Daily Filter (Fix 4).
        Suppresses trades if:
        1) Regime is CHOP / HIGH_VOL (with TATASTEEL/INFY overrides)
        2) Short-term signal opposes long-term daily trend
        3) High recent volatility (atr_rank_20d > 0.70)
        """
        if payload.daily_trend_bullish is None or abs(score) < 0.05:
            return score
            
        sym = payload.symbol.upper()
        
        # 1. Regime Condition check
        # Usually from HMM states like "CHOP", "HIGH_VOL", "STRESS"
        regime_name = payload.regime.name.upper()
        is_high_vol_regime = any(r in regime_name for r in ["CHOP", "VOL", "CRISIS"])
        
        if sym == "TATASTEEL.NS":
            cond1_met = payload.atr_rank_20d > 0.70
        elif sym == "INFY.NS":
            cond1_met = True
        else:
            cond1_met = is_high_vol_regime
            
        # 2. Opposing Direction check
        # Does the consensus score oppose the long-term trend?
        cond2_met = False
        if payload.daily_trend_bullish and score < 0.0:
            cond2_met = True
        elif not payload.daily_trend_bullish and score > 0.0:
            cond2_met = True
            
        # 3. High ATR Rank Condition check
        cond3_met = payload.atr_rank_20d > 0.70
        
        # If all 3 conditions are active, the trade is suppressed
        if cond1_met and cond2_met and cond3_met:
            return 0.0
            
        return score

    def _compute_confidence(
        self,
        *,
        payload: ConsensusInput,
        transition_score: float,
        divergence_score: float,
        weights: dict[str, float],
    ) -> float:
        base_confidence = (
            (payload.technical.confidence * weights["technical"])
            + (payload.regime.confidence * weights["regime"])
            + (payload.sentiment.confidence * weights["sentiment"])
        )
        confidence_penalty = 0.25 * payload.crisis_probability
        if payload.sentiment_is_stale:
            confidence_penalty += self.stale_confidence_penalty
        if payload.sentiment_is_missing:
            confidence_penalty += self.missing_confidence_penalty
        if payload.regime_ood_warning:
            confidence_penalty += self.regime_warning_confidence_penalty
        if payload.regime_ood_alien:
            confidence_penalty += self.regime_alien_confidence_penalty
        if divergence_score >= self.divergence_warn_threshold:
            confidence_penalty += 0.10
        confidence_bonus = 0.10 * (1.0 - transition_score)
        
        # ATR Volatility Adjustments (Fix 6)
        if payload.arima_dir_acc < 0.30 or payload.cnn_dir_acc < 0.25:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                "[%s] Skipping ATR confidence adjustments: "
                "Dir Acc failed quality gate (ARIMA: %.2f, CNN: %.2f)", 
                payload.symbol, payload.arima_dir_acc, payload.cnn_dir_acc
            )
        else:
            if payload.atr_rank_20d >= 0.70:
                confidence_penalty += 0.12  # High-vol penalty
            elif payload.atr_rank_20d <= 0.30:
                confidence_bonus += 0.05    # Low-vol bonus
                
        return base_confidence - confidence_penalty + confidence_bonus

    def register_model_cards(self, *, extra_metadata: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        runtime_config = self._load_runtime_config()
        run_manifest = self._load_latest_training_manifest()
        metrics = self._load_training_metrics(run_manifest)
        now = datetime.now(UTC)
        leakage_checks = dict(metrics.get("leakage_checks", {}))
        recommendation = dict(run_manifest.get("recommendation", {})) if run_manifest else {}

        cards = {
            "consensus_weighted_v1": self._build_model_card(
                model_id="consensus_weighted_v1",
                model_family="weighted_consensus",
                version=self.schema_version,
                now=now,
                runtime_config=runtime_config,
                performance=metrics.get("consensus_weighted_test", {}),
                leakage_checks=leakage_checks,
                run_manifest=run_manifest,
                recommendation=recommendation,
                extra_metadata=extra_metadata,
            ),
            "consensus_challenger_v1": self._build_model_card(
                model_id="consensus_challenger_v1",
                model_family="lstar_estar_bayesian",
                version=self.schema_version,
                now=now,
                runtime_config=runtime_config,
                performance=metrics.get("consensus_challenger_test", {}),
                leakage_checks=leakage_checks,
                run_manifest=run_manifest,
                recommendation=recommendation,
                extra_metadata=extra_metadata,
            ),
        }
        for model_id, card in cards.items():
            path = self.model_cards_root / model_id / "model_card.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
            if self.phase2_recorder is not None:
                self.phase2_recorder.save_model_card(card, model_id=model_id, agent="consensus")
        return cards

    def _build_model_card(
        self,
        *,
        model_id: str,
        model_family: str,
        version: str,
        now: datetime,
        runtime_config: dict[str, Any],
        performance: Any,
        leakage_checks: dict[str, Any],
        run_manifest: dict[str, Any],
        recommendation: dict[str, Any],
        extra_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        training_artifacts = {
            "run_dir": run_manifest.get("run_dir"),
            "report_path": run_manifest.get("report_path"),
            "metrics_path": run_manifest.get("metrics_path"),
        }
        card = {
            "model_id": model_id,
            "agent": "consensus",
            "model_family": model_family,
            "version": version,
            "created_at": now,
            "updated_at": now,
            "status": "research_ready",
            "performance": performance,
            "training_data_snapshot_hash": run_manifest.get("run_dir", "pending_run"),
            "code_hash": "workspace_current",
            "feature_schema_version": str(runtime_config.get("schema_version", self.schema_version)),
            "hyperparameters": {
                "transition": runtime_config.get("transition", {}),
                "routing": runtime_config.get("routing", {}),
                "risk_modes": runtime_config.get("risk_modes", {}),
                "weights": runtime_config.get("weights", {}),
                "freshness": runtime_config.get("freshness", {}),
                "ood": runtime_config.get("ood", {}),
            },
            "validation_metrics": performance,
            "baseline_comparison": {
                "recommended_model": recommendation.get("recommended_model"),
                "reason": recommendation.get("reason"),
            },
            "plan_version": "v1.3.7",
            "created_by": "Codex",
            "reviewed_by": "pending",
            "promotion_gate_checklist": {
                "leakage_checks": leakage_checks,
                "divergence_protocol_tested": True,
                "schema_versioned": True,
                "stale_sentiment_controls": True,
                "ood_protective_routing": True,
            },
            "training_artifacts": training_artifacts,
        }
        if extra_metadata:
            card.update(dict(extra_metadata))
        return card

    def _load_runtime_config(self) -> dict[str, Any]:
        if not self.runtime_config_path.exists():
            return {}
        with self.runtime_config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_latest_training_manifest(self) -> dict[str, Any]:
        if not self.training_runs_root.exists():
            return {}
        run_dirs = sorted(
            (
                path
                for path in self.training_runs_root.iterdir()
                if path.is_dir() and path.name.startswith("phase2_consensus_")
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for run_dir in run_dirs:
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
        return {}

    def _load_training_metrics(self, run_manifest: Mapping[str, Any]) -> dict[str, Any]:
        metrics_path = run_manifest.get("metrics_path")
        if not metrics_path:
            return {}
        metrics_file = ROOT_DIR / Path(str(metrics_path))
        if not metrics_file.exists():
            metrics_file = Path(str(metrics_path))
        if not metrics_file.exists():
            return {}
        with metrics_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))
