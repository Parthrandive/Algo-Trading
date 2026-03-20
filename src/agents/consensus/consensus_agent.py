"""
Consensus Agent — Combines Technical, Regime, and Sentiment signals.

Reads the latest predictions from each sub-agent and produces a weighted
consensus signal (LONG / SHORT / NEUTRAL) per symbol.

Usage:
    from src.agents.consensus.consensus_agent import ConsensusAgent

    agent = ConsensusAgent(database_url="postgresql://...")
    signal = agent.generate_signal("RELIANCE.NS")
    # signal.final_direction = "LONG" | "SHORT" | "NEUTRAL"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from src.db.phase2_recorder import Phase2Recorder

logger = logging.getLogger(__name__)


# ── Enums & Schemas ──────────────────────────────────────────────────────────


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TransitionModel(str, Enum):
    """How the consensus was computed."""
    WEIGHTED_AVG = "wtd_avg"           # standard weighted average
    CRISIS_OVERRIDE = "crisis"         # regime override in crisis
    SENTIMENT_GATE = "sent_gate"       # sentiment blocked a trade
    INSUFFICIENT_DATA = "no_data"      # not enough sub-agent signals


@dataclass
class AgentSignal:
    """Normalised sub-agent signal: direction + confidence."""
    direction: Direction
    confidence: float           # 0.0 – 1.0
    model_id: str
    available: bool = True


@dataclass
class ConsensusSignal:
    """The final consensus output for one symbol at one point in time."""
    symbol: str
    timestamp: datetime
    final_direction: Direction
    final_confidence: float
    technical_weight: float
    regime_weight: float
    sentiment_weight: float
    crisis_mode: bool
    agent_divergence: bool
    transition_model: TransitionModel
    model_id: str = "consensus_v1"
    schema_version: str = "1.0"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "final_direction": self.final_direction.value,
            "final_confidence": self.final_confidence,
            "technical_weight": self.technical_weight,
            "regime_weight": self.regime_weight,
            "sentiment_weight": self.sentiment_weight,
            "crisis_mode": self.crisis_mode,
            "agent_divergence": self.agent_divergence,
            "transition_model": self.transition_model.value,
            "model_id": self.model_id,
            "schema_version": self.schema_version,
        }


# ── Default Weights ──────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "technical": 0.50,
    "regime": 0.25,
    "sentiment": 0.25,
}

CRISIS_WEIGHTS = {
    "technical": 0.20,
    "regime": 0.60,
    "sentiment": 0.20,
}

# Confidence thresholds
MIN_CONFIDENCE_FOR_TRADE = 0.55
CRISIS_REGIME_STATES = {"CRISIS", "HIGH_VOLATILITY", "CRASH"}
DIVERGENCE_THRESHOLD = 1.5  # if agents disagree by more than this score gap


# ── Consensus Agent ──────────────────────────────────────────────────────────


class ConsensusAgent:
    """
    Reads latest predictions from Technical, Regime, and Sentiment agents.
    Produces a weighted consensus signal per symbol.

    Architecture:
        technical_predictions ──┐
        regime_predictions   ──── ConsensusAgent ──→ consensus_signals
        sentiment_scores     ──┘

    Consensus rules:
        1. Normal mode: weighted average (50/25/25)
        2. Crisis mode: regime weight jumps to 60% (20/60/20)
        3. Sentiment gate: if strong neg sentiment (< -0.5), block LONG signals
        4. Divergence flag: if agents disagree strongly, reduce confidence
    """

    def __init__(
        self,
        database_url: str = "postgresql://sentinel:sentinel@localhost:5432/sentinel_db",
        weights: dict[str, float] | None = None,
    ):
        self.engine = create_engine(database_url)
        self.recorder = Phase2Recorder(engine=self.engine)
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.model_id = "consensus_v1"

    # ── Public API ───────────────────────────────────────────────────────

    def generate_signal(
        self,
        symbol: str,
        *,
        lookback_hours: int = 24,
        save_to_db: bool = True,
    ) -> ConsensusSignal:
        """Generate a consensus signal for the given symbol."""
        now = datetime.now(timezone.utc)

        # 1. Fetch latest sub-agent signals
        tech = self._get_technical_signal(symbol, lookback_hours)
        regime = self._get_regime_signal(symbol, lookback_hours)
        sentiment = self._get_sentiment_signal(symbol, lookback_hours)

        # 2. Determine if crisis mode
        crisis_mode = self._is_crisis(regime)
        weights = CRISIS_WEIGHTS.copy() if crisis_mode else self.weights.copy()

        # 3. Handle missing agents
        available_agents = [a for a in [tech, regime, sentiment] if a.available]
        if len(available_agents) == 0:
            signal = ConsensusSignal(
                symbol=symbol, timestamp=now,
                final_direction=Direction.NEUTRAL,
                final_confidence=0.0,
                technical_weight=0.0, regime_weight=0.0, sentiment_weight=0.0,
                crisis_mode=False, agent_divergence=False,
                transition_model=TransitionModel.INSUFFICIENT_DATA,
                details={"reason": "no_agent_signals_available"},
            )
            if save_to_db:
                self._save(signal)
            return signal

        # Redistribute weights if agents are missing
        if not tech.available:
            weights = self._redistribute(weights, "technical")
        if not regime.available:
            weights = self._redistribute(weights, "regime")
        if not sentiment.available:
            weights = self._redistribute(weights, "sentiment")

        # 4. Compute weighted score (-1 to +1)
        score = (
            self._direction_score(tech) * weights["technical"]
            + self._direction_score(regime) * weights["regime"]
            + self._direction_score(sentiment) * weights["sentiment"]
        )

        # 5. Apply sentiment gate
        transition = TransitionModel.CRISIS_OVERRIDE if crisis_mode else TransitionModel.WEIGHTED_AVG
        if sentiment.available and sentiment.confidence > 0.7:
            sent_score = self._direction_score(sentiment)
            if sent_score < -0.5 and score > 0:
                # Strong negative sentiment blocks LONG
                score = min(score, 0.0)
                transition = TransitionModel.SENTIMENT_GATE
                logger.info(f"{symbol}: sentiment gate blocked LONG signal")

        # 6. Convert score to direction
        if score > 0.1:
            direction = Direction.LONG
        elif score < -0.1:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        # 7. Compute confidence and divergence
        confidence = min(abs(score), 1.0)
        divergence = self._check_divergence(tech, regime, sentiment)

        if divergence:
            confidence *= 0.7  # reduce confidence on divergence

        signal = ConsensusSignal(
            symbol=symbol,
            timestamp=now,
            final_direction=direction,
            final_confidence=round(confidence, 4),
            technical_weight=round(weights["technical"], 2),
            regime_weight=round(weights["regime"], 2),
            sentiment_weight=round(weights["sentiment"], 2),
            crisis_mode=crisis_mode,
            agent_divergence=divergence,
            transition_model=transition,
            details={
                "technical": {"dir": tech.direction.value, "conf": tech.confidence, "avail": tech.available},
                "regime": {"dir": regime.direction.value, "conf": regime.confidence, "avail": regime.available},
                "sentiment": {"dir": sentiment.direction.value, "conf": sentiment.confidence, "avail": sentiment.available},
                "raw_score": round(score, 4),
            },
        )

        if save_to_db:
            self._save(signal)

        return signal

    def generate_signals_batch(
        self, symbols: list[str], *, save_to_db: bool = True
    ) -> list[ConsensusSignal]:
        """Generate consensus for multiple symbols."""
        return [self.generate_signal(s, save_to_db=save_to_db) for s in symbols]

    # ── Sub-agent signal readers ─────────────────────────────────────────

    def _get_technical_signal(self, symbol: str, lookback_hours: int) -> AgentSignal:
        try:
            df = pd.read_sql(text("""
                SELECT direction, confidence, model_id FROM technical_predictions
                WHERE symbol = :sym
                AND timestamp > NOW() - INTERVAL ':hours hours'
                ORDER BY timestamp DESC LIMIT 1
            """.replace(":hours", str(lookback_hours))), self.engine, params={"sym": symbol})

            if df.empty:
                return AgentSignal(Direction.NEUTRAL, 0.0, "none", available=False)

            row = df.iloc[0]
            direction = self._parse_direction(row["direction"])
            return AgentSignal(direction, float(row["confidence"]), str(row["model_id"]))
        except Exception as e:
            logger.warning(f"Technical signal fetch failed for {symbol}: {e}")
            return AgentSignal(Direction.NEUTRAL, 0.0, "error", available=False)

    def _get_regime_signal(self, symbol: str, lookback_hours: int) -> AgentSignal:
        try:
            df = pd.read_sql(text("""
                SELECT regime_state, confidence, risk_level, model_id FROM regime_predictions
                WHERE symbol = :sym
                AND timestamp > NOW() - INTERVAL ':hours hours'
                ORDER BY timestamp DESC LIMIT 1
            """.replace(":hours", str(lookback_hours))), self.engine, params={"sym": symbol})

            if df.empty:
                return AgentSignal(Direction.NEUTRAL, 0.0, "none", available=False)

            row = df.iloc[0]
            state = str(row["regime_state"]).upper()
            risk = str(row["risk_level"]).upper()

            # Map regime to direction
            if state in CRISIS_REGIME_STATES or "HIGH" in risk:
                direction = Direction.SHORT
            elif state in {"BULLISH", "UPTREND", "TRENDING_UP"}:
                direction = Direction.LONG
            elif state in {"BEARISH", "DOWNTREND", "TRENDING_DOWN"}:
                direction = Direction.SHORT
            else:
                direction = Direction.NEUTRAL

            return AgentSignal(direction, float(row["confidence"]), str(row["model_id"]))
        except Exception as e:
            logger.warning(f"Regime signal fetch failed for {symbol}: {e}")
            return AgentSignal(Direction.NEUTRAL, 0.0, "error", available=False)

    def _get_sentiment_signal(self, symbol: str, lookback_hours: int) -> AgentSignal:
        try:
            # Try symbol-specific first, fall back to MARKET
            df = pd.read_sql(text("""
                SELECT sentiment_class, sentiment_score, confidence, model_id
                FROM sentiment_scores
                WHERE (symbol = :sym OR symbol = 'MARKET')
                AND timestamp > NOW() - INTERVAL ':hours hours'
                ORDER BY
                    CASE WHEN symbol = :sym THEN 0 ELSE 1 END,
                    timestamp DESC
                LIMIT 1
            """.replace(":hours", str(lookback_hours))), self.engine, params={"sym": symbol})

            if df.empty:
                return AgentSignal(Direction.NEUTRAL, 0.0, "none", available=False)

            row = df.iloc[0]
            score = float(row["sentiment_score"])
            if score > 0.3:
                direction = Direction.LONG
            elif score < -0.3:
                direction = Direction.SHORT
            else:
                direction = Direction.NEUTRAL

            return AgentSignal(direction, float(row["confidence"]), str(row["model_id"]))
        except Exception as e:
            logger.warning(f"Sentiment signal fetch failed for {symbol}: {e}")
            return AgentSignal(Direction.NEUTRAL, 0.0, "error", available=False)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_direction(raw: str) -> Direction:
        raw_upper = str(raw).upper()
        if raw_upper in {"LONG", "UP", "BULLISH", "BUY"}:
            return Direction.LONG
        if raw_upper in {"SHORT", "DOWN", "BEARISH", "SELL"}:
            return Direction.SHORT
        return Direction.NEUTRAL

    @staticmethod
    def _direction_score(signal: AgentSignal) -> float:
        if not signal.available:
            return 0.0
        multiplier = {Direction.LONG: 1.0, Direction.SHORT: -1.0, Direction.NEUTRAL: 0.0}
        return multiplier[signal.direction] * signal.confidence

    @staticmethod
    def _is_crisis(regime: AgentSignal) -> bool:
        return regime.available and regime.direction == Direction.SHORT and regime.confidence > 0.7

    @staticmethod
    def _redistribute(weights: dict[str, float], remove_key: str) -> dict[str, float]:
        removed_weight = weights.pop(remove_key, 0.0)
        remaining = sum(weights.values())
        if remaining > 0:
            for k in weights:
                weights[k] += removed_weight * (weights[k] / remaining)
        weights[remove_key] = 0.0
        return weights

    @staticmethod
    def _check_divergence(tech: AgentSignal, regime: AgentSignal, sentiment: AgentSignal) -> bool:
        signals = [s for s in [tech, regime, sentiment] if s.available]
        if len(signals) < 2:
            return False
        scores = [ConsensusAgent._direction_score(s) for s in signals]
        return (max(scores) - min(scores)) > DIVERGENCE_THRESHOLD

    def _save(self, signal: ConsensusSignal) -> None:
        try:
            self.recorder.save_consensus_signal(signal.to_dict())
            logger.info(f"Consensus saved: {signal.symbol} -> {signal.final_direction.value} "
                        f"(conf={signal.final_confidence})")
        except Exception as e:
            logger.error(f"Failed to save consensus for {signal.symbol}: {e}")
