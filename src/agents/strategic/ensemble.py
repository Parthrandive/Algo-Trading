from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from src.agents.strategic.reward import max_drawdown, sharpe_ratio
from src.agents.strategic.schemas import (
    ActionType,
    LoopType,
    PolicyType,
    StrategicToExecutiveContract,
)


@dataclass(frozen=True)
class PolicySignal:
    policy_id: str
    action: float
    confidence: float
    diversity_hint: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if not -1.0 <= float(self.action) <= 1.0:
            raise ValueError("action must be between -1.0 and 1.0")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ThresholdTable:
    hold_threshold: float = 0.08
    buy_threshold: float = 0.22
    sell_threshold: float = 0.22
    reduce_threshold: float = 0.45
    close_threshold: float = 0.70

    def __post_init__(self) -> None:
        for field_name in (
            "hold_threshold",
            "buy_threshold",
            "sell_threshold",
            "reduce_threshold",
            "close_threshold",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0.0, 1.0]")
        if self.reduce_threshold < max(self.buy_threshold, self.sell_threshold):
            raise ValueError("reduce_threshold must be >= buy_threshold and sell_threshold")
        if self.close_threshold < self.reduce_threshold:
            raise ValueError("close_threshold must be >= reduce_threshold")

    def to_action(
        self,
        score: float,
        *,
        crisis_mode: bool = False,
        agent_divergence: bool = False,
    ) -> tuple[ActionType, float, str]:
        normalized = float(np.clip(score, -1.0, 1.0))
        magnitude = abs(normalized)

        if crisis_mode:
            if magnitude >= self.close_threshold:
                return ActionType.CLOSE, min(1.0, magnitude), "crisis close threshold"
            if magnitude >= self.reduce_threshold:
                return ActionType.REDUCE, min(0.5, magnitude), "crisis reduce threshold"

        if agent_divergence and magnitude >= self.reduce_threshold:
            return ActionType.REDUCE, min(0.5, magnitude), "agent divergence protective reduce"

        if normalized >= self.buy_threshold:
            return ActionType.BUY, min(1.0, magnitude), "buy threshold"
        if normalized <= -self.sell_threshold:
            return ActionType.SELL, min(1.0, magnitude), "sell threshold"
        if magnitude <= self.hold_threshold:
            return ActionType.HOLD, 0.0, "hold band"
        return ActionType.REDUCE, min(0.25, magnitude), "between hold and directional threshold"


@dataclass(frozen=True)
class MaxEntropyConfig:
    temperature: float = 0.8
    diversity_weight: float = 0.35
    min_weight_floor: float = 0.0

    def __post_init__(self) -> None:
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        if float(self.diversity_weight) < 0.0:
            raise ValueError("diversity_weight must be >= 0")
        if not 0.0 <= float(self.min_weight_floor) <= 0.33:
            raise ValueError("min_weight_floor must be within [0.0, 0.33]")


@dataclass(frozen=True)
class EnsembleDecision:
    combined_score: float
    confidence: float
    entropy: float
    weights: Mapping[str, float]
    action: ActionType
    action_size: float
    reason: str
    temperature: float
    threshold_table: ThresholdTable

    def to_contract(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        snapshot_id: str,
        observation_id: int | None = None,
        policy_id: str = "phase3_max_entropy_ensemble_v1",
        loop_type: LoopType = LoopType.SLOW,
        policy_type: PolicyType = PolicyType.TEACHER,
    ) -> StrategicToExecutiveContract:
        risk_override = None
        if self.action == ActionType.REDUCE:
            risk_override = "reduce_only"
        elif self.action == ActionType.CLOSE:
            risk_override = "close_only"

        return StrategicToExecutiveContract(
            timestamp=timestamp,
            symbol=symbol,
            policy_id=policy_id,
            policy_type=policy_type,
            loop_type=loop_type,
            action=self.action,
            action_size=float(np.clip(self.action_size, 0.0, 1.0)),
            confidence=float(np.clip(self.confidence, 0.0, 1.0)),
            observation_id=observation_id,
            snapshot_id=snapshot_id,
            risk_override=risk_override,
            decision_reason=self.reason,
            metadata={
                "entropy": float(self.entropy),
                "combined_score": float(self.combined_score),
                "temperature": float(self.temperature),
                "weights": dict(self.weights),
                "threshold_table": {
                    "hold": self.threshold_table.hold_threshold,
                    "buy": self.threshold_table.buy_threshold,
                    "sell": self.threshold_table.sell_threshold,
                    "reduce": self.threshold_table.reduce_threshold,
                    "close": self.threshold_table.close_threshold,
                },
            },
        )


class MaxEntropyEnsemble:
    """
    Week 2 Day 1 max-entropy action combiner.

    Uses confidence-weighted softmax with diversity adjustment so each teacher
    contributes according to both confidence and behavioral difference.
    """

    def __init__(self, config: MaxEntropyConfig | None = None):
        self.config = config or MaxEntropyConfig()

    def combine(
        self,
        signals: Sequence[PolicySignal],
        *,
        threshold_table: ThresholdTable | None = None,
        crisis_mode: bool = False,
        agent_divergence: bool = False,
    ) -> EnsembleDecision:
        if not signals:
            raise ValueError("signals must be non-empty")

        table = threshold_table or ThresholdTable()
        actions = np.asarray([float(item.action) for item in signals], dtype=float)
        confidences = np.asarray([float(item.confidence) for item in signals], dtype=float)

        diversity = np.zeros(shape=(len(signals),), dtype=float)
        for idx, signal in enumerate(signals):
            distances = np.abs(actions[idx] - actions)
            diversity[idx] = float(np.mean(distances))
            diversity[idx] += max(0.0, float(signal.diversity_hint))

        raw_score = np.clip(confidences * (1.0 + self.config.diversity_weight * diversity), 1e-9, None)
        weights = _softmax(np.log(raw_score), temperature=self.config.temperature)
        weights = _apply_min_weight_floor(weights, self.config.min_weight_floor)

        combined_score = float(np.clip(np.dot(weights, actions), -1.0, 1.0))
        combined_confidence = float(np.clip(np.dot(weights, confidences), 0.0, 1.0))
        entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0))))

        action, action_size, reason = table.to_action(
            combined_score,
            crisis_mode=crisis_mode,
            agent_divergence=agent_divergence,
        )

        return EnsembleDecision(
            combined_score=combined_score,
            confidence=combined_confidence,
            entropy=entropy,
            weights={signal.policy_id: float(weights[idx]) for idx, signal in enumerate(signals)},
            action=action,
            action_size=float(action_size),
            reason=reason,
            temperature=self.config.temperature,
            threshold_table=table,
        )

    def combine_contracts(
        self,
        contracts: Sequence[StrategicToExecutiveContract],
        *,
        threshold_table: ThresholdTable | None = None,
        crisis_mode: bool = False,
        agent_divergence: bool = False,
    ) -> EnsembleDecision:
        signals = [
            PolicySignal(
                policy_id=contract.policy_id,
                action=_contract_to_scalar(contract),
                confidence=float(contract.confidence),
                metadata={"action": contract.action.value, "loop_type": contract.loop_type.value},
            )
            for contract in contracts
        ]
        return self.combine(
            signals,
            threshold_table=threshold_table,
            crisis_mode=crisis_mode,
            agent_divergence=agent_divergence,
        )


@dataclass(frozen=True)
class GAThresholdConfig:
    population_size: int = 36
    generations: int = 28
    elite_count: int = 6
    mutation_rate: float = 0.20
    crossover_rate: float = 0.70
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size < 8:
            raise ValueError("population_size must be >= 8")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 1 and population_size - 1")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be within [0.0, 1.0]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be within [0.0, 1.0]")


@dataclass(frozen=True)
class GAOptimizationResult:
    thresholds: ThresholdTable
    best_fitness: float
    history: tuple[float, ...]


class OfflineGeneticThresholdOptimizer:
    """
    Week 2 Day 1 GA threshold search.
    Hard rule: optimization is offline-only and cannot be used on Fast Loop path.
    """

    def __init__(self, config: GAThresholdConfig | None = None):
        self.config = config or GAThresholdConfig()

    def optimize(
        self,
        *,
        action_scores: Sequence[float],
        realized_returns: Sequence[float],
        crisis_flags: Sequence[bool] | None = None,
        runtime_mode: str = "offline",
    ) -> GAOptimizationResult:
        if runtime_mode.strip().lower() != "offline":
            raise ValueError("genetic threshold optimization is offline-only")

        scores = np.asarray(list(action_scores), dtype=float)
        returns = np.asarray(list(realized_returns), dtype=float)
        if scores.shape[0] != returns.shape[0]:
            raise ValueError("action_scores and realized_returns must have matching lengths")
        if scores.size == 0:
            return GAOptimizationResult(thresholds=ThresholdTable(), best_fitness=0.0, history=tuple())

        if crisis_flags is None:
            crisis = np.zeros(shape=scores.shape, dtype=bool)
        else:
            crisis = np.asarray(list(crisis_flags), dtype=bool)
            if crisis.shape[0] != scores.shape[0]:
                raise ValueError("crisis_flags must have matching length")

        rng = random.Random(self.config.seed)
        population = [self._random_thresholds(rng) for _ in range(self.config.population_size)]
        history: list[float] = []

        for _ in range(self.config.generations):
            ranked = sorted(
                ((self._fitness(candidate, scores, returns, crisis), candidate) for candidate in population),
                key=lambda item: item[0],
                reverse=True,
            )
            history.append(float(ranked[0][0]))
            elites = [candidate for _, candidate in ranked[: self.config.elite_count]]

            next_population = list(elites)
            while len(next_population) < self.config.population_size:
                parent_a = rng.choice(elites)
                parent_b = rng.choice(ranked[: max(self.config.elite_count * 2, 2)])[1]
                if rng.random() <= self.config.crossover_rate:
                    child = self._crossover(parent_a, parent_b, rng)
                else:
                    child = parent_a
                if rng.random() <= self.config.mutation_rate:
                    child = self._mutate(child, rng)
                next_population.append(child)
            population = next_population

        final_ranked = sorted(
            ((self._fitness(candidate, scores, returns, crisis), candidate) for candidate in population),
            key=lambda item: item[0],
            reverse=True,
        )
        best_fitness, best_thresholds = final_ranked[0]
        return GAOptimizationResult(
            thresholds=best_thresholds,
            best_fitness=float(best_fitness),
            history=tuple(history),
        )

    def _fitness(
        self,
        thresholds: ThresholdTable,
        scores: np.ndarray,
        returns: np.ndarray,
        crisis: np.ndarray,
    ) -> float:
        pnl_steps: list[float] = []
        for idx in range(scores.shape[0]):
            action, size, _ = thresholds.to_action(float(scores[idx]), crisis_mode=bool(crisis[idx]))
            exposure = _action_to_exposure(action, size, float(scores[idx]))
            pnl_steps.append(exposure * float(returns[idx]))

        sharpe = sharpe_ratio(pnl_steps)
        drawdown = max_drawdown(pnl_steps)
        avg_return = float(np.mean(pnl_steps)) if pnl_steps else 0.0

        crisis_penalty = 0.0
        if np.any(crisis):
            crisis_pnl = [pnl_steps[idx] for idx in range(len(pnl_steps)) if bool(crisis[idx])]
            if crisis_pnl:
                crisis_penalty = abs(min(0.0, float(np.mean(crisis_pnl)))) * 2.5

        return avg_return + (0.08 * sharpe) - (0.6 * drawdown) - crisis_penalty

    @staticmethod
    def _random_thresholds(rng: random.Random) -> ThresholdTable:
        hold = rng.uniform(0.03, 0.14)
        buy = rng.uniform(0.15, 0.40)
        sell = rng.uniform(0.15, 0.40)
        reduce = rng.uniform(max(buy, sell), 0.65)
        close = rng.uniform(max(0.68, reduce), 0.95)
        return ThresholdTable(
            hold_threshold=hold,
            buy_threshold=buy,
            sell_threshold=sell,
            reduce_threshold=reduce,
            close_threshold=close,
        )

    @staticmethod
    def _crossover(a: ThresholdTable, b: ThresholdTable, rng: random.Random) -> ThresholdTable:
        alpha = rng.uniform(0.20, 0.80)
        merged = ThresholdTable(
            hold_threshold=(alpha * a.hold_threshold) + ((1.0 - alpha) * b.hold_threshold),
            buy_threshold=(alpha * a.buy_threshold) + ((1.0 - alpha) * b.buy_threshold),
            sell_threshold=(alpha * a.sell_threshold) + ((1.0 - alpha) * b.sell_threshold),
            reduce_threshold=(alpha * a.reduce_threshold) + ((1.0 - alpha) * b.reduce_threshold),
            close_threshold=(alpha * a.close_threshold) + ((1.0 - alpha) * b.close_threshold),
        )
        return _repair_threshold_table(merged)

    @staticmethod
    def _mutate(candidate: ThresholdTable, rng: random.Random) -> ThresholdTable:
        jitter = lambda span: rng.uniform(-span, span)
        mutated = ThresholdTable(
            hold_threshold=float(np.clip(candidate.hold_threshold + jitter(0.03), 0.01, 0.20)),
            buy_threshold=float(np.clip(candidate.buy_threshold + jitter(0.05), 0.08, 0.60)),
            sell_threshold=float(np.clip(candidate.sell_threshold + jitter(0.05), 0.08, 0.60)),
            reduce_threshold=float(np.clip(candidate.reduce_threshold + jitter(0.06), 0.12, 0.80)),
            close_threshold=float(np.clip(candidate.close_threshold + jitter(0.06), 0.20, 0.95)),
        )
        return _repair_threshold_table(mutated)


def normalize_policy_weights(weights: Mapping[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        if not cleaned:
            return {}
        uniform = 1.0 / float(len(cleaned))
        return {key: uniform for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = values / float(max(temperature, 1e-6))
    shifted = scaled - float(np.max(scaled))
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if total <= 0.0 or not math.isfinite(total):
        return np.full(shape=values.shape, fill_value=1.0 / max(values.shape[0], 1), dtype=float)
    return exps / total


def _apply_min_weight_floor(weights: np.ndarray, floor: float) -> np.ndarray:
    if floor <= 0.0 or weights.size == 0:
        return weights
    max_floor = 1.0 / float(weights.size)
    floor = min(floor, max_floor)
    if floor <= 0.0:
        return weights
    scaled = weights * (1.0 - floor * weights.size)
    adjusted = scaled + floor
    total = float(np.sum(adjusted))
    if total <= 0.0:
        return np.full(shape=weights.shape, fill_value=1.0 / float(weights.size), dtype=float)
    return adjusted / total


def _contract_to_scalar(contract: StrategicToExecutiveContract) -> float:
    if contract.action == ActionType.BUY:
        return float(np.clip(contract.action_size, 0.0, 1.0))
    if contract.action == ActionType.SELL:
        return -float(np.clip(contract.action_size, 0.0, 1.0))
    if contract.action in {ActionType.CLOSE, ActionType.REDUCE}:
        # Risk-protective actions intentionally map toward neutral sizing.
        return 0.0
    return 0.0


def _action_to_exposure(action: ActionType, size: float, score_hint: float) -> float:
    magnitude = float(np.clip(size, 0.0, 1.0))
    if action == ActionType.BUY:
        return magnitude
    if action == ActionType.SELL:
        return -magnitude
    if action == ActionType.REDUCE:
        return 0.25 * float(np.sign(score_hint))
    if action == ActionType.CLOSE:
        return 0.0
    return 0.0


def _repair_threshold_table(table: ThresholdTable) -> ThresholdTable:
    hold = float(np.clip(table.hold_threshold, 0.01, 0.20))
    buy = float(np.clip(table.buy_threshold, hold + 0.01, 0.70))
    sell = float(np.clip(table.sell_threshold, hold + 0.01, 0.70))
    reduce = float(np.clip(table.reduce_threshold, max(buy, sell), 0.85))
    close = float(np.clip(table.close_threshold, max(reduce, 0.25), 0.95))
    return ThresholdTable(
        hold_threshold=hold,
        buy_threshold=buy,
        sell_threshold=sell,
        reduce_threshold=reduce,
        close_threshold=close,
    )
