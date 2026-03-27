from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.agents.risk_overseer.schemas import RiskAssessment
from src.agents.strategic.config import EnsembleConfig
from src.agents.strategic.schemas import (
    ActionType,
    EnsembleDecision,
    PolicyAction,
    PolicyWeight,
    RiskMode,
    StrategicObservation,
    ThresholdCandidate,
)

_ACTION_TARGETS = {
    ActionType.BUY: 1.0,
    ActionType.SELL: -1.0,
    ActionType.HOLD: 0.0,
    ActionType.CLOSE: -0.5,
    ActionType.REDUCE: -0.25,
}


@dataclass(frozen=True)
class PolicySignal:
    policy_id: str
    action: ActionType
    action_size: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdTable:
    buy: float = 0.35
    sell: float = -0.35
    reduce: float = 0.15
    hold: float = 0.10
    version: str = "1.0"


@dataclass(frozen=True)
class GAThresholdConfig:
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_count: int = 2


@dataclass(frozen=True)
class GAOptimizationResult:
    best_thresholds: ThresholdTable
    best_fitness: float
    history: list[float]
    optimized_at: datetime


class MaxEntropyEnsemble:
    """
    Advanced Phase 3 Maximum Entropy Ensemble with risk-overseer overlays.

    Supports both:
    1. `decide(symbol, observations, policy_actions, ...)` from the newer Phase 3 path.
    2. `decide(observation, teacher_actions, ...)` from the earlier Week 1/2 path.
    """

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()

    def decide(self, *args, **kwargs) -> EnsembleDecision:
        risk_assessment: RiskAssessment | None = kwargs.pop("risk_assessment", None)
        thresholds = kwargs.pop("thresholds", None)
        threshold = kwargs.pop("threshold", None)
        risk_mode = kwargs.pop("risk_mode", RiskMode.NORMAL)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        selected_thresholds = thresholds or threshold

        if len(args) >= 2 and isinstance(args[0], StrategicObservation):
            observation = args[0]
            policy_actions = tuple(args[1])
            symbol = observation.symbol
            observations = (observation,)
        elif len(args) >= 3:
            symbol = str(args[0])
            observations = tuple(args[1])
            policy_actions = tuple(args[2])
            observation = observations[0] if observations else None
        else:
            raise TypeError("decide() expects either (observation, policy_actions) or (symbol, observations, policy_actions)")

        if not policy_actions:
            raise ValueError("policy_actions must not be empty")

        now = datetime.now(UTC)
        snapshot_id = observation.snapshot_id if observation is not None else "missing"
        confidences = np.asarray([max(self.config.confidence_floor, float(a.confidence)) for a in policy_actions], dtype=float)
        diversities = self._calculate_diversity_scores(policy_actions)
        scores = confidences * (1.0 + diversities * self.config.diversity_penalty)

        if observation is not None:
            for i, action in enumerate(policy_actions):
                if observation.consensus_direction == "BUY" and action.action == ActionType.BUY:
                    scores[i] += self.config.consensus_boost
                elif observation.consensus_direction == "SELL" and action.action == ActionType.SELL:
                    scores[i] += self.config.consensus_boost
                if observation.crisis_mode:
                    scores[i] *= (1.0 - self.config.crisis_confidence_haircut)
                if observation.agent_divergence:
                    scores[i] = min(scores[i], self.config.divergence_hold_confidence_cap)

        weights = self._softmax(scores / max(self.config.temperature, 1e-6))
        weights = self._apply_min_weight_floor(weights, self.config.minimum_policy_weight)
        if risk_assessment is not None:
            weights = self._apply_risk_assessment(weights, risk_assessment)

        action_vals = np.asarray([_ACTION_TARGETS[a.action] * a.action_size for a in policy_actions], dtype=float)
        aggregate_val = float(np.sum(action_vals * weights))
        final_action = self._threshold_to_action(aggregate_val, selected_thresholds)

        resolved_risk_mode = risk_mode
        if observation is not None:
            if observation.crisis_mode:
                resolved_risk_mode = RiskMode.CLOSE_ONLY
                if final_action == ActionType.BUY:
                    final_action = ActionType.HOLD
            elif observation.agent_divergence:
                resolved_risk_mode = RiskMode.REDUCE_ONLY
        if risk_assessment is not None:
            resolved_risk_mode = _max_risk_mode(resolved_risk_mode, risk_assessment.mode)

        dominant_idx = int(np.argmax(weights))
        dominant_policy_id = policy_actions[dominant_idx].policy_id
        action_size = abs(aggregate_val) if final_action != ActionType.HOLD else 0.0
        if observation is not None:
            action_size = self._final_action_size(final_action, action_size, observation, risk_assessment)
        elif risk_assessment is not None:
            action_size = min(action_size, risk_assessment.exposure_cap)

        policy_weight_schemas = tuple(
            PolicyWeight(
                policy_id=action.policy_id,
                weight=float(weights[i]),
                confidence=float(action.confidence),
                diversity_score=float(diversities[i]),
            )
            for i, action in enumerate(policy_actions)
        )

        return EnsembleDecision(
            timestamp=now,
            symbol=symbol,
            observation_snapshot_id=snapshot_id,
            action=final_action,
            action_size=action_size,
            confidence=float(np.mean(confidences)),
            mode="max_entropy_v1",
            dominant_policy_id=dominant_policy_id,
            policy_weights=policy_weight_schemas,
            threshold_candidate_id=getattr(selected_thresholds, "candidate_id", "default") if selected_thresholds else "default",
            rationale=f"ensemble_score={aggregate_val:.4f}; dominant={dominant_policy_id}",
            risk_mode=resolved_risk_mode,
            metadata={
                "temperature": self.config.temperature,
                "diversity_penalty": self.config.diversity_penalty,
                "weighted_aggregate": aggregate_val,
                "risk_overlay_exposure_cap": risk_assessment.exposure_cap if risk_assessment else 1.0,
                "risk_overlay_state": risk_assessment.crisis_state.value if risk_assessment else "none",
            },
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _apply_min_weight_floor(self, weights: np.ndarray, floor: float) -> np.ndarray:
        if floor <= 0.0:
            return weights
        weights = np.maximum(weights, floor)
        return weights / weights.sum()

    def _calculate_diversity_scores(self, actions: Sequence[PolicyAction]) -> np.ndarray:
        if len(actions) <= 1:
            return np.ones(len(actions))
        action_signs = np.asarray([_ACTION_TARGETS[a.action] for a in actions], dtype=float)
        diversities = np.zeros(len(actions))
        for i in range(len(actions)):
            distances = np.abs(action_signs[i] - action_signs)
            diversities[i] = float(np.mean(distances))
        return diversities

    def _threshold_to_action(self, score: float, thresholds: ThresholdTable | ThresholdCandidate | None) -> ActionType:
        if thresholds is None:
            thresholds = ThresholdTable()

        buy_t = getattr(thresholds, "buy", getattr(thresholds, "buy_threshold", 0.35))
        sell_t = getattr(thresholds, "sell", getattr(thresholds, "sell_threshold", -0.35))
        reduce_t = getattr(thresholds, "reduce", getattr(thresholds, "reduce_threshold", 0.15))

        if score >= buy_t:
            return ActionType.BUY
        if score <= sell_t:
            return ActionType.SELL
        if abs(score) >= reduce_t:
            return ActionType.REDUCE
        return ActionType.HOLD

    def _apply_risk_assessment(self, weights: np.ndarray, risk_assessment: RiskAssessment) -> np.ndarray:
        adjusted = np.array(weights, copy=True)
        if adjusted.size == 0:
            return adjusted
        if risk_assessment.crisis_state.value == "full_crisis":
            dominant_idx = int(np.argmax(adjusted))
            capped = min(float(adjusted[dominant_idx]), risk_assessment.crisis_weight_cap)
            remainder = max(0.0, 1.0 - capped)
            peer_total = float(np.sum(adjusted) - adjusted[dominant_idx])
            adjusted[dominant_idx] = capped
            if peer_total > 0.0:
                for idx in range(adjusted.size):
                    if idx == dominant_idx:
                        continue
                    adjusted[idx] = remainder * float(adjusted[idx] / peer_total)
        total = float(np.sum(adjusted))
        if total <= 0.0:
            return weights
        return adjusted / total

    def _final_action_size(
        self,
        action: ActionType,
        size: float,
        observation: StrategicObservation,
        risk_assessment: RiskAssessment | None = None,
    ) -> float:
        if action == ActionType.HOLD:
            return 0.0
        size = max(0.05, min(1.0, size))
        if observation.crisis_mode:
            size = min(size, 0.25)
        if observation.agent_divergence:
            size = min(size, 0.10)
        if action in {ActionType.REDUCE, ActionType.CLOSE}:
            size = min(size, 0.50)
        if risk_assessment is not None:
            size = min(size, risk_assessment.exposure_cap)
        return size


class OfflineGeneticThresholdOptimizer:
    """Optimizes ThresholdTable using GA against historical deliberation logs."""

    def __init__(self, config: GAThresholdConfig | None = None):
        self.config = config or GAThresholdConfig()

    def optimize(
        self,
        historical_data: Iterable[Mapping[str, Any]],
        fitness_fn: Callable[[ThresholdTable, Iterable[Mapping[str, Any]]], float],
    ) -> GAOptimizationResult:
        population = self._init_population()
        history: list[float] = []
        fitnesses: list[float] = []

        for _ in range(self.config.generations):
            fitnesses = [fitness_fn(ind, historical_data) for ind in population]
            history.append(float(np.max(fitnesses)))
            new_pop = self._select_elites(population, fitnesses)
            while len(new_pop) < self.config.population_size:
                p1, p2 = self._select_parents(population, fitnesses)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            population = new_pop

        best_idx = int(np.argmax(fitnesses))
        return GAOptimizationResult(
            best_thresholds=population[best_idx],
            best_fitness=fitnesses[best_idx],
            history=history,
            optimized_at=datetime.now(UTC),
        )

    def _init_population(self) -> list[ThresholdTable]:
        return [
            ThresholdTable(
                buy=random.uniform(0.2, 0.6),
                sell=random.uniform(-0.6, -0.2),
                reduce=random.uniform(0.1, 0.3),
                hold=random.uniform(0.05, 0.15),
            )
            for _ in range(self.config.population_size)
        ]

    def _select_elites(self, pop: list[ThresholdTable], fitness: list[float]) -> list[ThresholdTable]:
        indices = np.argsort(fitness)[-self.config.elite_count :]
        return [pop[i] for i in indices]

    def _select_parents(self, pop: list[ThresholdTable], fitness: list[float]) -> tuple[ThresholdTable, ThresholdTable]:
        def pick():
            pool = random.sample(list(range(len(pop))), 3)
            return pop[max(pool, key=lambda i: fitness[i])]

        return pick(), pick()

    def _crossover(self, p1: ThresholdTable, p2: ThresholdTable) -> ThresholdTable:
        if random.random() > self.config.crossover_rate:
            return p1
        return ThresholdTable(
            buy=(p1.buy + p2.buy) / 2,
            sell=(p1.sell + p2.sell) / 2,
            reduce=(p1.reduce + p2.reduce) / 2,
            hold=(p1.hold + p2.hold) / 2,
        )

    def _mutate(self, ind: ThresholdTable) -> ThresholdTable:
        if random.random() > self.config.mutation_rate:
            return ind
        return ThresholdTable(
            buy=max(0.1, ind.buy + random.uniform(-0.05, 0.05)),
            sell=min(-0.1, ind.sell + random.uniform(-0.05, 0.05)),
            reduce=max(0.05, ind.reduce + random.uniform(-0.02, 0.02)),
            hold=max(0.01, ind.hold + random.uniform(-0.01, 0.01)),
        )


def build_threshold_candidates(count: int = 5) -> tuple[ThresholdCandidate, ...]:
    candidates = []
    for i in range(count):
        spread = 0.20 + (i * 0.05)
        candidates.append(
            ThresholdCandidate(
                candidate_id=f"ga_candidate_{i + 1}",
                buy_threshold=spread,
                sell_threshold=-spread,
                reduce_threshold=max(0.1, spread - 0.1),
                hold_threshold=0.05 + (i * 0.01),
                metadata={"offline_only": True},
            )
        )
    return tuple(candidates)


MaxEntropyConfig = EnsembleConfig
GAThresholdConfig = GAThresholdConfig


def _max_risk_mode(left: RiskMode, right: RiskMode) -> RiskMode:
    ordering = {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }
    return left if ordering[left] >= ordering[right] else right
