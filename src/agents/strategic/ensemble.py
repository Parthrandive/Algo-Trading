from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

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
    Advanced Phase 3 Maximum Entropy Ensemble.
    Calculates dynamic policy weights based on confidence and diversity.
    """

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()

    def decide(
        self,
        symbol: str,
        observations: tuple[StrategicObservation, ...],
        policy_actions: Sequence[PolicyAction],
        *,
        thresholds: ThresholdTable | ThresholdCandidate | None = None,
        risk_mode: RiskMode = RiskMode.NORMAL,
    ) -> EnsembleDecision:
        if not policy_actions:
            raise ValueError("policy_actions must not be empty")

        now = datetime.now(UTC)
        snapshot_id = observations[0].snapshot_id if observations else "missing"

        # 1. Calculate weights using confidence-weighted softmax
        confidences = np.asarray([max(self.config.confidence_floor, float(a.confidence)) for a in policy_actions])
        
        # Diversity check: penalize redundant actions
        diversities = self._calculate_diversity_scores(policy_actions)
        
        # Combined score for softmax
        scores = confidences * (1.0 + diversities * self.config.diversity_penalty)
        
        # Apply consensus and crisis adjustments
        if observations:
            obs = observations[0]
            for i, action in enumerate(policy_actions):
                if obs.consensus_direction == "BUY" and action.action == ActionType.BUY:
                    scores[i] += self.config.consensus_boost
                elif obs.consensus_direction == "SELL" and action.action == ActionType.SELL:
                    scores[i] += self.config.consensus_boost
                
                if obs.crisis_mode:
                    scores[i] *= (1.0 - self.config.crisis_confidence_haircut)
                if obs.agent_divergence:
                    scores[i] = min(scores[i], self.config.divergence_hold_confidence_cap)

        weights = self._softmax(scores / max(self.config.temperature, 1e-6))
        
        # Apply minimum weight floor
        weights = self._apply_min_weight_floor(weights, self.config.minimum_policy_weight)

        # 2. Weighted aggregate action
        action_vals = np.asarray([_ACTION_TARGETS[a.action] * a.action_size for a in policy_actions])
        aggregate_val = float(np.sum(action_vals * weights))

        # 3. Apply thresholds to determine discrete action
        final_action = self._threshold_to_action(aggregate_val, thresholds)
        
        # 4. Resolve Risk Mode
        if observations:
            obs = observations[0]
            if obs.crisis_mode:
                risk_mode = RiskMode.CLOSE_ONLY
                if final_action == ActionType.BUY:
                    final_action = ActionType.HOLD
            elif obs.agent_divergence:
                risk_mode = RiskMode.REDUCE_ONLY

        # Dominant policy for audit
        dominant_idx = int(np.argmax(weights))
        dominant_policy_id = policy_actions[dominant_idx].policy_id

        policy_weight_schemas = tuple(
            PolicyWeight(
                policy_id=a.policy_id,
                weight=float(weights[i]),
                confidence=float(a.confidence),
                diversity_score=float(diversities[i]),
            )
            for i, a in enumerate(policy_actions)
        )

        return EnsembleDecision(
            timestamp=now,
            symbol=symbol,
            observation_snapshot_id=snapshot_id,
            action=final_action,
            action_size=abs(aggregate_val) if final_action != ActionType.HOLD else 0.0,
            confidence=float(np.mean(confidences)),
            mode="max_entropy_v1",
            dominant_policy_id=dominant_policy_id,
            policy_weights=policy_weight_schemas,
            threshold_candidate_id=getattr(thresholds, "candidate_id", "default") if thresholds else "default",
            rationale=f"ensemble_score={aggregate_val:.4f}; dominant={dominant_policy_id}",
            risk_mode=risk_mode,
            metadata={
                "temperature": self.config.temperature,
                "diversity_penalty": self.config.diversity_penalty,
                "weighted_aggregate": aggregate_val,
            },
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _apply_min_weight_floor(self, weights: np.ndarray, floor: float) -> np.ndarray:
        if floor <= 0.0:
            return weights
        num = weights.size
        # Simple floor logic
        weights = np.maximum(weights, floor)
        return weights / weights.sum()

    def _calculate_diversity_scores(self, actions: Sequence[PolicyAction]) -> np.ndarray:
        if len(actions) <= 1:
            return np.ones(len(actions))
        
        action_signs = np.asarray([_ACTION_TARGETS[a.action] for a in actions])
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


class OfflineGeneticThresholdOptimizer:
    """
    Optimizes ThresholdTable using GA against historical deliberation logs.
    """

    def __init__(self, config: GAThresholdConfig | None = None):
        self.config = config or GAThresholdConfig()

    def optimize(
        self,
        historical_data: Iterable[Mapping[str, Any]],
        fitness_fn: Callable[[ThresholdTable, Iterable[Mapping[str, Any]]], float],
    ) -> GAOptimizationResult:
        population = self._init_population()
        history: list[float] = []

        for gen in range(self.config.generations):
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
        # Tournament selection
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
    """Offline-only threshold candidate generator."""
    candidates = []
    for i in range(count):
        spread = 0.20 + (i * 0.05)
        candidates.append(
            ThresholdCandidate(
                candidate_id=f"ga_candidate_{i+1}",
                buy_threshold=spread,
                sell_threshold=-spread,
                reduce_threshold=max(0.1, spread - 0.1),
                hold_threshold=0.05 + (i * 0.01),
                metadata={"offline_only": True},
            )
        )
    return tuple(candidates)

# Compatibility Aliases
MaxEntropyConfig = EnsembleConfig
GAThresholdConfig = GAThresholdConfig
