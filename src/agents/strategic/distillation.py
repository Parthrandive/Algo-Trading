from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from statistics import mean
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from src.agents.strategic.config import DistillationConfig
from src.agents.strategic.schemas import (
    ActionType,
    AgreementReport as AgreementReportSchema,
    DistillationSample,
    EnsembleDecision,
    LoopType,
    PolicyAction,
    PolicyType,
    StrategicObservation,
    StudentPolicyArtifact,
)


class StudentPolicyStatus(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEMOTED = "demoted"


@dataclass(frozen=True)
class DistillationGateConfig:
    average_day_agreement_min: float = 0.85
    crisis_agreement_min: float = 0.80
    action_tolerance: float = 0.15
    fast_loop_p99_budget_ms: float = 8.0
    fast_loop_p999_budget_ms: float = 10.0
    drift_threshold: float = 0.75
    rolling_window: int = 200

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.average_day_agreement_min) <= 1.0:
            raise ValueError("average_day_agreement_min must be within [0.0, 1.0]")
        if not 0.0 <= float(self.crisis_agreement_min) <= 1.0:
            raise ValueError("crisis_agreement_min must be within [0.0, 1.0]")
        if float(self.action_tolerance) <= 0.0:
            raise ValueError("action_tolerance must be > 0")
        if float(self.fast_loop_p99_budget_ms) <= 0.0:
            raise ValueError("fast_loop_p99_budget_ms must be > 0")
        if float(self.fast_loop_p999_budget_ms) <= 0.0:
            raise ValueError("fast_loop_p999_budget_ms must be > 0")
        if not 0.0 <= float(self.drift_threshold) <= 1.0:
            raise ValueError("drift_threshold must be within [0.0, 1.0]")
        if int(self.rolling_window) < 10:
            raise ValueError("rolling_window must be >= 10")


@dataclass(frozen=True)
class AgreementReport:
    overall_agreement: float
    average_day_agreement: float
    crisis_slice_agreement: float
    kl_divergence: float
    sample_count: int


@dataclass(frozen=True)
class StudentPolicyRecord:
    student_id: str
    teacher_policy_id: str
    version: str
    status: StudentPolicyStatus = StudentPolicyStatus.CANDIDATE
    compression_method: str = "distillation"
    drift_threshold: float = 0.75
    teacher_agreement_pct: float = 0.0
    crisis_agreement_pct: float = 0.0
    p99_inference_ms: float = 0.0
    p999_inference_ms: float = 0.0
    promoted_at: datetime | None = None
    demoted_at: datetime | None = None


_ACTION_TARGETS = {
    ActionType.BUY: 1.0,
    ActionType.SELL: -1.0,
    ActionType.HOLD: 0.0,
    ActionType.CLOSE: -0.5,
    ActionType.REDUCE: -0.25,
}


class DeterministicStudentPolicy:
    """
    Lightweight deterministic student inference model for Fast Loop path.
    Optimized with NumPy for sub-millisecond execution.
    """

    def __init__(self, artifact: StudentPolicyArtifact):
        self.artifact = artifact
        self._weights = np.asarray(artifact.weights, dtype=float)
        self._bias = float(artifact.bias)
        if self._weights.ndim != 1:
            raise ValueError("weights must be a 1D vector")

    @property
    def input_dim(self) -> int:
        return int(self._weights.shape[0])

    def predict(self, observation_vector: Sequence[float]) -> float:
        vector = np.asarray(list(observation_vector), dtype=float)
        if vector.shape[0] != self.input_dim:
            raise ValueError(
                f"observation length {vector.shape[0]} does not match student input_dim {self.input_dim}"
            )
        raw = float(np.dot(vector, self._weights) + self._bias)
        return float(np.tanh(raw))

    def as_policy_action(self, observation: StrategicObservation) -> PolicyAction:
        raw_score = self.predict(observation.observation_vector)
        confidence = max(0.0, min(1.0, abs(raw_score)))
        action = self._map_score_to_action(raw_score, observation)
        
        if observation.crisis_mode:
            size = min(0.25, confidence)
        elif observation.agent_divergence:
            size = min(0.10, confidence)
        else:
            size = min(1.0, max(0.05, confidence))

        return PolicyAction(
            policy_id=self.artifact.policy_id,
            policy_type=PolicyType.STUDENT,
            loop_type=LoopType.FAST,
            action=action,
            action_size=size if action != ActionType.HOLD else 0.0,
            confidence=confidence,
            latency_ms=0.25, # Base latency placeholder
            reasoning="distilled deterministic student inference",
            metadata={"raw_score": raw_score, "version": self.artifact.version},
        )

    def _map_score_to_action(self, raw_score: float, observation: StrategicObservation) -> ActionType:
        if observation.crisis_mode:
            return ActionType.REDUCE
        if raw_score >= 0.30:
            return ActionType.BUY
        if raw_score <= -0.30:
            return ActionType.SELL
        if abs(raw_score) >= 0.12:
            return ActionType.REDUCE
        return ActionType.HOLD


# Alias for backward compatibility
StudentPolicy = DeterministicStudentPolicy


class DistillationMonitor:
    def __init__(self, config: DistillationConfig | None = None):
        self.config = config or DistillationConfig()

    def build_agreement_report(
        self,
        *,
        teacher_actions: Sequence[float],
        student_actions: Sequence[float],
        crisis_flags: Sequence[bool] | None = None,
    ) -> AgreementReport:
        teacher = np.asarray(list(teacher_actions), dtype=float)
        student = np.asarray(list(student_actions), dtype=float)
        if teacher.shape[0] != student.shape[0]:
            raise ValueError("teacher_actions and student_actions must have matching lengths")
        if teacher.size == 0:
            return AgreementReport(0.0, 0.0, 0.0, 0.0, 0)

        if crisis_flags is None:
            crisis = np.zeros(shape=teacher.shape, dtype=bool)
        else:
            crisis = np.asarray(list(crisis_flags), dtype=bool)

        agreement_mask = np.abs(teacher - student) <= 0.15 # tolerance
        overall = float(np.mean(agreement_mask))
        avg_day = float(np.mean(agreement_mask[~crisis])) if np.any(~crisis) else overall
        crisis_agreement = float(np.mean(agreement_mask[crisis])) if np.any(crisis) else overall
        kl = float(np.mean(_bernoulli_kl(teacher, student)))

        return AgreementReport(
            overall_agreement=overall,
            average_day_agreement=avg_day,
            crisis_slice_agreement=crisis_agreement,
            kl_divergence=kl,
            sample_count=int(teacher.size),
        )


class DistillationPlanner:
    def __init__(self, config: DistillationConfig | None = None) -> None:
        self.config = config or DistillationConfig()

    def build_student_artifact(
        self,
        samples: Iterable[DistillationSample],
        *,
        generated_at: datetime | None = None,
    ) -> StudentPolicyArtifact:
        sample_list = tuple(samples)
        if not sample_list:
            raise ValueError("samples must not be empty")
        generated_at = generated_at or datetime.now(UTC)
        vector_length = len(sample_list[0].observation.observation_vector)
        aggregates = np.zeros(vector_length)
        target_values = []
        
        for sample in sample_list:
            vector = np.asarray(sample.observation.observation_vector)
            target = _ACTION_TARGETS[sample.teacher_decision.action] * sample.teacher_decision.confidence
            aggregates += vector * target
            target_values.append(target)

        norm = max(float(len(sample_list)), 1.0)
        weights = aggregates / norm
        bias = float(np.mean(target_values)) if target_values else 0.0
        
        return StudentPolicyArtifact(
            policy_id=self.config.student_policy_id,
            version=self.config.student_version,
            generated_at=generated_at,
            input_dim=vector_length,
            output_dim=1,
            weights=tuple(weights.tolist()),
            bias=bias,
            action_map={action.value: score for action, score in _ACTION_TARGETS.items()},
            metadata={"teacher_sample_count": len(sample_list)},
        )


def benchmark_inference_latency(
    predict_fn: Callable[[Sequence[float]], float],
    observation_vectors: Sequence[Sequence[float]],
    *,
    warmup: int = 16,
) -> dict[str, float]:
    vectors = [np.asarray(list(item), dtype=float) for item in observation_vectors]
    if not vectors:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}

    for _ in range(max(0, warmup)):
        _ = predict_fn(vectors[0])

    samples_ms: list[float] = []
    for vector in vectors:
        start_ns = time.perf_counter_ns()
        _ = predict_fn(vector)
        samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)

    values = np.asarray(samples_ms)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
    }


def _bernoulli_kl(teacher_actions: np.ndarray, student_actions: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip((teacher_actions + 1.0) * 0.5, eps, 1.0 - eps)
    q = np.clip((student_actions + 1.0) * 0.5, eps, 1.0 - eps)
    return p * np.log(p / q) + ((1.0 - p) * np.log((1.0 - p) / (1.0 - q)))
