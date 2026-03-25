from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Iterable

from src.agents.strategic.config import DistillationConfig
from src.agents.strategic.schemas import (
    ActionType,
    AgreementReport,
    DistillationSample,
    EnsembleDecision,
    PolicyAction,
    PolicyType,
    LoopType,
    StudentPolicyArtifact,
    StrategicObservation,
)


_ACTION_TARGETS = {
    ActionType.BUY: 1.0,
    ActionType.SELL: -1.0,
    ActionType.HOLD: 0.0,
    ActionType.CLOSE: -0.5,
    ActionType.REDUCE: -0.25,
}


@dataclass(frozen=True)
class StudentDecision:
    action: ActionType
    action_size: float
    confidence: float
    raw_score: float
    latency_ms: float


class StudentPolicy:
    """
    Lightweight deterministic student-policy runtime.

    This intentionally avoids RL framework imports so it is safe for the Week 2
    Fast Loop candidate path and consistent with the RL dependency policy.
    """

    def __init__(self, artifact: StudentPolicyArtifact) -> None:
        self.artifact = artifact

    def infer(self, observation: StrategicObservation) -> StudentDecision:
        vector = observation.observation_vector
        weights = list(self.artifact.weights)
        if len(weights) != len(vector):
            raise ValueError("student artifact input_dim does not match observation vector length")
        raw_score = sum(value * weight for value, weight in zip(vector, weights)) + self.artifact.bias
        confidence = max(0.0, min(1.0, abs(raw_score)))
        action = self._map_score_to_action(raw_score, observation)
        if observation.crisis_mode:
            size = min(0.25, confidence)
        elif observation.agent_divergence:
            size = min(0.10, confidence)
        else:
            size = min(1.0, max(0.05, confidence))
        latency_ms = 0.25
        return StudentDecision(
            action=action,
            action_size=size if action != ActionType.HOLD else 0.0,
            confidence=confidence,
            raw_score=raw_score,
            latency_ms=latency_ms,
        )

    def as_policy_action(self, observation: StrategicObservation) -> PolicyAction:
        decision = self.infer(observation)
        return PolicyAction(
            policy_id=self.artifact.policy_id,
            policy_type=PolicyType.STUDENT,
            loop_type=LoopType.FAST,
            action=decision.action,
            action_size=decision.action_size,
            confidence=decision.confidence,
            latency_ms=decision.latency_ms,
            reasoning="distilled deterministic student inference",
            metadata={"raw_score": decision.raw_score, "version": self.artifact.version},
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
        aggregates = [0.0] * vector_length
        target_values: list[float] = []
        size_values: list[float] = []
        for sample in sample_list:
            vector = sample.observation.observation_vector
            target = _ACTION_TARGETS[sample.teacher_decision.action] * sample.teacher_decision.confidence
            for index, value in enumerate(vector):
                aggregates[index] += value * target
            target_values.append(target)
            size_values.append(sample.teacher_decision.action_size)

        norm = max(float(len(sample_list)), 1.0)
        weights = tuple(value / norm for value in aggregates)
        bias = mean(target_values) if target_values else 0.0
        return StudentPolicyArtifact(
            policy_id=self.config.student_policy_id,
            version=self.config.student_version,
            generated_at=generated_at,
            input_dim=vector_length,
            output_dim=1,
            weights=weights,
            bias=bias,
            action_map={action.value: score for action, score in _ACTION_TARGETS.items()},
            metadata={
                "offline_only": False,
                "teacher_sample_count": len(sample_list),
                "mean_teacher_action_size": mean(size_values) if size_values else 0.0,
            },
        )

    def evaluate_student(
        self,
        student: StudentPolicy,
        *,
        average_day_observations: Iterable[StrategicObservation],
        crisis_observations: Iterable[StrategicObservation],
        teacher_decisions_by_snapshot: dict[str, EnsembleDecision],
    ) -> AgreementReport:
        average_results = self._evaluate_slice(student, average_day_observations, teacher_decisions_by_snapshot)
        crisis_results = self._evaluate_slice(student, crisis_observations, teacher_decisions_by_snapshot)
        all_latencies = average_results["latencies"] + crisis_results["latencies"]
        average_agreement = average_results["agreement_rate"]
        crisis_agreement = crisis_results["agreement_rate"]
        drift_score = max(0.0, 1.0 - average_agreement)
        demotion_triggered = (
            average_agreement < self.config.agreement_threshold
            or crisis_agreement < self.config.crisis_agreement_threshold
            or drift_score > self.config.drift_alert_threshold
        )
        return AgreementReport(
            policy_id=student.artifact.policy_id,
            agreement_rate=average_agreement,
            crisis_agreement_rate=crisis_agreement,
            average_latency_ms=mean(all_latencies) if all_latencies else 0.0,
            p99_latency_ms=max(all_latencies) if all_latencies else 0.0,
            drift_score=drift_score,
            demotion_triggered=demotion_triggered,
            metadata={
                "average_slice_count": average_results["count"],
                "crisis_slice_count": crisis_results["count"],
                "fast_loop_candidate": True,
            },
        )

    def _evaluate_slice(
        self,
        student: StudentPolicy,
        observations: Iterable[StrategicObservation],
        teacher_decisions_by_snapshot: dict[str, EnsembleDecision],
    ) -> dict[str, float | int | list[float]]:
        matches = 0
        total = 0
        latencies: list[float] = []
        for observation in observations:
            teacher = teacher_decisions_by_snapshot.get(observation.snapshot_id)
            if teacher is None:
                continue
            decision = student.infer(observation)
            total += 1
            latencies.append(decision.latency_ms)
            if decision.action == teacher.action:
                matches += 1
        return {
            "agreement_rate": float(matches / total) if total else 0.0,
            "count": total,
            "latencies": latencies,
        }
