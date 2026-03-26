from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
import time
from typing import Callable, Mapping, Sequence

import numpy as np


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


class DeterministicStudentPolicy:
    """
    Lightweight deterministic student inference model for Fast Loop path.

    This class intentionally omits training logic and only exposes deterministic
    inference from fixed, pre-generated weights.
    """

    def __init__(self, weights: np.ndarray, bias: float = 0.0):
        matrix = np.asarray(weights, dtype=float)
        if matrix.ndim != 1:
            raise ValueError("weights must be a 1D vector")
        self._weights = matrix
        self._bias = float(bias)

    @property
    def input_dim(self) -> int:
        return int(self._weights.shape[0])

    def predict(self, observation_vector: Sequence[float]) -> float:
        vector = np.asarray(list(observation_vector), dtype=float)
        if vector.shape[0] != self.input_dim:
            raise ValueError(
                f"observation length {vector.shape[0]} does not match student input_dim {self.input_dim}"
            )
        # Bounded deterministic output in [-1, 1].
        raw = float(np.dot(vector, self._weights) + self._bias)
        return float(np.tanh(raw))


class DistillationMonitor:
    def __init__(self, config: DistillationGateConfig | None = None):
        self.config = config or DistillationGateConfig()

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
            return AgreementReport(
                overall_agreement=0.0,
                average_day_agreement=0.0,
                crisis_slice_agreement=0.0,
                kl_divergence=0.0,
                sample_count=0,
            )

        if crisis_flags is None:
            crisis = np.zeros(shape=teacher.shape, dtype=bool)
        else:
            crisis = np.asarray(list(crisis_flags), dtype=bool)
            if crisis.shape[0] != teacher.shape[0]:
                raise ValueError("crisis_flags must have matching length")

        agreement_mask = np.abs(teacher - student) <= self.config.action_tolerance
        overall = float(np.mean(agreement_mask))

        if np.any(~crisis):
            avg_day = float(np.mean(agreement_mask[~crisis]))
        else:
            avg_day = overall

        if np.any(crisis):
            crisis_agreement = float(np.mean(agreement_mask[crisis]))
        else:
            crisis_agreement = overall

        kl = float(np.mean(_bernoulli_kl(teacher, student)))
        return AgreementReport(
            overall_agreement=overall,
            average_day_agreement=avg_day,
            crisis_slice_agreement=crisis_agreement,
            kl_divergence=kl,
            sample_count=int(teacher.size),
        )

    def is_promotion_eligible(
        self,
        report: AgreementReport,
        *,
        p99_latency_ms: float,
        p999_latency_ms: float,
    ) -> bool:
        return (
            report.average_day_agreement >= self.config.average_day_agreement_min
            and report.crisis_slice_agreement >= self.config.crisis_agreement_min
            and float(p99_latency_ms) <= self.config.fast_loop_p99_budget_ms
            and float(p999_latency_ms) <= self.config.fast_loop_p999_budget_ms
        )

    def drift_detected(self, agreement_history: Sequence[float]) -> bool:
        if not agreement_history:
            return False
        window = np.asarray(list(agreement_history)[-self.config.rolling_window :], dtype=float)
        if window.size == 0:
            return False
        return float(np.mean(window)) < float(self.config.drift_threshold)

    def apply_demotion_if_needed(
        self,
        record: StudentPolicyRecord,
        *,
        agreement_history: Sequence[float],
        now: datetime | None = None,
    ) -> StudentPolicyRecord:
        if not self.drift_detected(agreement_history):
            return record

        timestamp = (now or datetime.now(UTC)).astimezone(UTC)
        return replace(
            record,
            status=StudentPolicyStatus.DEMOTED,
            demoted_at=timestamp,
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

    warmup_runs = max(0, int(warmup))
    first = vectors[0]
    for _ in range(warmup_runs):
        _ = predict_fn(first)

    samples_ms: list[float] = []
    for vector in vectors:
        start_ns = time.perf_counter_ns()
        _ = predict_fn(vector)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        samples_ms.append(float(elapsed_ms))

    values = np.asarray(samples_ms, dtype=float)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
    }


def build_student_record(
    *,
    student_id: str,
    teacher_policy_id: str,
    version: str,
    report: AgreementReport,
    latency_metrics_ms: Mapping[str, float],
    drift_threshold: float,
    promoted: bool,
) -> StudentPolicyRecord:
    promoted_at = datetime.now(UTC) if promoted else None
    status = StudentPolicyStatus.ACTIVE if promoted else StudentPolicyStatus.CANDIDATE
    return StudentPolicyRecord(
        student_id=student_id,
        teacher_policy_id=teacher_policy_id,
        version=version,
        status=status,
        drift_threshold=float(drift_threshold),
        teacher_agreement_pct=float(report.average_day_agreement),
        crisis_agreement_pct=float(report.crisis_slice_agreement),
        p99_inference_ms=float(latency_metrics_ms.get("p99", 0.0)),
        p999_inference_ms=float(latency_metrics_ms.get("p999", 0.0)),
        promoted_at=promoted_at,
    )


def _bernoulli_kl(teacher_actions: np.ndarray, student_actions: np.ndarray) -> np.ndarray:
    # Convert actions in [-1, 1] into Bernoulli probabilities in (0, 1).
    eps = 1e-6
    p = np.clip((teacher_actions + 1.0) * 0.5, eps, 1.0 - eps)
    q = np.clip((student_actions + 1.0) * 0.5, eps, 1.0 - eps)
    return p * np.log(p / q) + ((1.0 - p) * np.log((1.0 - p) / (1.0 - q)))
