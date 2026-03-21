from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol, Sequence

from src.agents.strategic_executive.schemas import Phase3Observation


@dataclass(frozen=True)
class PolicyAction:
    action: float
    confidence: float
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= float(self.action) <= 1.0:
            raise ValueError("action must be between -1.0 and 1.0")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PolicySnapshot:
    snapshot_id: str
    generated_at_utc: datetime
    policy_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty")
        if self.generated_at_utc.tzinfo is None:
            object.__setattr__(self, "generated_at_utc", self.generated_at_utc.replace(tzinfo=UTC))
        else:
            object.__setattr__(self, "generated_at_utc", self.generated_at_utc.astimezone(UTC))


class TeacherPolicy(Protocol):
    policy_name: str

    def train(self, observations: Sequence[Phase3Observation], *, seed: int = 42) -> Mapping[str, float]:
        """Train or refresh the teacher policy offline."""

    def predict(self, observation: Phase3Observation) -> PolicyAction:
        """Infer teacher action from one observation."""


class EnsembleDecisionEngine(Protocol):
    def select_action(
        self,
        observation: Phase3Observation,
        teacher_actions: Mapping[str, PolicyAction],
    ) -> PolicyAction:
        """Combine teacher actions into one maximum-entropy ensemble action."""


class StudentExecutionPolicy(Protocol):
    policy_name: str

    def predict_fast(self, observation: Phase3Observation) -> PolicyAction:
        """Deterministic low-latency student inference for Fast Loop."""


class TeacherMonitoringPolicy(Protocol):
    def evaluate_agreement(
        self,
        observation: Phase3Observation,
        *,
        teacher_action: PolicyAction,
        student_action: PolicyAction,
    ) -> Mapping[str, float]:
        """Return agreement/drift metrics used by promotion and demotion gates."""


class StrategicExecutiveRuntime(Protocol):
    def publish_snapshot(self, snapshot: PolicySnapshot) -> None:
        """Atomically publish a new policy snapshot for Fast Loop consumption."""

    def run_fast_loop(self, observation: Phase3Observation) -> PolicyAction:
        """Fast Loop execution-path decision."""

    def run_slow_loop(self, observations: Sequence[Phase3Observation]) -> PolicySnapshot:
        """Slow Loop monitoring, calibration, and publication path."""
