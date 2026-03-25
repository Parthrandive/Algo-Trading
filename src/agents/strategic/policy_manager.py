from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from src.agents.strategic.config import DistillationConfig, PolicySnapshotConfig
from src.agents.strategic.distillation import StudentPolicy
from src.agents.strategic.schemas import (
    ActionType,
    LoopType,
    PolicyAction,
    PolicySnapshot,
    PolicySnapshotSource,
    PolicyType,
    QualityStatus,
    RiskMode,
    SnapshotRefreshResult,
    StudentPolicyArtifact,
    StrategicObservation,
)


@dataclass(frozen=True)
class SnapshotStatus:
    valid_for_fast_loop: bool
    risk_mode: RiskMode
    reason: str


class PolicySnapshotManager:
    """
    Atomic cross-loop publication manager.

    Fast Loop always reads the in-memory active snapshot; Slow Loop publishes a
    fully-formed replacement snapshot via one pointer swap.
    """

    def __init__(
        self,
        *,
        snapshot_config: PolicySnapshotConfig | None = None,
        distillation_config: DistillationConfig | None = None,
    ) -> None:
        self.snapshot_config = snapshot_config or PolicySnapshotConfig()
        self.distillation_config = distillation_config or DistillationConfig()
        self._lock = Lock()
        self._active_snapshot: PolicySnapshot | None = None
        self._previous_snapshot: PolicySnapshot | None = None

    def publish(self, snapshot: PolicySnapshot) -> SnapshotRefreshResult:
        with self._lock:
            previous_id = self._active_snapshot.snapshot_id if self._active_snapshot else None
            changed = previous_id != snapshot.snapshot_id
            self._previous_snapshot = self._active_snapshot
            self._active_snapshot = snapshot
        status = self.evaluate_snapshot(snapshot)
        return SnapshotRefreshResult(
            previous_snapshot_id=previous_id,
            active_snapshot_id=snapshot.snapshot_id,
            changed=changed,
            degraded=not status.valid_for_fast_loop,
            risk_mode=status.risk_mode,
            reason=status.reason,
        )

    def build_snapshot(
        self,
        *,
        artifact: StudentPolicyArtifact,
        source_type: PolicySnapshotSource,
        generated_at: datetime | None = None,
        risk_mode: RiskMode = RiskMode.NORMAL,
        quality_status: QualityStatus = QualityStatus.PASS,
        metadata: dict[str, object] | None = None,
    ) -> PolicySnapshot:
        generated_at = generated_at.astimezone(UTC) if generated_at else datetime.now(UTC)
        expires_at = generated_at + self.snapshot_config.snapshot_ttl
        return PolicySnapshot(
            snapshot_id=f"policy-snapshot-{uuid4().hex[:12]}",
            generated_at=generated_at,
            expires_at=expires_at,
            quality_status=quality_status,
            source_type=source_type,
            active_policy_id=artifact.policy_id,
            fallback_policy_id=self._active_snapshot.active_policy_id if self._active_snapshot else None,
            risk_mode=risk_mode,
            policy=artifact,
            metadata=metadata or {},
        )

    def current_snapshot(self) -> PolicySnapshot | None:
        with self._lock:
            return self._active_snapshot

    def evaluate_snapshot(self, snapshot: PolicySnapshot | None = None, *, now: datetime | None = None) -> SnapshotStatus:
        snapshot = snapshot or self.current_snapshot()
        if snapshot is None:
            return SnapshotStatus(False, RiskMode.REDUCE_ONLY, "no active snapshot")
        now = now.astimezone(UTC) if now else datetime.now(UTC)
        if snapshot.quality_status == QualityStatus.FAIL:
            return SnapshotStatus(False, RiskMode.CLOSE_ONLY, "snapshot quality fail")
        if snapshot.expires_at <= now:
            return SnapshotStatus(False, RiskMode.REDUCE_ONLY, "snapshot expired")
        stale_at = snapshot.generated_at + self.snapshot_config.stale_after
        if stale_at <= now:
            return SnapshotStatus(False, RiskMode.REDUCE_ONLY, "snapshot stale")
        return SnapshotStatus(True, snapshot.risk_mode, "snapshot healthy")

    def fast_loop_action(self, observation: StrategicObservation) -> PolicyAction:
        snapshot = self.current_snapshot()
        status = self.evaluate_snapshot(snapshot)
        if snapshot is None or not status.valid_for_fast_loop:
            fallback_action = "reduce_only_fallback" if status.risk_mode != RiskMode.CLOSE_ONLY else "close_only_fallback"
            return PolicyAction(
                policy_id="snapshot_guardrail",
                policy_type=PolicyType.STUDENT,
                loop_type=LoopType.FAST,
                action=_fallback_action_type(status.risk_mode),
                action_size=0.25 if status.risk_mode != RiskMode.CLOSE_ONLY else 1.0,
                confidence=1.0,
                latency_ms=0.05,
                reasoning=fallback_action,
                metadata={"reason": status.reason},
            )
        student = StudentPolicy(snapshot.policy)
        action = student.as_policy_action(observation)
        if action.latency_ms is not None and action.latency_ms > self.distillation_config.fast_loop_degrade_ms:
            return PolicyAction(
                policy_id=action.policy_id,
                policy_type=action.policy_type,
                loop_type=action.loop_type,
                action=_fallback_action_type(RiskMode.REDUCE_ONLY),
                action_size=0.25,
                confidence=1.0,
                latency_ms=action.latency_ms,
                reasoning="fast loop latency breach safeguard",
                metadata={"source_snapshot_id": snapshot.snapshot_id},
            )
        return action

    def emergency_swap_to_previous(self, reason: str) -> SnapshotRefreshResult:
        with self._lock:
            if self._previous_snapshot is None:
                raise RuntimeError("no previous snapshot is available for emergency swap")
            self._active_snapshot, self._previous_snapshot = self._previous_snapshot, self._active_snapshot
            active = self._active_snapshot
        return SnapshotRefreshResult(
            previous_snapshot_id=self._previous_snapshot.snapshot_id if self._previous_snapshot else None,
            active_snapshot_id=active.snapshot_id,
            changed=True,
            degraded=active.risk_mode != RiskMode.NORMAL,
            risk_mode=active.risk_mode,
            reason=reason,
        )


def _fallback_action_type(risk_mode: RiskMode):
    if risk_mode == RiskMode.CLOSE_ONLY:
        return ActionType.CLOSE
    return ActionType.REDUCE
