from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.agents.strategic.config import PolicySnapshotConfig
from src.agents.strategic.schemas import (
    PolicySnapshot,
    PolicySnapshotSource,
    QualityStatus,
    RiskMode,
    StudentPolicyArtifact,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotManagerConfig:
    """Compatibility wrapper for Week 2 configuration."""
    refresh_interval: timedelta = timedelta(minutes=10)
    snapshot_ttl: timedelta = timedelta(minutes=15)
    stale_after: timedelta = timedelta(minutes=12)
    required_quality_status: QualityStatus = QualityStatus.PASS


class PolicySnapshotManager:
    """
    Manages active policy snapshots with atomic pointer-swap.
    Ensures Fast Loop always has a valid (or neutral) policy to execute.
    """

    def __init__(self, config: PolicySnapshotConfig | SnapshotManagerConfig | None = None):
        if isinstance(config, SnapshotManagerConfig):
             # Map legacy config to central config if needed, but we prefer central
             self.config = PolicySnapshotConfig(
                 refresh_interval=config.refresh_interval,
                 snapshot_ttl=config.snapshot_ttl,
                 stale_after=config.stale_after,
                 required_quality_status=config.required_quality_status.value if isinstance(config.required_quality_status, QualityStatus) else str(config.required_quality_status)
             )
        else:
            self.config = config or PolicySnapshotConfig()
            
        self._active_snapshot: PolicySnapshot | None = None
        self._lock = threading.Lock()
        self._forced_mode: RiskMode | None = None
        self._forced_reason: str | None = None

    def active_snapshot(self) -> PolicySnapshot | None:
        return self._active_snapshot

    def publish_snapshot(self, snapshot: PolicySnapshot, *, trigger: str = "manual", emergency: bool = False) -> None:
        """Atomically swap the active snapshot."""
        with self._lock:
            if not emergency and snapshot.quality_status == QualityStatus.FAIL:
                logger.warning(f"Rejected failed quality snapshot: {snapshot.snapshot_id}")
                return
            
            # Atomic update
            self._active_snapshot = snapshot
            logger.info(f"Published new snapshot {snapshot.snapshot_id} triggered by {trigger}")

    def force_mode(self, mode: RiskMode, reason: str = "manual_override") -> None:
        """Forces the risk mode (e.g. Kill Switch)."""
        with self._lock:
            self._forced_mode = mode
            self._forced_reason = reason
            logger.warning(f"Forcing RiskMode.{mode.name} due to: {reason}")

    def get_effective_risk_mode(self) -> RiskMode:
        """Returns the forced mode or the snapshot's mode."""
        if self._forced_mode is not None:
            return self._forced_mode
        if self._active_snapshot is not None:
            return self._active_snapshot.risk_mode
        return RiskMode.NORMAL

    def is_stale(self, now: datetime | None = None) -> bool:
        """Checks if the active snapshot is stale based on config."""
        if self._active_snapshot is None:
            return True
        now = now or datetime.now(UTC)
        return now > self._active_snapshot.generated_at + self.config.stale_after

    def build_snapshot(
        self,
        artifact: StudentPolicyArtifact,
        source_type: PolicySnapshotSource,
        *,
        generated_at: datetime | None = None,
        risk_mode: RiskMode = RiskMode.NORMAL,
        quality_status: QualityStatus = QualityStatus.PASS,
        metadata: dict[str, Any] | None = None,
    ) -> PolicySnapshot:
        """Helper to build a validated snapshot artifact."""
        generated_at = generated_at or datetime.now(UTC)
        expires_at = generated_at + self.config.snapshot_ttl
        
        return PolicySnapshot(
            snapshot_id=f"snap_{artifact.policy_id}_{generated_at.strftime('%Y%m%dT%H%M%S')}",
            generated_at=generated_at,
            expires_at=expires_at,
            quality_status=quality_status,
            source_type=source_type,
            active_policy_id=artifact.policy_id,
            risk_mode=risk_mode,
            policy=artifact,
            metadata=metadata or {},
        )

    # Alias for main branch compatibility
    def publish(self, snapshot: PolicySnapshot) -> None:
        self.publish_snapshot(snapshot, trigger="main_compat")
