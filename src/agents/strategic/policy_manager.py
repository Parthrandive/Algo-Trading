from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import threading
from typing import Any, Mapping


class RiskMode(str, Enum):
    NORMAL = "normal"
    REDUCE_ONLY = "reduce_only"
    CLOSE_ONLY = "close_only"
    KILL_SWITCH = "kill_switch"


_MODE_ORDER = {
    RiskMode.NORMAL: 0,
    RiskMode.REDUCE_ONLY: 1,
    RiskMode.CLOSE_ONLY: 2,
    RiskMode.KILL_SWITCH: 3,
}


@dataclass(frozen=True)
class PolicySnapshot:
    snapshot_id: str
    policy_id: str
    policy_type: str
    generated_at: datetime
    expires_at: datetime
    schema_version: str
    quality_status: str
    source_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty")
        if not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if self.generated_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("generated_at and expires_at must be timezone-aware")
        if self.expires_at <= self.generated_at:
            raise ValueError("expires_at must be greater than generated_at")

    @property
    def generated_at_utc(self) -> datetime:
        return self.generated_at.astimezone(UTC)

    @property
    def expires_at_utc(self) -> datetime:
        return self.expires_at.astimezone(UTC)


@dataclass(frozen=True)
class FastLoopSnapshotView:
    snapshot: PolicySnapshot | None
    mode: RiskMode
    reason: str
    refresh_required: bool


@dataclass(frozen=True)
class SnapshotManagerConfig:
    refresh_interval: timedelta = timedelta(minutes=10)
    stale_after: timedelta = timedelta(minutes=2)
    refresh_buffer_before_expiry: timedelta = timedelta(seconds=30)

    def __post_init__(self) -> None:
        if self.refresh_interval.total_seconds() <= 0:
            raise ValueError("refresh_interval must be > 0")
        if self.stale_after.total_seconds() <= 0:
            raise ValueError("stale_after must be > 0")
        if self.refresh_buffer_before_expiry.total_seconds() < 0:
            raise ValueError("refresh_buffer_before_expiry must be >= 0")


class PolicySnapshotManager:
    """
    Week 2 Day 3 snapshot handoff manager.

    Guarantees atomic pointer-swap semantics for Fast Loop reads:
    readers observe either the old complete snapshot or the new complete snapshot.
    """

    def __init__(self, config: SnapshotManagerConfig | None = None):
        self.config = config or SnapshotManagerConfig()
        self._lock = threading.RLock()
        self._active_snapshot: PolicySnapshot | None = None
        self._last_refresh_at: datetime | None = None
        self._last_refresh_trigger: str | None = None
        self._mode: RiskMode = RiskMode.NORMAL
        self._mode_reason: str = "healthy"

    def publish_snapshot(
        self,
        snapshot: PolicySnapshot,
        *,
        trigger: str,
        emergency: bool = False,
    ) -> PolicySnapshot:
        now = datetime.now(UTC)
        with self._lock:
            self._active_snapshot = snapshot
            self._last_refresh_at = now
            self._last_refresh_trigger = trigger

            quality = snapshot.quality_status.strip().lower()
            if emergency:
                self._escalate_mode(RiskMode.REDUCE_ONLY, reason=f"emergency swap via {trigger}")
            if quality == "fail":
                self._escalate_mode(RiskMode.CLOSE_ONLY, reason="snapshot quality fail")
            elif quality == "warn":
                self._escalate_mode(RiskMode.REDUCE_ONLY, reason="snapshot quality warn")

        return snapshot

    def active_snapshot(self) -> PolicySnapshot | None:
        with self._lock:
            return self._active_snapshot

    def consume_fast_loop(self, *, now: datetime | None = None) -> FastLoopSnapshotView:
        ts = (now or datetime.now(UTC)).astimezone(UTC)
        with self._lock:
            snapshot = self._active_snapshot
            mode = self._mode
            reason = self._mode_reason

        if mode == RiskMode.KILL_SWITCH:
            return FastLoopSnapshotView(snapshot=snapshot, mode=mode, reason=reason, refresh_required=True)

        if snapshot is None:
            self._escalate_mode(RiskMode.REDUCE_ONLY, reason="missing snapshot")
            return FastLoopSnapshotView(
                snapshot=None,
                mode=self._mode,
                reason="missing snapshot",
                refresh_required=True,
            )

        quality = snapshot.quality_status.strip().lower()
        if quality == "fail":
            self._escalate_mode(RiskMode.CLOSE_ONLY, reason="snapshot quality fail")
            return FastLoopSnapshotView(
                snapshot=snapshot,
                mode=self._mode,
                reason="snapshot quality fail",
                refresh_required=True,
            )

        if snapshot.expires_at_utc <= ts:
            self._escalate_mode(RiskMode.REDUCE_ONLY, reason="snapshot expired")
            return FastLoopSnapshotView(
                snapshot=snapshot,
                mode=self._mode,
                reason="snapshot expired",
                refresh_required=True,
            )

        age = ts - snapshot.generated_at_utc
        if age > self.config.stale_after:
            self._escalate_mode(RiskMode.REDUCE_ONLY, reason="snapshot stale")
            return FastLoopSnapshotView(
                snapshot=snapshot,
                mode=self._mode,
                reason="snapshot stale",
                refresh_required=True,
            )

        refresh_required = self.should_refresh(now=ts)
        return FastLoopSnapshotView(
            snapshot=snapshot,
            mode=self._mode,
            reason=self._mode_reason,
            refresh_required=refresh_required,
        )

    def should_refresh(self, *, now: datetime | None = None) -> bool:
        ts = (now or datetime.now(UTC)).astimezone(UTC)
        with self._lock:
            snapshot = self._active_snapshot
            last_refresh = self._last_refresh_at

        if snapshot is None or last_refresh is None:
            return True

        interval_elapsed = (ts - last_refresh) >= self.config.refresh_interval
        expires_soon = (snapshot.expires_at_utc - ts) <= self.config.refresh_buffer_before_expiry
        return bool(interval_elapsed or expires_soon)

    def next_refresh_due_at(self) -> datetime | None:
        with self._lock:
            if self._last_refresh_at is None:
                return None
            return self._last_refresh_at + self.config.refresh_interval

    def force_mode(self, mode: RiskMode, *, reason: str) -> RiskMode:
        with self._lock:
            self._escalate_mode(mode, reason=reason)
            return self._mode

    def recover_mode(self, *, target: RiskMode = RiskMode.NORMAL, operator_ack: bool = False) -> bool:
        with self._lock:
            if _MODE_ORDER[target] > _MODE_ORDER[self._mode]:
                self._escalate_mode(target, reason="explicit escalation")
                return True

            if self._mode == RiskMode.KILL_SWITCH and not operator_ack:
                return False

            self._mode = target
            self._mode_reason = "manual recovery"
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._active_snapshot
            return {
                "mode": self._mode.value,
                "mode_reason": self._mode_reason,
                "last_refresh_at": self._last_refresh_at,
                "last_refresh_trigger": self._last_refresh_trigger,
                "active_snapshot_id": snapshot.snapshot_id if snapshot else None,
                "active_snapshot_quality": snapshot.quality_status if snapshot else None,
            }

    def _escalate_mode(self, mode: RiskMode, *, reason: str) -> None:
        if _MODE_ORDER[mode] >= _MODE_ORDER[self._mode]:
            self._mode = mode
            self._mode_reason = reason
