from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from src.agents.strategic.config import LatencyDisciplineConfig


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1)))
    return float(ordered[idx])


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    jitter_ms: float


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    changed: bool
    event_type: str
    reason: str
    p99_ms: float


@dataclass(frozen=True)
class BenchmarkEvidence:
    replay_p99_ms: float
    replay_p999_ms: float
    peak_p99_ms: float
    peak_p999_ms: float
    baseline_p99_ms: float
    correctness_pass: bool
    degrade_path_pass: bool


@dataclass(frozen=True)
class BenchmarkGateResult:
    passed: bool
    reasons: tuple[str, ...]
    artifact: dict[str, object]


class FastLoopLatencyDiscipline:
    """
    Tier 1-D latency instrumentation + degrade safeguard controller.
    """

    def __init__(self, config: LatencyDisciplineConfig | None = None) -> None:
        self.config = config or LatencyDisciplineConfig()
        self._stage_samples: dict[str, list[float]] = {}
        self._mode = "normal"
        self._healthy_windows = 0
        self._events: list[dict[str, object]] = []

    def record_stage_latency(self, stage: str, duration_ms: float) -> None:
        self._stage_samples.setdefault(stage, []).append(float(duration_ms))

    def summarize(self, stage: str) -> LatencySummary:
        samples = list(self._stage_samples.get(stage, []))
        if not samples:
            return LatencySummary(count=0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, p999_ms=0.0, jitter_ms=0.0)
        p50 = _percentile(samples, 50.0)
        p95 = _percentile(samples, 95.0)
        p99 = _percentile(samples, 99.0)
        p999 = _percentile(samples, 99.9)
        jitter = max(0.0, p999 - p50)
        return LatencySummary(
            count=len(samples),
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            p999_ms=p999,
            jitter_ms=jitter,
        )

    def evaluate_mode(self, stage: str = "decision_path") -> ModeDecision:
        summary = self.summarize(stage)
        p99 = summary.p99_ms
        changed = False
        reason = "within_budget"
        event_type = "FASTLOOP_OK"

        if p99 > float(self.config.degrade_threshold_ms):
            self._healthy_windows = 0
            if self._mode != "degraded":
                self._mode = "degraded"
                changed = True
            reason = "p99_exceeds_degrade_threshold"
            event_type = "FASTLOOP_DEGRADE"
        elif p99 <= float(self.config.p99_target_ms):
            self._healthy_windows += 1
            if self._mode == "degraded" and self._healthy_windows >= int(self.config.restore_consecutive_windows):
                self._mode = "normal"
                changed = True
                reason = "healthy_windows_recovered"
                event_type = "FASTLOOP_RESTORE"
            else:
                reason = "within_target"
                event_type = "FASTLOOP_OK"
        else:
            self._healthy_windows = 0
            reason = "between_target_and_degrade_threshold"
            event_type = "FASTLOOP_WARN"

        self._events.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "mode": self._mode,
                "stage": stage,
                "p99_ms": p99,
                "event_type": event_type,
                "reason": reason,
            }
        )
        return ModeDecision(mode=self._mode, changed=changed, event_type=event_type, reason=reason, p99_ms=p99)

    def mode(self) -> str:
        return self._mode

    def recent_events(self, limit: int = 100) -> tuple[dict[str, object], ...]:
        return tuple(self._events[-max(0, limit) :])

    def weekly_report(self, stage: str = "decision_path") -> dict[str, object]:
        summary = self.summarize(stage)
        return {
            "owner_required": True,
            "stage": stage,
            "mode": self._mode,
            "count": summary.count,
            "p50_ms": summary.p50_ms,
            "p95_ms": summary.p95_ms,
            "p99_ms": summary.p99_ms,
            "p999_ms": summary.p999_ms,
            "jitter_ms": summary.jitter_ms,
            "events": self.recent_events(limit=20),
        }


class CILatencyBenchmarkGate:
    """
    CI gate for execution-path changes (replay + peak synthetic).
    """

    def __init__(self, config: LatencyDisciplineConfig | None = None) -> None:
        self.config = config or LatencyDisciplineConfig()

    def evaluate(self, evidence: BenchmarkEvidence) -> BenchmarkGateResult:
        reasons: list[str] = []

        max_p99 = max(evidence.replay_p99_ms, evidence.peak_p99_ms)
        max_p999 = max(evidence.replay_p999_ms, evidence.peak_p999_ms)
        regression = max_p99 - float(evidence.baseline_p99_ms)

        if not evidence.correctness_pass:
            reasons.append("correctness_test_failed")
        if not evidence.degrade_path_pass:
            reasons.append("degrade_path_test_failed")
        if max_p99 > float(self.config.degrade_threshold_ms):
            reasons.append("p99_exceeds_degrade_threshold")
        if regression > float(self.config.regression_guard_ms):
            reasons.append("p99_regression_exceeds_guard")

        artifact = {
            "workloads": {
                "replay": {"p99_ms": evidence.replay_p99_ms, "p999_ms": evidence.replay_p999_ms},
                "peak_synthetic": {"p99_ms": evidence.peak_p99_ms, "p999_ms": evidence.peak_p999_ms},
            },
            "baseline_p99_ms": evidence.baseline_p99_ms,
            "max_p99_ms": max_p99,
            "max_p999_ms": max_p999,
            "regression_ms": regression,
            "target_p99_ms": float(self.config.p99_target_ms),
            "degrade_threshold_ms": float(self.config.degrade_threshold_ms),
        }
        return BenchmarkGateResult(passed=len(reasons) == 0, reasons=tuple(reasons), artifact=artifact)
