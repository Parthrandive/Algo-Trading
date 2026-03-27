from __future__ import annotations

from src.agents.strategic.latency_discipline import (
    BenchmarkEvidence,
    CILatencyBenchmarkGate,
    FastLoopLatencyDiscipline,
)


def test_fastloop_degrades_above_threshold_and_restores_after_healthy_windows():
    tracker = FastLoopLatencyDiscipline()

    for sample in (6.0, 7.0, 12.0):
        tracker.record_stage_latency("window_a", sample)
    degraded = tracker.evaluate_mode(stage="window_a")
    assert degraded.mode == "degraded"
    assert degraded.event_type == "FASTLOOP_DEGRADE"

    for _ in range(3):
        tracker.record_stage_latency("window_b", 7.5)
        recovered = tracker.evaluate_mode(stage="window_b")
    assert recovered.mode == "normal"
    assert recovered.event_type == "FASTLOOP_RESTORE"


def test_ci_latency_gate_blocks_regressions_and_threshold_breaches():
    gate = CILatencyBenchmarkGate()
    result = gate.evaluate(
        BenchmarkEvidence(
            replay_p99_ms=10.5,
            replay_p999_ms=11.0,
            peak_p99_ms=11.2,
            peak_p999_ms=12.0,
            baseline_p99_ms=8.0,
            correctness_pass=True,
            degrade_path_pass=True,
        ),
    )
    assert result.passed is False
    assert "p99_exceeds_degrade_threshold" in result.reasons
    assert "p99_regression_exceeds_guard" in result.reasons


def test_ci_latency_gate_passes_when_all_constraints_hold():
    gate = CILatencyBenchmarkGate()
    result = gate.evaluate(
        BenchmarkEvidence(
            replay_p99_ms=7.8,
            replay_p999_ms=9.1,
            peak_p99_ms=8.2,
            peak_p999_ms=9.5,
            baseline_p99_ms=8.0,
            correctness_pass=True,
            degrade_path_pass=True,
        ),
    )
    assert result.passed is True
    assert result.reasons == ()
