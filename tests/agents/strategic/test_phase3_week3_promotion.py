from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.promotion_gates import (
    PolicyStage,
    PromotionEvidence,
    PromotionGatePipeline,
    RollbackDrillManager,
    SafeRLTrainingGuard,
)
from src.agents.strategic.schemas import RiskMode


def _passing_evidence() -> PromotionEvidence:
    return PromotionEvidence(
        walk_forward_outperformance=True,
        non_regression=True,
        crisis_slice_agreement=0.90,
        latency_gate_passed=True,
        rollback_ready=True,
        critical_compliance_violations=0,
    )


def test_candidate_to_shadow_requires_rollback_and_zero_critical_compliance():
    pipeline = PromotionGatePipeline()
    decision = pipeline.transition(
        current_stage=PolicyStage.CANDIDATE,
        requested_stage=PolicyStage.SHADOW,
        evidence=_passing_evidence(),
    )
    assert decision.approved is True
    assert decision.to_stage == PolicyStage.SHADOW


def test_shadow_to_champion_blocks_if_evidence_is_incomplete():
    pipeline = PromotionGatePipeline()
    evidence = PromotionEvidence(
        walk_forward_outperformance=False,
        non_regression=False,
        crisis_slice_agreement=0.65,
        latency_gate_passed=False,
        rollback_ready=False,
        critical_compliance_violations=1,
    )
    decision = pipeline.transition(
        current_stage=PolicyStage.SHADOW,
        requested_stage=PolicyStage.CHAMPION,
        evidence=evidence,
    )
    assert decision.approved is False
    assert decision.to_stage == PolicyStage.SHADOW
    assert "walk_forward_outperformance_missing" in decision.reasons
    assert "critical_compliance_violation_detected" in decision.reasons


def test_safe_training_guard_clamps_sizes_by_risk_mode():
    guard = SafeRLTrainingGuard()
    assert guard.clamp_action_size(0.9, risk_mode=RiskMode.NORMAL) == 0.9
    assert guard.clamp_action_size(0.9, risk_mode=RiskMode.REDUCE_ONLY) == 0.5
    assert guard.clamp_action_size(0.9, risk_mode=RiskMode.CLOSE_ONLY) == 0.25
    assert guard.clamp_action_size(0.9, risk_mode=RiskMode.KILL_SWITCH) == 0.0


def test_rollback_drill_reverts_to_previous_champion_and_reports_mttr():
    drill = RollbackDrillManager()
    drill.set_champion("model_v1")
    drill.set_champion("model_v2")

    t0 = datetime(2026, 4, 18, 10, 0, tzinfo=UTC)
    result = drill.rollback(failed_model_id="model_v2", started_at=t0, ended_at=t0 + timedelta(seconds=42))
    assert result.executed is True
    assert result.reverted_to == "model_v1"
    assert result.mttr_seconds == 42.0
