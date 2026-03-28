from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.risk_overseer import (
    RiskOverseerStateMachine,
    RiskOverseerThresholds,
    RiskSignalSnapshot,
)
from src.agents.strategic.schemas import RiskMode


def _snapshot(**kwargs) -> RiskSignalSnapshot:
    payload = {
        "timestamp": datetime(2026, 5, 4, 9, 15, tzinfo=UTC),
    }
    payload.update(kwargs)
    return RiskSignalSnapshot(**payload)


def test_full_crisis_requires_hysteresis_then_closes_only():
    overseer = RiskOverseerStateMachine(
        thresholds=RiskOverseerThresholds(crisis_hysteresis_ticks=2, crisis_weight_cap=0.70),
    )

    first = overseer.evaluate(
        _snapshot(
            realized_vol_break=True,
            liquidity_deterioration=True,
            confidence_floor_breach=True,
        )
    )
    assert first.mode == RiskMode.NORMAL
    assert first.trigger_layer == "NONE"

    second = overseer.evaluate(
        _snapshot(
            timestamp=datetime(2026, 5, 4, 9, 16, tzinfo=UTC),
            realized_vol_break=True,
            liquidity_deterioration=True,
            confidence_floor_breach=True,
        )
    )
    assert second.mode == RiskMode.CLOSE_ONLY
    assert second.trigger_layer == "FULL_CRISIS"
    assert second.metadata["crisis_weight_cap"] == 0.70
    assert second.metadata["crisis_persistence_ticks"] == 2


def test_full_crisis_timer_auto_reverts_if_not_revalidated():
    overseer = RiskOverseerStateMachine(
        thresholds=RiskOverseerThresholds(
            crisis_hysteresis_ticks=1,
            crisis_max_duration_ticks=2,
        ),
    )

    entered = overseer.evaluate(
        _snapshot(
            realized_vol_break=True,
            liquidity_deterioration=True,
            confidence_floor_breach=True,
        )
    )
    assert entered.mode == RiskMode.CLOSE_ONLY
    assert entered.trigger_layer == "FULL_CRISIS"

    holding = overseer.evaluate(_snapshot())
    assert holding.mode == RiskMode.CLOSE_ONLY
    assert holding.trigger_layer == "FULL_CRISIS"
    assert holding.trigger_reason == "full_crisis_holding_for_revalidation"

    expired = overseer.evaluate(_snapshot(timestamp=datetime(2026, 5, 4, 9, 17, tzinfo=UTC)))
    assert expired.mode == RiskMode.REDUCE_ONLY
    assert expired.trigger_layer == "RECOVERY"
    assert expired.trigger_reason == "full_crisis_expired_without_revalidation"
    assert expired.event_id is not None


def test_divergence_protocol_staged_re_risking_reaches_100_percent():
    overseer = RiskOverseerStateMachine(
        thresholds=RiskOverseerThresholds(divergence_re_risk_steps=(0.25, 0.5, 0.75, 1.0)),
    )

    hold = overseer.evaluate(_snapshot(agent_divergence=True))
    assert hold.mode == RiskMode.REDUCE_ONLY
    assert hold.trigger_layer == "AGENT_DIVERGENCE"
    assert hold.metadata["risk_budget_fraction"] == 0.25

    step_50 = overseer.evaluate(_snapshot(alignment_recovered=True))
    assert step_50.trigger_reason == "staged_re_risking"
    assert step_50.metadata["risk_budget_fraction"] == 0.5

    step_75 = overseer.evaluate(_snapshot(alignment_recovered=True))
    assert step_75.trigger_reason == "staged_re_risking"
    assert step_75.metadata["risk_budget_fraction"] == 0.75

    step_100 = overseer.evaluate(_snapshot(alignment_recovered=True))
    assert step_100.trigger_reason == "staged_re_risking_complete"
    assert step_100.metadata["risk_budget_fraction"] == 1.0


def test_slow_crash_applies_budget_cap_without_forcing_reduce_only():
    overseer = RiskOverseerStateMachine(
        thresholds=RiskOverseerThresholds(slow_crash_risk_budget_fraction=0.40),
    )

    decision = overseer.evaluate(_snapshot(slow_crash=True))
    assert decision.mode == RiskMode.NORMAL
    assert decision.trigger_layer == "SLOW_CRASH"
    assert decision.metadata["risk_budget_fraction"] == 0.40
    assert decision.block_new_orders is False


def test_negative_sentiment_and_mismatch_trigger_protective_caps():
    overseer = RiskOverseerStateMachine(
        thresholds=RiskOverseerThresholds(
            negative_sentiment_z_threshold=-2.0,
            sentiment_price_mismatch_return_threshold=-0.015,
            sentiment_protective_risk_budget_fraction=0.45,
        ),
    )

    negative = overseer.evaluate(_snapshot(sentiment_z_t=-2.3))
    assert negative.mode == RiskMode.NORMAL
    assert negative.trigger_layer == "SENTIMENT"
    assert negative.trigger_reason == "extreme_negative_sentiment"
    assert negative.metadata["risk_budget_fraction"] == 0.45

    mismatch = overseer.evaluate(_snapshot(sentiment_z_t=0.6, price_return=-0.02))
    assert mismatch.mode == RiskMode.NORMAL
    assert mismatch.trigger_layer == "SENTIMENT"
    assert mismatch.trigger_reason == "sentiment_price_mismatch"
    assert mismatch.metadata["risk_budget_fraction"] == 0.45


def test_ood_is_staged_and_resets_after_clear():
    overseer = RiskOverseerStateMachine(
        thresholds=RiskOverseerThresholds(
            ood_stage_1_persistence=1,
            ood_stage_2_persistence=2,
            ood_stage_3_persistence=3,
            ood_stage_1_risk_budget_fraction=0.60,
            ood_stage_2_risk_budget_fraction=0.30,
            ood_stage_3_risk_budget_fraction=0.00,
        ),
    )

    stage_1 = overseer.evaluate(_snapshot(ood_flag=True))
    assert stage_1.mode == RiskMode.NORMAL
    assert stage_1.trigger_reason == "ood_stage_1"
    assert stage_1.metadata["ood_stage"] == 1
    assert stage_1.metadata["risk_budget_fraction"] == 0.60

    stage_2 = overseer.evaluate(_snapshot(ood_flag=True))
    assert stage_2.trigger_reason == "ood_stage_2"
    assert stage_2.metadata["ood_stage"] == 2
    assert stage_2.metadata["risk_budget_fraction"] == 0.30

    stage_3 = overseer.evaluate(_snapshot(ood_flag=True))
    assert stage_3.trigger_reason == "ood_stage_3"
    assert stage_3.metadata["ood_stage"] == 3
    assert stage_3.metadata["risk_budget_fraction"] == 0.00

    cleared = overseer.evaluate(_snapshot(ood_flag=False))
    assert cleared.trigger_layer == "NONE"
    assert cleared.trigger_reason == "no_breach"
    assert cleared.metadata["ood_persistence_ticks"] == 0

    stage_1_again = overseer.evaluate(_snapshot(ood_flag=True))
    assert stage_1_again.trigger_reason == "ood_stage_1"


def test_ood_hard_limit_breach_immediately_kill_switches():
    overseer = RiskOverseerStateMachine()
    decision = overseer.evaluate(
        _snapshot(
            timestamp=datetime(2026, 5, 4, 9, 15, tzinfo=UTC) + timedelta(minutes=1),
            ood_flag=True,
            hard_limit_breach=True,
        )
    )
    assert decision.mode == RiskMode.KILL_SWITCH
    assert decision.trigger_layer == "OOD_HARD_LIMIT"
    assert decision.trigger_reason == "ood_with_hard_limit_breach"
