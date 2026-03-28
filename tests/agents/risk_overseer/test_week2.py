from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.risk_overseer import (
    CrisisRiskSnapshot,
    CrisisState,
    OODRiskSnapshot,
    PortfolioRiskSnapshot,
    RecoveryRequest,
    RiskEvaluationInput,
    RiskOverseerConfig,
    RiskOverseerService,
    SentimentRiskSnapshot,
)
from src.agents.strategic.ensemble import MaxEntropyEnsemble
from src.agents.strategic.schemas import (
    ActionType,
    LoopType,
    PolicyAction,
    PolicyType,
    RiskMode,
    StrategicObservation,
)


def _ts(offset: int = 0) -> datetime:
    return datetime(2026, 4, 28, 9, 15, tzinfo=UTC) + timedelta(minutes=offset)


def _observation() -> StrategicObservation:
    return StrategicObservation(
        timestamp=_ts(),
        symbol="RELIANCE.NS",
        snapshot_id="obs-1",
        technical_direction="up",
        technical_confidence=0.7,
        price_forecast=100.0,
        var_95=1.0,
        es_95=1.2,
        regime_state="Bull",
        regime_transition_prob=0.2,
        sentiment_score=0.1,
        sentiment_z_t=0.2,
        consensus_direction="BUY",
        consensus_confidence=0.8,
    )


def _teacher_actions() -> tuple[PolicyAction, ...]:
    return (
        PolicyAction(
            policy_id="sac",
            policy_type=PolicyType.TEACHER,
            loop_type=LoopType.SLOW,
            action=ActionType.BUY,
            action_size=0.9,
            confidence=0.95,
        ),
        PolicyAction(
            policy_id="ppo",
            policy_type=PolicyType.TEACHER,
            loop_type=LoopType.SLOW,
            action=ActionType.BUY,
            action_size=0.6,
            confidence=0.50,
        ),
        PolicyAction(
            policy_id="td3",
            policy_type=PolicyType.TEACHER,
            loop_type=LoopType.SLOW,
            action=ActionType.HOLD,
            action_size=0.1,
            confidence=0.25,
        ),
    )


def test_full_crisis_requires_hysteresis_before_activation():
    service = RiskOverseerService(RiskOverseerConfig(crisis_hysteresis_ticks=2))
    crisis = CrisisRiskSnapshot(realized_vol=2.1, baseline_vol=1.0, liquidity_score=0.2, agent_confidence=0.2)

    first = service.evaluate(RiskEvaluationInput(timestamp=_ts(0), crisis=crisis))
    second = service.evaluate(RiskEvaluationInput(timestamp=_ts(1), crisis=crisis))

    assert first.crisis_state == CrisisState.NORMAL
    assert second.crisis_state == CrisisState.FULL_CRISIS
    assert second.mode == RiskMode.CLOSE_ONLY
    assert second.exposure_cap == 0.15
    assert any(event.trigger_code.value == "crisis_entry" for event in second.trigger_events)


def test_divergence_protocol_enforces_neutral_hold_then_staged_rerisk():
    service = RiskOverseerService(RiskOverseerConfig(divergence_alignment_signals_required=1))

    divergence = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(0),
            crisis=CrisisRiskSnapshot(major_agent_disagreement_count=2),
        )
    )
    assert divergence.crisis_state == CrisisState.AGENT_DIVERGENCE
    assert divergence.neutral_hold_active is True
    assert divergence.exposure_cap == 0.25
    assert ActionType.BUY not in divergence.permitted_actions

    step_50 = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(1),
            recovery=RecoveryRequest(confirmed_alignment=True),
        )
    )
    assert step_50.rerisk_budget_fraction == 0.50
    assert step_50.exposure_cap == 0.50

    step_75 = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(2),
            recovery=RecoveryRequest(confirmed_alignment=True),
        )
    )
    assert step_75.rerisk_budget_fraction == 0.75

    step_100 = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(3),
            recovery=RecoveryRequest(confirmed_alignment=True),
        )
    )
    assert step_100.rerisk_budget_fraction == 1.00
    assert step_100.neutral_hold_active is False
    assert ActionType.BUY in step_100.permitted_actions


def test_slow_crash_and_feed_freeze_apply_staged_derisking():
    service = RiskOverseerService()

    slow = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(0),
            crisis=CrisisRiskSnapshot(drawdown_velocity=0.05),
        )
    )
    freeze = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(1),
            crisis=CrisisRiskSnapshot(feed_staleness_seconds=120),
        )
    )

    assert slow.crisis_state == CrisisState.SLOW_CRASH
    assert slow.mode == RiskMode.REDUCE_ONLY
    assert slow.exposure_cap == 0.50
    assert freeze.crisis_state == CrisisState.FEED_FREEZE
    assert freeze.mode == RiskMode.REDUCE_ONLY
    assert freeze.exposure_cap == 0.25


def test_negative_sentiment_and_mismatch_raise_protective_overlay():
    service = RiskOverseerService()

    negative = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(0),
            sentiment=SentimentRiskSnapshot(z_t=-3.0, sentiment_score=-0.4, price_return=-0.01),
        )
    )
    mismatch = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(1),
            sentiment=SentimentRiskSnapshot(z_t=0.2, sentiment_score=0.8, price_return=-0.03),
        )
    )

    assert negative.crisis_state == CrisisState.NEGATIVE_SENTIMENT
    assert negative.exposure_cap == 0.40
    assert negative.hedge_bias >= 0.50
    assert mismatch.crisis_state == CrisisState.NEGATIVE_SENTIMENT
    assert mismatch.mode == RiskMode.REDUCE_ONLY


def test_ood_alien_with_hard_limit_escalates_to_kill_switch():
    service = RiskOverseerService()

    assessment = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(0),
            portfolio=PortfolioRiskSnapshot(daily_loss=0.05),
            ood=OODRiskSnapshot(alien=True, provenance_reliability=0.2),
        )
    )

    assert assessment.crisis_state == CrisisState.OOD_ALIEN
    assert assessment.mode == RiskMode.KILL_SWITCH
    assert assessment.exposure_cap == 0.25
    assert any(event.trigger_code.value == "ood_hard_limit_breach" for event in assessment.trigger_events)


def test_ensemble_caps_dominant_policy_weight_in_full_crisis():
    service = RiskOverseerService(RiskOverseerConfig(crisis_hysteresis_ticks=1))
    assessment = service.evaluate(
        RiskEvaluationInput(
            timestamp=_ts(0),
            crisis=CrisisRiskSnapshot(realized_vol=2.0, baseline_vol=1.0, liquidity_score=0.2, agent_confidence=0.2),
        )
    )
    decision = MaxEntropyEnsemble().decide(
        _observation(),
        _teacher_actions(),
        risk_assessment=assessment,
    )

    dominant_weight = max(weight.weight for weight in decision.policy_weights)
    assert dominant_weight <= assessment.crisis_weight_cap + 1e-9
    assert decision.action_size <= assessment.exposure_cap + 1e-9
    assert decision.risk_mode == RiskMode.CLOSE_ONLY
