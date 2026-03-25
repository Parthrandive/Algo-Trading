from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.agents.strategic.action_space import export_week2_action_space
from src.agents.strategic.placeholder_teacher import generate_placeholder_teacher_actions
from src.agents.strategic.schemas import (
    ActionType,
    LoopType,
    PolicyType,
    StrategicObservation,
    StrategicToExecutiveContract,
    Week2ActionSpaceRecord,
)


def _obs(
    *,
    symbol: str = "RELIANCE.NS",
    crisis_mode: bool = False,
    agent_divergence: bool = False,
    consensus_direction: str = "HOLD",
    consensus_confidence: float = 0.3,
) -> StrategicObservation:
    return StrategicObservation(
        timestamp=datetime(2026, 3, 25, 0, 0, tzinfo=timezone.utc),
        symbol=symbol,
        snapshot_id=f"{symbol}:snap",
        technical_direction="neutral",
        technical_confidence=0.25,
        price_forecast=100.0,
        var_95=1.0,
        es_95=1.2,
        regime_state="Normal",
        regime_transition_prob=0.2,
        sentiment_score=0.1,
        sentiment_z_t=0.2,
        consensus_direction=consensus_direction,
        consensus_confidence=consensus_confidence,
        crisis_mode=crisis_mode,
        agent_divergence=agent_divergence,
        quality_status="pass",
    )


def test_contract_blocks_teacher_fast_loop():
    with pytest.raises(ValueError, match="Teacher policies are blocked from Fast Loop"):
        StrategicToExecutiveContract(
            timestamp=datetime(2026, 3, 25, 0, 0, tzinfo=timezone.utc),
            symbol="RELIANCE.NS",
            policy_id="teacher_v1",
            policy_type=PolicyType.TEACHER,
            loop_type=LoopType.FAST,
            action=ActionType.HOLD,
            action_size=0.0,
            confidence=0.5,
            snapshot_id="snap",
            observation_schema_version="1.0",
        )


def test_week2_record_blocks_teacher_fast_loop():
    with pytest.raises(ValueError, match="cannot contain teacher actions for Fast Loop"):
        Week2ActionSpaceRecord(
            timestamp=datetime(2026, 3, 25, 0, 0, tzinfo=timezone.utc),
            symbol="RELIANCE.NS",
            policy_id="teacher_v1",
            policy_type="teacher",
            loop_type="fast",
            action=ActionType.HOLD,
            action_size=0.0,
            confidence=0.5,
            snapshot_id="snap",
            observation_schema_version="1.0",
        )


def test_placeholder_teacher_reduce_on_crisis():
    observation = _obs(crisis_mode=True, consensus_direction="BUY", consensus_confidence=0.9)
    actions = generate_placeholder_teacher_actions([observation], policy_id="teacher_placeholder_v0")
    assert len(actions) == 1
    assert actions[0].action == ActionType.REDUCE
    assert actions[0].loop_type == LoopType.SLOW
    assert actions[0].policy_type == PolicyType.TEACHER


def test_placeholder_teacher_buy_on_aligned_signal():
    observation = _obs(consensus_direction="BUY", consensus_confidence=0.8)
    actions = generate_placeholder_teacher_actions([observation], policy_id="teacher_placeholder_v0")
    assert actions[0].action == ActionType.BUY
    assert actions[0].action_size > 0.0


def test_action_export_jsonl(tmp_path):
    observation = _obs(consensus_direction="HOLD", consensus_confidence=0.4)
    actions = generate_placeholder_teacher_actions([observation], policy_id="teacher_placeholder_v0")
    export_path = tmp_path / "week2_action_space_export.jsonl"
    export_week2_action_space(export_path, actions)

    lines = export_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["policy_type"] == "teacher"
    assert payload["loop_type"] == "slow"
    assert payload["export_schema_version"] == "week2_action_space_v1"
