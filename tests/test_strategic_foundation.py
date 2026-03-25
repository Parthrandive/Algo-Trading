from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.agents.strategic import ObservationAssembler, StrategicTradingEnv, sharpe_ratio
from src.agents.strategic.evaluation import evaluate_equal_weight_ensemble
from src.agents.strategic.model_cards import build_teacher_model_card
from src.agents.strategic.policies import PPOPolicyFoundation, SACPolicyFoundation, TD3PolicyFoundation
from src.agents.strategic.splits import build_planned_training_run, build_walk_forward_mask
from src.db.models import RLPolicyDB, RLTrainingRunDB, RewardLogDB, StrategicObservationDB
from src.db.phase3_recorder import Phase3Recorder as StrategicRecorder


def _sample_frames():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [base + timedelta(hours=i) for i in range(4)]
    technical = pd.DataFrame(
        {
            "symbol": ["TEST.NS"] * 4,
            "timestamp": timestamps,
            "price_forecast": [100.0, 101.0, 102.0, 103.0],
            "direction": ["up", "down", "neutral", "up"],
            "var_95": [-0.02, -0.03, -0.01, -0.02],
            "es_95": [-0.03, -0.04, -0.02, -0.03],
            "model_id": ["tech-v1"] * 4,
        }
    )
    regime = pd.DataFrame(
        {
            "symbol": ["TEST.NS"] * 4,
            "timestamp": timestamps,
            "regime_state": ["Bull", "Bear", "Sideways", "Bull"],
            "transition_probability": [0.1, 0.6, 0.3, 0.2],
            "model_id": ["reg-v1"] * 4,
        }
    )
    sentiment = pd.DataFrame(
        {
            "symbol": ["TEST.NS"] * 4,
            "timestamp": timestamps,
            "sentiment_score": [0.4, -0.2, 0.1, 0.3],
            "z_t": [0.2, -0.1, 0.0, 0.15],
            "model_id": ["sent-v1"] * 4,
        }
    )
    consensus = pd.DataFrame(
        {
            "symbol": ["TEST.NS"] * 4,
            "timestamp": timestamps,
            "final_direction": ["up", "down", "neutral", "up"],
            "final_confidence": [0.8, 0.6, 0.4, 0.75],
            "crisis_mode": [False, True, False, False],
            "model_id": ["cons-v1"] * 4,
        }
    )
    portfolio = pd.DataFrame(
        {
            "symbol": ["TEST.NS"] * 4,
            "timestamp": timestamps,
            "current_position": [0.0, 0.25, -0.1, 0.4],
            "unrealized_pnl": [0.0, 100.0, -50.0, 200.0],
        }
    )
    return technical, regime, sentiment, consensus, portfolio


def _walk_forward_config():
    from src.agents.strategic.config import WalkForwardConfig

    return WalkForwardConfig()


def test_observation_assembler_creates_fixed_length_vectors():
    technical, regime, sentiment, consensus, portfolio = _sample_frames()
    observations = ObservationAssembler().assemble_from_frames(
        technical=technical,
        regime=regime,
        sentiment=sentiment,
        consensus=consensus,
        portfolio=portfolio,
    )

    assert len(observations) == 4
    assert len(observations[0].observation_vector) == 18
    assert observations[1].observation_vector[1] == -1.0
    assert observations[0].observation_vector[4] == 1.0


def test_strategic_env_logs_rewards_without_training():
    technical, regime, sentiment, consensus, portfolio = _sample_frames()
    observations = ObservationAssembler().assemble_from_frames(
        technical=technical,
        regime=regime,
        sentiment=sentiment,
        consensus=consensus,
        portfolio=portfolio,
    )
    env = StrategicTradingEnv(observations=observations, prices=[100.0, 101.0, 99.0, 103.0])

    env.reset()
    env.step(0.5)
    env.step(-0.25)

    assert len(env.reward_logs) == 2
    assert env.reward_logs[0].reward_name == "step_return"
    assert env.reward_logs[0].episode_id


def test_recorder_bootstraps_strategic_tables_in_sqlite():
    engine = create_engine("sqlite:///:memory:")
    recorder = StrategicRecorder(engine=engine, session_factory=sessionmaker(bind=engine))
    technical, regime, sentiment, consensus, portfolio = _sample_frames()
    observations = ObservationAssembler().assemble_from_frames(
        technical=technical,
        regime=regime,
        sentiment=sentiment,
        consensus=consensus,
        portfolio=portfolio,
    )
    recorder.save_observation(observations[0])
    recorder.save_reward_log(
        {
            "symbol": observations[0].symbol,
            "timestamp": observations[0].timestamp,
            "episode_id": "ep-1",
            "reward_name": "step_return",
            "reward_value": 0.01,
        }
    )
    recorder.save_rl_policy(SACPolicyFoundation().registry_entry())
    recorder.save_training_run(
        build_planned_training_run("phase3_sac_teacher_v1", _walk_forward_config(), reward_name="step_return")
    )

    with recorder.Session() as session:
        assert session.execute(select(StrategicObservationDB)).scalar_one() is not None
        assert session.execute(select(RewardLogDB)).scalar_one() is not None
        assert session.execute(select(RLPolicyDB)).scalar_one() is not None
        assert session.execute(select(RLTrainingRunDB)).scalar_one() is not None


def test_policy_foundations_are_offline_teacher_only():
    policies = [SACPolicyFoundation(), PPOPolicyFoundation(), TD3PolicyFoundation()]
    ids = [policy.registry_entry().policy_id for policy in policies]
    assert ids == [
        "phase3_sac_teacher_v1",
        "phase3_ppo_teacher_v1",
        "phase3_td3_teacher_v1",
    ]
    assert all(policy.registry_entry().offline_only for policy in policies)
    card = build_teacher_model_card(policies[0].registry_entry(), policies[0].planned_run())
    assert card["teacher_policy"] is True
    assert card["promotion_gate"]["distillation_required"] is True


def test_walk_forward_mask_and_ensemble_eval_foundation():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2022-01-01T00:00:00Z", "2024-06-01T00:00:00Z", "2025-06-01T00:00:00Z"],
                utc=True,
            )
        }
    )
    masked = build_walk_forward_mask(frame, _walk_forward_config())
    assert masked["split"].tolist() == ["train", "validation", "test"]

    ensemble = evaluate_equal_weight_ensemble(
        {"sac": 0.4, "ppo": 0.2, "td3": -0.1},
        {"sac": 0.7, "ppo": 0.6, "td3": 0.8},
    )
    assert round(ensemble.equal_weight_action, 6) == round((0.4 + 0.2 - 0.1) / 3.0, 6)
    assert 0.0 <= ensemble.mean_confidence <= 1.0


def test_reward_library_exposes_sharpe_foundation():
    ratio = sharpe_ratio([0.01, 0.02, -0.01, 0.015])
    assert ratio > 0.0
