from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.strategic import ObservationAssembler, StrategicTradingEnv
from src.agents.strategic.config import WalkForwardConfig
from src.agents.strategic.policies import PPOPolicyFoundation, SACPolicyFoundation, TD3PolicyFoundation
from src.agents.strategic.splits import build_planned_training_run
from src.db.strategic_recorder import StrategicRecorder


def _sample_phase2_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = datetime(2026, 3, 24, tzinfo=UTC)
    timestamps = [base + timedelta(hours=i) for i in range(4)]
    technical = pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS"] * 4,
            "timestamp": timestamps,
            "price_forecast": [100.2, 100.7, 100.5, 101.0],
            "direction": ["up", "up", "neutral", "up"],
            "var_95": [-0.02, -0.021, -0.019, -0.018],
            "es_95": [-0.03, -0.031, -0.028, -0.027],
            "model_id": ["tech-v1"] * 4,
        }
    )
    regime = pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS"] * 4,
            "timestamp": timestamps,
            "regime_state": ["Bull", "Bull", "Sideways", "Bull"],
            "transition_probability": [0.2, 0.3, 0.4, 0.25],
            "model_id": ["regime-v1"] * 4,
        }
    )
    sentiment = pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS"] * 4,
            "timestamp": timestamps,
            "sentiment_score": [0.1, 0.2, -0.1, 0.3],
            "z_t": [0.05, 0.10, -0.04, 0.12],
            "model_id": ["sent-v1"] * 4,
        }
    )
    consensus = pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS"] * 4,
            "timestamp": timestamps,
            "final_direction": ["up", "up", "neutral", "up"],
            "final_confidence": [0.72, 0.74, 0.60, 0.80],
            "crisis_mode": [False, False, True, False],
            "model_id": ["cons-v1"] * 4,
        }
    )
    return technical, regime, sentiment, consensus


def main() -> None:
    technical, regime, sentiment, consensus = _sample_phase2_frames()
    assembler = ObservationAssembler()
    observations = assembler.assemble_from_frames(
        technical=technical,
        regime=regime,
        sentiment=sentiment,
        consensus=consensus,
    )

    env = StrategicTradingEnv(
        observations=observations,
        prices=[100.0, 101.0, 100.5, 102.0],
    )
    env.reset()
    env.step(0.25)
    env.step(0.50)

    engine = create_engine("sqlite:///:memory:")
    recorder = StrategicRecorder(engine=engine, session_factory=sessionmaker(bind=engine))
    for observation in observations:
        recorder.save_observation(observation)
    for reward_log in env.reward_logs:
        recorder.save_reward_log(reward_log)

    config = WalkForwardConfig()
    policies = [SACPolicyFoundation(), PPOPolicyFoundation(), TD3PolicyFoundation()]
    for policy in policies:
        recorder.save_rl_policy(policy.registry_entry())
        recorder.save_training_run(build_planned_training_run(policy.policy_name, config, reward_name="step_return"))

    print(
        json.dumps(
            {
                "observations_materialized": len(observations),
                "reward_logs_materialized": len(env.reward_logs),
                "policies_registered": [policy.policy_name for policy in policies],
                "training_executed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
