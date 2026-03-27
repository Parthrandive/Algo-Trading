from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.agents.strategic.environment import StrategicTradingEnv
from src.agents.strategic.policies import PPOPolicyFoundation, SACPolicyFoundation, TD3PolicyFoundation
from src.agents.strategic.schemas import StrategicObservation


def _build_synthetic_dataset(length: int = 96) -> tuple[list[StrategicObservation], list[float]]:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    prices = []
    observations: list[StrategicObservation] = []
    for idx in range(length):
        price = 100.0 + (0.12 * idx) + (1.4 * np.sin(idx / 6.0))
        prices.append(float(price))
    for idx in range(length):
        direction = "BUY" if idx == 0 or prices[idx] >= prices[idx - 1] else "SELL"
        observations.append(
            StrategicObservation(
                timestamp=base + timedelta(hours=idx),
                symbol="RELIANCE.NS",
                snapshot_id=f"snap-{idx}",
                technical_direction="up" if direction == "BUY" else "down",
                technical_confidence=0.7,
                price_forecast=prices[idx],
                var_95=-0.03,
                es_95=-0.04,
                regime_state="Bull",
                regime_transition_prob=0.15,
                sentiment_score=0.2,
                sentiment_z_t=0.1,
                consensus_direction=direction,
                consensus_confidence=0.65,
                crisis_mode=False,
                agent_divergence=False,
                quality_status="pass",
            )
        )
    return observations, prices


def test_environment_composite_reward_logs_components():
    observations, prices = _build_synthetic_dataset(length=8)
    observations[1] = observations[1].model_copy(update={"crisis_mode": True, "agent_divergence": True})
    env = StrategicTradingEnv(observations=observations, prices=prices, reward_name="ra_drl_composite")

    env.reset()
    env.step(np.asarray([1.0], dtype=np.float32))
    assert len(env.reward_logs) == 1
    components = env.reward_logs[0].metadata.get("reward_components", {})
    assert "drawdown_penalty" in components
    assert "crisis_penalty" in components
    assert "divergence_penalty" in components


@pytest.mark.parametrize(
    "policy,extra_kwargs",
    [
        (
            SACPolicyFoundation(),
            {"hidden_dims": (32, 32), "batch_size": 32, "buffer_size": 2_000, "learning_starts": 50},
        ),
        (
            PPOPolicyFoundation(),
            {"hidden_dims": (32, 32), "n_steps": 64, "batch_size": 32, "update_epochs": 2},
        ),
        (
            TD3PolicyFoundation(),
            {"hidden_dims": (32, 32), "batch_size": 32, "buffer_size": 2_000, "learning_starts": 50},
        ),
    ],
)
def test_policy_trains_offline_and_writes_checkpoint(tmp_path, policy, extra_kwargs):
    observations, prices = _build_synthetic_dataset(length=96)
    result = policy.train_offline(
        observations=observations,
        prices=prices,
        total_timesteps=250,
        seed=7,
        reward_name="ra_drl_composite",
        output_dir=tmp_path,
        device="cpu",
        **extra_kwargs,
    )

    assert result.checkpoint_path.exists()
    assert result.total_steps > 0
    assert "sharpe" in result.metrics
    assert result.metrics["final_portfolio_value"] >= 0.0
