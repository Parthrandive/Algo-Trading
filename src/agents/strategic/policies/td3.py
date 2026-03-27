from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.agents.strategic.config import PolicyFoundationConfig
from src.agents.strategic.reward import CompositeRewardConfig
from src.agents.strategic.schemas import StrategicObservation
from src.agents.strategic.policies.base import PolicyTrainResult, ReplayBuffer, TeacherPolicyFoundation


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        nn.ReLU(),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.ReLU(),
        nn.Linear(hidden_dims[1], output_dim),
    )


class _TD3Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, int]):
        super().__init__()
        self.net = _build_mlp(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class _TD3Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, int]):
        super().__init__()
        self.q1 = _build_mlp(state_dim + action_dim, 1, hidden_dims)
        self.q2 = _build_mlp(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([state, action], dim=-1)
        return self.q1(inputs), self.q2(inputs)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))


class TD3PolicyFoundation(TeacherPolicyFoundation):
    def __init__(self) -> None:
        super().__init__(
            config=PolicyFoundationConfig(policy_id="phase3_td3_teacher_v1", algorithm="TD3", action_space="continuous"),
            dependency_name="torch",
            default_hyperparameters={
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "buffer_size": 150_000,
                "batch_size": 256,
                "tau": 0.005,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "noise_clip": 0.5,
                "exploration_noise": 0.1,
                "learning_starts": 1_000,
                "hidden_dims": (256, 256),
            },
        )

    def train_offline(
        self,
        *,
        observations: list[StrategicObservation],
        prices: list[float],
        total_timesteps: int,
        seed: int,
        reward_name: str,
        output_dir: Path,
        device: str | None = None,
        reward_config: CompositeRewardConfig | None = None,
        **kwargs: Any,
    ) -> PolicyTrainResult:
        self.ensure_dependency()
        self.seed_everything(seed)
        train_start = datetime.now(UTC)
        params = {**self.default_hyperparameters, **kwargs}
        train_steps = int(max(100, total_timesteps))
        learning_rate = float(params["learning_rate"])
        gamma = float(params["gamma"])
        batch_size = int(params["batch_size"])
        tau = float(params["tau"])
        policy_delay = int(params["policy_delay"])
        target_policy_noise = float(params["target_policy_noise"])
        noise_clip = float(params["noise_clip"])
        exploration_noise = float(params["exploration_noise"])
        learning_starts = int(params["learning_starts"])
        hidden_dims = tuple(params["hidden_dims"])
        buffer_size = int(params["buffer_size"])
        device_obj = self.resolve_device(device)

        env = self.build_env(
            observations=observations,
            prices=prices,
            reward_name=reward_name,
            reward_config=reward_config,
        )
        state_dim = len(observations[0].observation_vector)
        action_dim = 1

        actor = _TD3Actor(state_dim, action_dim, hidden_dims).to(device_obj)
        actor_target = _TD3Actor(state_dim, action_dim, hidden_dims).to(device_obj)
        actor_target.load_state_dict(actor.state_dict())
        critic = _TD3Critic(state_dim, action_dim, hidden_dims).to(device_obj)
        critic_target = _TD3Critic(state_dim, action_dim, hidden_dims).to(device_obj)
        critic_target.load_state_dict(critic.state_dict())

        actor_opt = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=learning_rate)
        replay = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=buffer_size)

        obs, _ = env.reset()
        episodes = 0
        episode_reward = 0.0
        episode_rewards: list[float] = []
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        update_step = 0

        for step in range(train_steps):
            if step < learning_starts:
                action_np = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            else:
                state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device_obj).unsqueeze(0)
                with torch.no_grad():
                    action_tensor = actor(state_tensor)
                noise = np.random.normal(0.0, exploration_noise, size=action_dim).astype(np.float32)
                action_np = np.clip(action_tensor.cpu().numpy().reshape(-1) + noise, -1.0, 1.0).astype(np.float32)

            step_result = env.step(action_np)
            if len(step_result) == 5:
                next_obs, reward, done, _, _ = step_result
            else:
                next_obs, reward, done, _ = step_result
            replay.add(obs, action_np, float(reward), next_obs, bool(done))

            obs = next_obs
            episode_reward += float(reward)
            if done:
                episodes += 1
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()

            if step < learning_starts or len(replay) < batch_size:
                continue

            update_step += 1
            state, action, reward_batch, next_state, done_batch = replay.sample(batch_size, device_obj)
            with torch.no_grad():
                noise = torch.randn_like(action) * target_policy_noise
                noise = torch.clamp(noise, -noise_clip, noise_clip)
                next_action = torch.clamp(actor_target(next_state) + noise, -1.0, 1.0)
                target_q1, target_q2 = critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                expected_q = reward_batch + (1.0 - done_batch) * gamma * target_q

            current_q1, current_q2 = critic(state, action)
            critic_loss = F.mse_loss(current_q1, expected_q) + F.mse_loss(current_q2, expected_q)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            critic_losses.append(float(critic_loss.detach().cpu().item()))

            if update_step % max(1, policy_delay) == 0:
                actor_loss = -critic.q1_value(state, actor(state)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                actor_losses.append(float(actor_loss.detach().cpu().item()))

                with torch.no_grad():
                    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                        target_param.data.mul_(1.0 - tau).add_(tau * param.data)
                    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                        target_param.data.mul_(1.0 - tau).add_(tau * param.data)

        eval_summary = self.evaluate_policy(
            observations=observations,
            prices=prices,
            reward_name=reward_name,
            reward_config=reward_config,
            action_fn=lambda state: actor(state),
        )
        metrics = {
            "episode_reward": float(eval_summary["episode_reward"]),
            "cumulative_return": float(eval_summary["cumulative_return"]),
            "mean_step_return": float(eval_summary["mean_step_return"]),
            "sharpe": float(eval_summary["sharpe"]),
            "sortino": float(eval_summary["sortino"]),
            "calmar": float(eval_summary["calmar"]),
            "max_drawdown": float(eval_summary["max_drawdown"]),
            "win_rate": float(eval_summary["win_rate"]),
            "ra_drl_objective": float(eval_summary["ra_drl_objective"]),
            "final_portfolio_value": float(eval_summary["final_portfolio_value"]),
            "train_episode_reward_mean": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
            "critic_loss_mean": float(np.mean(critic_losses[-100:])) if critic_losses else 0.0,
            "actor_loss_mean": float(np.mean(actor_losses[-100:])) if actor_losses else 0.0,
        }

        checkpoint_path = output_dir / self.policy_name / "checkpoint.pt"
        self.save_checkpoint(
            {
                "algorithm": self.config.algorithm,
                "policy_id": self.policy_name,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "seed": seed,
                "reward_name": reward_name,
                "hyperparameters": params,
                "metrics": metrics,
                "actor_state_dict": actor.state_dict(),
                "actor_target_state_dict": actor_target.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "critic_target_state_dict": critic_target.state_dict(),
            },
            checkpoint_path,
        )

        train_end = datetime.now(UTC)
        return PolicyTrainResult(
            policy_id=self.policy_name,
            algorithm=self.config.algorithm,
            checkpoint_path=checkpoint_path,
            episodes=max(1, episodes),
            total_steps=train_steps,
            final_reward=float(metrics["episode_reward"]),
            duration_seconds=max(0.0, (train_end - train_start).total_seconds()),
            metrics=metrics,
            params={**params, "seed": seed, "device": str(device_obj)},
            train_start=train_start,
            train_end=train_end,
            reward_name=reward_name,
            notes="Offline TD3 training run completed on historical observations.",
        )
