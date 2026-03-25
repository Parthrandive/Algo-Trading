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


def _mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        nn.ReLU(),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.ReLU(),
        nn.Linear(hidden_dims[1], output_dim),
    )


class _SACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, int]):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.backbone(state)
        mu = self.mu(latent)
        log_std = torch.clamp(self.log_std(latent), -5.0, 2.0)
        return mu, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mu)
        return action, log_prob, deterministic


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, int]):
        super().__init__()
        self.net = _mlp(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class SACPolicyFoundation(TeacherPolicyFoundation):
    def __init__(self) -> None:
        super().__init__(
            config=PolicyFoundationConfig(policy_id="phase3_sac_teacher_v1", algorithm="SAC", action_space="continuous"),
            dependency_name="torch",
            default_hyperparameters={
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "buffer_size": 150_000,
                "batch_size": 256,
                "tau": 0.005,
                "hidden_dims": (256, 256),
                "learning_starts": 1_000,
                "gradient_steps": 1,
                "alpha_init": 0.20,
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
        train_steps = int(max(100, total_timesteps))
        params = {**self.default_hyperparameters, **kwargs}
        batch_size = int(params["batch_size"])
        gamma = float(params["gamma"])
        tau = float(params["tau"])
        learning_rate = float(params["learning_rate"])
        buffer_size = int(params["buffer_size"])
        learning_starts = int(params["learning_starts"])
        gradient_steps = int(params["gradient_steps"])
        alpha_init = float(params["alpha_init"])
        hidden_dims = tuple(params["hidden_dims"])
        device_obj = self.resolve_device(device)

        env = self.build_env(
            observations=observations,
            prices=prices,
            reward_name=reward_name,
            reward_config=reward_config,
        )
        state_dim = len(observations[0].observation_vector)
        action_dim = 1

        actor = _SACActor(state_dim, action_dim, hidden_dims).to(device_obj)
        q1 = _QNetwork(state_dim, action_dim, hidden_dims).to(device_obj)
        q2 = _QNetwork(state_dim, action_dim, hidden_dims).to(device_obj)
        q1_target = _QNetwork(state_dim, action_dim, hidden_dims).to(device_obj)
        q2_target = _QNetwork(state_dim, action_dim, hidden_dims).to(device_obj)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        actor_opt = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        q1_opt = torch.optim.Adam(q1.parameters(), lr=learning_rate)
        q2_opt = torch.optim.Adam(q2.parameters(), lr=learning_rate)
        log_alpha = torch.tensor(np.log(max(alpha_init, 1e-4)), dtype=torch.float32, device=device_obj, requires_grad=True)
        alpha_opt = torch.optim.Adam([log_alpha], lr=learning_rate)
        target_entropy = -float(action_dim)

        replay = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=buffer_size)
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_rewards: list[float] = []
        episodes = 0
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []

        for step in range(train_steps):
            if step < learning_starts:
                action_np = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            else:
                state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device_obj).unsqueeze(0)
                with torch.no_grad():
                    sampled_action, _, _ = actor.sample(state_tensor)
                action_np = sampled_action.cpu().numpy().reshape(-1).astype(np.float32)

            step_result = env.step(action_np)
            if len(step_result) == 5:
                next_obs, reward, done, _, _ = step_result
            else:
                next_obs, reward, done, _ = step_result
            replay.add(obs, action_np, float(reward), next_obs, bool(done))

            obs = next_obs
            episode_reward += float(reward)
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                episodes += 1
                obs, _ = env.reset()

            if step < learning_starts or len(replay) < batch_size:
                continue

            for _ in range(gradient_steps):
                state, action, reward_batch, next_state, done_batch = replay.sample(batch_size, device_obj)
                with torch.no_grad():
                    next_action, next_log_prob, _ = actor.sample(next_state)
                    alpha_value = log_alpha.exp()
                    target_q = torch.min(
                        q1_target(next_state, next_action),
                        q2_target(next_state, next_action),
                    ) - alpha_value * next_log_prob
                    expected_q = reward_batch + (1.0 - done_batch) * gamma * target_q

                q1_loss = F.mse_loss(q1(state, action), expected_q)
                q2_loss = F.mse_loss(q2(state, action), expected_q)

                q1_opt.zero_grad()
                q1_loss.backward()
                q1_opt.step()

                q2_opt.zero_grad()
                q2_loss.backward()
                q2_opt.step()

                sampled_action, log_prob, _ = actor.sample(state)
                q_pi = torch.min(q1(state, sampled_action), q2(state, sampled_action))
                alpha_value = log_alpha.exp()
                actor_loss = (alpha_value * log_prob - q_pi).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                with torch.no_grad():
                    for target_param, param in zip(q1_target.parameters(), q1.parameters()):
                        target_param.data.mul_(1.0 - tau).add_(tau * param.data)
                    for target_param, param in zip(q2_target.parameters(), q2.parameters()):
                        target_param.data.mul_(1.0 - tau).add_(tau * param.data)

                critic_losses.append(float((q1_loss + q2_loss).detach().cpu().item() * 0.5))
                actor_losses.append(float(actor_loss.detach().cpu().item()))
                alpha_losses.append(float(alpha_loss.detach().cpu().item()))

        eval_summary = self.evaluate_policy(
            observations=observations,
            prices=prices,
            reward_name=reward_name,
            reward_config=reward_config,
            action_fn=lambda state: actor.sample(state)[2],
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
            "alpha_loss_mean": float(np.mean(alpha_losses[-100:])) if alpha_losses else 0.0,
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
                "q1_state_dict": q1.state_dict(),
                "q2_state_dict": q2.state_dict(),
                "q1_target_state_dict": q1_target.state_dict(),
                "q2_target_state_dict": q2_target.state_dict(),
                "log_alpha": float(log_alpha.detach().cpu().item()),
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
            notes="Offline SAC training run completed on historical observations.",
        )
