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
from src.agents.strategic.policies.base import PolicyTrainResult, TeacherPolicyFoundation


def _atanh(x: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(x, -0.999_999, 0.999_999)
    return 0.5 * torch.log((1.0 + clipped) / (1.0 - clipped))


class _PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, int]):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dims[1], action_dim)
        self.value = nn.Linear(hidden_dims[1], 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _dist_value(self, state: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        hidden = self.backbone(state)
        mean = self.mu(hidden)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value(hidden)
        return dist, value

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._dist_value(state)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, log_prob.sum(dim=-1, keepdim=True), value

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._dist_value(state)
        raw_action = _atanh(action)
        log_prob = dist.log_prob(raw_action) - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob.sum(dim=-1, keepdim=True), entropy, value

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(state)
        return torch.tanh(self.mu(hidden))


class PPOPolicyFoundation(TeacherPolicyFoundation):
    def __init__(self) -> None:
        super().__init__(
            config=PolicyFoundationConfig(policy_id="phase3_ppo_teacher_v1", algorithm="PPO", action_space="continuous"),
            dependency_name="torch",
            default_hyperparameters={
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "n_steps": 1024,
                "batch_size": 128,
                "update_epochs": 8,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
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
        gamma = float(params["gamma"])
        gae_lambda = float(params["gae_lambda"])
        n_steps = int(params["n_steps"])
        batch_size = int(params["batch_size"])
        update_epochs = int(params["update_epochs"])
        clip_range = float(params["clip_range"])
        ent_coef = float(params["ent_coef"])
        vf_coef = float(params["vf_coef"])
        max_grad_norm = float(params["max_grad_norm"])
        learning_rate = float(params["learning_rate"])
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

        model = _PPOActorCritic(state_dim, action_dim, hidden_dims).to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        obs, _ = env.reset()
        total_collected_steps = 0
        episodes = 0
        episode_reward = 0.0
        episode_rewards: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        while total_collected_steps < train_steps:
            rollout_states: list[np.ndarray] = []
            rollout_actions: list[np.ndarray] = []
            rollout_log_probs: list[float] = []
            rollout_rewards: list[float] = []
            rollout_dones: list[float] = []
            rollout_values: list[float] = []

            for _ in range(n_steps):
                state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device_obj).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_prob_tensor, value_tensor = model.sample(state_tensor)
                action_np = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                rollout_states.append(np.asarray(obs, dtype=np.float32))
                rollout_actions.append(np.asarray(action_np, dtype=np.float32))
                rollout_log_probs.append(float(log_prob_tensor.squeeze(0).cpu().item()))
                rollout_values.append(float(value_tensor.squeeze(0).cpu().item()))

                step_result = env.step(action_np)
                if len(step_result) == 5:
                    next_obs, reward, done, _, _ = step_result
                else:
                    next_obs, reward, done, _ = step_result
                rollout_rewards.append(float(reward))
                rollout_dones.append(float(done))

                obs = next_obs
                total_collected_steps += 1
                episode_reward += float(reward)
                if done:
                    episodes += 1
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    obs, _ = env.reset()
                if total_collected_steps >= train_steps:
                    break

            if not rollout_rewards:
                break

            with torch.no_grad():
                next_value = model.sample(torch.as_tensor(obs, dtype=torch.float32, device=device_obj).unsqueeze(0))[2]
                next_value_float = float(next_value.squeeze(0).cpu().item())

            rewards_np = np.asarray(rollout_rewards, dtype=np.float32)
            dones_np = np.asarray(rollout_dones, dtype=np.float32)
            values_np = np.asarray(rollout_values, dtype=np.float32)
            advantages = np.zeros_like(rewards_np, dtype=np.float32)
            last_advantage = 0.0
            for idx in reversed(range(len(rewards_np))):
                if idx == len(rewards_np) - 1:
                    next_non_terminal = 1.0 - dones_np[idx]
                    next_val = next_value_float
                else:
                    next_non_terminal = 1.0 - dones_np[idx]
                    next_val = values_np[idx + 1]
                delta = rewards_np[idx] + gamma * next_val * next_non_terminal - values_np[idx]
                last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
                advantages[idx] = last_advantage
            returns = advantages + values_np

            states_tensor = torch.as_tensor(np.asarray(rollout_states), dtype=torch.float32, device=device_obj)
            actions_tensor = torch.as_tensor(np.asarray(rollout_actions), dtype=torch.float32, device=device_obj)
            old_log_probs_tensor = torch.as_tensor(np.asarray(rollout_log_probs), dtype=torch.float32, device=device_obj).unsqueeze(-1)
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device_obj).unsqueeze(-1)
            advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device_obj).unsqueeze(-1)
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)

            sample_count = states_tensor.shape[0]
            mini_batch = max(1, min(batch_size, sample_count))
            indices = np.arange(sample_count)
            for _ in range(update_epochs):
                np.random.shuffle(indices)
                for start_idx in range(0, sample_count, mini_batch):
                    mb_idx = indices[start_idx : start_idx + mini_batch]
                    batch_state = states_tensor[mb_idx]
                    batch_action = actions_tensor[mb_idx]
                    batch_old_log_prob = old_log_probs_tensor[mb_idx]
                    batch_return = returns_tensor[mb_idx]
                    batch_advantage = advantages_tensor[mb_idx]

                    new_log_prob, entropy, value = model.evaluate_actions(batch_state, batch_action)
                    ratio = torch.exp(new_log_prob - batch_old_log_prob)
                    unclipped = ratio * batch_advantage
                    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * batch_advantage
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(value, batch_return)
                    entropy_bonus = entropy.mean()
                    loss = policy_loss + (vf_coef * value_loss) - (ent_coef * entropy_bonus)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                    policy_losses.append(float(policy_loss.detach().cpu().item()))
                    value_losses.append(float(value_loss.detach().cpu().item()))
                    entropies.append(float(entropy_bonus.detach().cpu().item()))

        eval_summary = self.evaluate_policy(
            observations=observations,
            prices=prices,
            reward_name=reward_name,
            reward_config=reward_config,
            action_fn=lambda state: model.deterministic_action(state),
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
            "policy_loss_mean": float(np.mean(policy_losses[-100:])) if policy_losses else 0.0,
            "value_loss_mean": float(np.mean(value_losses[-100:])) if value_losses else 0.0,
            "entropy_mean": float(np.mean(entropies[-100:])) if entropies else 0.0,
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
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

        train_end = datetime.now(UTC)
        return PolicyTrainResult(
            policy_id=self.policy_name,
            algorithm=self.config.algorithm,
            checkpoint_path=checkpoint_path,
            episodes=max(1, episodes),
            total_steps=total_collected_steps,
            final_reward=float(metrics["episode_reward"]),
            duration_seconds=max(0.0, (train_end - train_start).total_seconds()),
            metrics=metrics,
            params={**params, "seed": seed, "device": str(device_obj)},
            train_start=train_start,
            train_end=train_end,
            reward_name=reward_name,
            notes="Offline PPO training run completed on historical observations.",
        )
