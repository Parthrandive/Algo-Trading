# Phase 3 Week 1: RL Foundation & Observation Space

**Dates**: March 30, 2026 – April 5, 2026
**Owner**: You (Technical Lead)
**Prerequisites**: Phase 2 Analyst Board models active and producing validated signals. Data pipelines fully operational.

> **AGENTS.md HARD RULE ALERT**:
> - The RL policies trained this week (SAC, PPO, TD3) are **Teacher policies**.
> - **Never** place these teacher policies in the Fast Loop execution path. They are computationally heavy and run offline only. Fast Loop execution (p99 ≤ 8ms) will use a distilled student model built in Week 2.

## Week 1 Objective
Establish the observation-to-reward pipeline. Train the three offline teacher policies (SAC, PPO, TD3) on historical walk-forward data using gymnasium-compatible environments, logging their checkpoints and trajectories to the database.

---

### Day 1 (Mon): Observation Space & Package Skeleton

**Goals:**
Create the `src/agents/strategic/` structure and an `Observation Assembler` that maps Phase 2 signals into an RL-ready `observations` DB table.

**Tasks:**
1. Create package skeleton: `src/agents/strategic/__init__.py`, `config.py`, `schemas.py`.
2. Define the Phase 3 observation space schema consuming Phase 2 outputs:
   - **Technical**: `price_forecast`, `direction`, `var_95`, `es_95`
   - **Regime**: `regime_state`, `transition_probability`
   - **Sentiment**: `sentiment_score`, `z_t`
   - **Consensus**: `final_direction`, `final_confidence`, `crisis_mode`
   - **Portfolio** (defaults for now): `current_position`, `unrealized_pnl`
3. Code the `Observation Assembler` (`src/agents/strategic/observation.py`) to safely join these into a fixed-length vector, handling missings.
4. Version the schema and save mappings to the `observations` table (ensure timestamp alignment).

---

### Day 2 (Tue): Reward Library & Environment Wrapper

**Goals:**
Build the RL `reward.py` functions and the primary `environment.py` wrapper, ensuring transaction costs and slippage are properly penalized.

**Tasks:**
1. Implement `src/agents/strategic/reward.py`:
   - Returns-based: Sharpe, Sortino, Calmar, standard step-return.
   - Risk-adjusted: RA-DRL composite penalizing volatility.
   - Kelly criterion variant.
2. Build the `StrategicTradingEnv(gym.Env)` in `src/agents/strategic/environment.py`:
   - Continuous/discrete action mappings (Buy/Sell/Hold sizing).
   - Simulates NSE brokerage fees + standard market impact.
   - Steps through the materialized `observations` table.
3. Write reward computations to the `reward_logs` tracking table.

---

### Day 3 (Wed): SAC Policy Implementation & Baseline

**Goals:**
Implement and train the Soft Actor-Critic (SAC) baseline policy.

**Tasks:**
1. Code `src/agents/strategic/policies/sac.py` using Stable-Baselines3 (or custom PyTorch implementation).
2. Set up walk-forward train/val splits (e.g., Train: 2019-2023, Test: 2023-2025).
3. Train the SAC policy offline on the historical environment.
4. Save the model checkpoint and register metadata in the `rl_policies` table.
5. Log the learning curves to `rl_training_runs`.

---

### Day 4 (Thu): PPO Policy Implementation

**Goals:**
Implement the Proximal Policy Optimization (PPO) policy to compare against SAC.

**Tasks:**
1. Code `src/agents/strategic/policies/ppo.py`.
2. Train PPO on the exact same walk-forward data splits.
3. Track advantage estimations (GAE) and clipping.
4. Save checkpoints and register in `rl_policies` and `rl_training_runs`.

---

### Day 5 (Fri): TD3 Policy & Sandbox Ensemble Eval

**Goals:**
Add the Twin Delayed DDPG (TD3) policy and evaluate all three side-by-side.

**Tasks:**
1. Code `src/agents/strategic/policies/td3.py`.
2. Train TD3 historically.
3. Build a simple evaluation sandbox (offline) to compute basic "equal-weight" ensemble metrics (preview for Week 2).
4. Register the policy and ensure `rl_training_runs` metrics are fully captured.

---

### Day 6 (Sat): Model Cards, Baselines & Ablations

**Goals:**
Validate the risk/return characteristics of the offline Teacher policies.

**Tasks:**
1. Generate complete Model Cards for SAC, PPO, and TD3. Document data bounds, hyperparameters (SB3 configs), and known limitations.
2. Run an ablation analysis (e.g., mask the `sentiment_score` or `regime_state` arrays in the gym environment) and log the drop in test-set Sharpe/Sortino.
3. Compare against Buy-and-Hold and Random-Action baselines.

---

### Day 7 (Sun): Review & Handoff to Week 2

**Goals:**
Integration gate to finalize the offline Teachers before they are distilled into a Fast Loop student model in Week 2.

**Tasks:**
1. End-to-end dry run: Data → Phase 2 DB → `Observation Assembler` → Gym Environment → SAC/PPO/TD3 Offline Actions.
2. Database checks: Verify `observations`, `rl_policies`, `rl_training_runs`, and `reward_logs` tables have valid data.
3. Ensure no latency-sensitive synchronous paths were added (RL Teachers MUST remain offline batch/Slow Loop workers).
