# MLOps Governance Standards

## 1. Model Metadata (The "Model Card")

Every deployed model MUST have a corresponding metadata file (JSON/YAML) containing:
- **Model ID**: Unique identifier (e.g., `model_v1.2.3_20260213`)
- **Version**: Semantic versioning (Major.Minor.Patch)
- **Owner**: Team member responsible
- **Creation Date**: Timestamp of training completion
- **Algorithm**: Type of model (e.g., LSTM, XGBoost, Transformer)
- **Hyperparameters**: Link to config file or embedded values

## 2. Training Data Lineage

- **Dataset ID**: Hash or path to the exact training dataset used.
- **Time Range**: Start and End dates of training data.
- **Features**: List of input features used.
- **Preprocessing**: Reference to the specific preprocessing pipeline version.

## 3. Performance Metrics & Guardrails

### Key Metrics
- **Backtest Sharpe Ratio**: > 1.5 (Minimum for deployment)
- **Max Drawdown**: < 20% (Hard limit)
- **Win Rate**: > 55% (Target)

### Drift Detection
- **Input Drift**: Monitor distribution shift of input features (KL Divergence).
- **Concept Drift**: Monitor degradation of model prediction accuracy over time.

### Research Throughput KPIs
- **Research-to-Production Lead Time**: Track idea -> offline backtest -> paper trading -> live.
- **Experiment Throughput**: Track experiments completed per day/week by strategy family.
- **Compute Cost Efficiency**: Track compute cost per approved strategy update.
- **Failure Attribution**: Report percentage of failed experiments due to data integrity vs model logic.
- **Evidence-First Uplift Rule**: Fixed Sharpe or accuracy uplift claims are prohibited in release approvals unless backed by controlled benchmark evidence.

### Low-Latency ML Release Guardrails
- **Execution Path Rule**: Fast Loop uses distilled student policy inference only.
- **Determinism Rule**: Model promotion requires deterministic runtime profile and tail-latency benchmark evidence.
- **Teacher Isolation Rule**: Teacher policy evaluation runs offline or Slow Loop analytics only.
- **Lightweight Inference Rule**: Fast-loop inference artifacts must stay within approved compute envelope and fallback safely on latency breach.

## 4. Model Lifecycle States

1. **Development**: Experimental phase. No production access.
2. **Paper Trading**: Receives live data, generates signals, but NO real orders. Minimum 2 weeks before production promotion.
3. **Production**: Live trading. Full automated execution.
4. **Retired**: Model deprecated. Archived for audit but not active.

## 5. Approval Process

- **Dev -> Paper Trading**: CI tests pass + Owner self-review.
- **Paper Trading -> Production**:
    - Minimum 2 weeks in paper trading with positive expectancy.
    - Owner verifies risk controls are not regressed.
    - Partner cross-checks critical logic and benchmark evidence.
    - No critical incidents during paper-trading period.
    - Research-throughput KPIs are current and attached to release record.
    - Change record includes expected impact, rollback trigger, and post-deploy monitoring plan.

## 6. Update Cadence Controls

- Nightly retraining remains default for production models.
- Optional micro-update cadence (15 to 30 minutes) is allowed only in paper environments until non-regression and risk checks are complete.
- If compute scales materially, parallelized/distributed training must preserve reproducibility metadata and cost accountability.
