# MLOps Governance Standards

## 1. Model Metadata (The "Model Card")

Every deployed model MUST have a corresponding metadata file (JSON/YAML) containing:
- **Model ID**: Unique identifier (e.g., `model_v1.2.3_20260213`)
- **Version**: Semantic versioning (Major.Minor.Patch)
- **Author/Owner**: Team or Individual responsible
- **Creation Date**: Timestamp of training completion
- **Algorithm**: Type of model (e.g., LSTM, XGBoost, Transformer)
- **Hyperparameters**: Link to config file or embedded values

## 2. Training Data Lineage

- **Dataset ID**: Hash or S3 path to the exact training dataset used.
- **Time Range**: Start and End dates of training data.
- **Features**: List of input features used.
- **Preprocessing**: Reference to the specific preprocessing pipeline version.

## 3. Performance Metrics & Guardrails

### Key Metrics
- **Backtest Sharpe Ratio**: > 1.5 (Minimum for deployment)
- **Max Drawdown**: < 20% (Hard limit)
- **Win Rate**: > 55% (Target)

### drift Detection
- **Input Drift**: Monitor distribution shift of input features (KL Divergence).
- **Concept Drift**: Monitor degradation of model prediction accuracy over time.

### Research Throughput KPIs
- **Research-to-Production Lead Time**: Track idea -> offline backtest -> paper -> shadow -> live.
- **Experiment Throughput**: Track experiments completed per day/week by strategy family.
- **Compute Cost Efficiency**: Track compute cost per approved strategy update.
- **Failure Attribution**: Report percentage of failed experiments due to data integrity vs model logic.
- **Evidence-First Uplift Rule**: Fixed Sharpe or accuracy uplift claims are prohibited in release approvals unless backed by controlled shadow A/B evidence.

### Low-Latency ML Release Guardrails
- **Execution Path Rule**: Fast Loop uses distilled student policy inference only.
- **Determinism Rule**: Model promotion requires deterministic runtime profile and tail-latency benchmark evidence.
- **Teacher Isolation Rule**: Teacher policy evaluation runs offline or Slow Loop analytics only.
- **Lightweight Inference Rule**: Fast-loop inference artifacts must stay within approved compute envelope and fallback safely on latency breach.

## 4. Model Lifecycle States

1. **Development**: Experimental phase. No production access.
2. **Staging**: Shadow mode deployment. Receives live data, generates signals, but NO orders.
3. **Production**: Live trading. Full automated execution.
4. **Retired**: Model deprecated. Archived for audit but not active.

## 5. Approval Process

- **Dev -> Staging**: Automated tests pass + Code Review.
- **Staging -> Production**: 
    - Minimum 1 week in Staging with positive expectancy.
    - Sign-off from Risk Manager.
    - No critical incidents during Staging period.
    - Research-throughput KPIs are current and attached to release record.
    - Change record includes expected impact, rollback trigger, and 48-hour post-deploy review owner.
    - Major USD/INR and gold strategy/model changes pass mandatory shadow A/B evidence.
    - Change submission includes completed `release_change_request_template.md`.
    - CI evidence submission includes completed `ci_benchmark_evidence_checklist.md`.

## 6. Update Cadence Controls

- Nightly retraining remains default for production models.
- Optional micro-update cadence (15 to 30 minutes) is allowed only in paper/shadow environments until non-regression and risk sign-off are complete.
- If compute scales materially, parallelized/distributed training must preserve reproducibility metadata and cost accountability.
