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
