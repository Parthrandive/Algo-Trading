# Technical Agent (Phase 2)

**Owner:** Sentinel Trade Bot  
**Role:** Market data analysis and statistical price/risk forecasting.

The Technical Agent serves as the first intelligent layer above the raw `.gold` market data. It consumes standard OHLCV streams and produces unified quantitative forecasts using three separate model families.

## Architecture

The Technical Agent operates as an ensemble containing:
1. **ARIMA-LSTM Hybrid:** Predicts numerical price magnitude ($) by separating linear trends from non-linear residuals.
2. **2D CNN Pattern Classifier:** Treats trailing price movement as a 2D image, predicting short-term direction (`up`, `down`, `neutral`).
3. **GARCH(1,1):** Estimates current market volatility and computes downside risk boundaries using Parametric Value-at-Risk (VaR) and Expected Shortfall (ES).

## Data Flow & Downstream Contract

All calls route through the main entrypoint: `TechnicalAgent.predict(symbol: str)`.

```python
from src.agents.technical.technical_agent import TechnicalAgent

# 1. Initialize agent (loads trained models from disk)
agent = TechnicalAgent(db_url="postgresql://user:pass@localhost/db", models_dir="data/models")

# 2. Predict next horizon
prediction = agent.predict("RELIANCE.NS")

if prediction:
    print(prediction.direction)            # "up", "down", "neutral"
    print(prediction.price_forecast)       # e.g., 1410.50
    print(prediction.volatility_estimate)  # e.g., 0.015 (1.5%)
    print(prediction.var_95)               # e.g., -0.02 (Max expected 2% loss at 95% confidence)
```

The returned object strictly adheres to the `TechnicalPrediction` Pydantic schema defined in `schemas.py`.

## Lifecycle Commands

You do not need to train models manually. The Technical module includes an automated continuous-learning suite.

### Unified Training and Testing

Use the top-level script to fetch the latest data, test quality, re-train all three models, run the walk-forward backtester, generate feature ablation metrics, and update the model catalog metadata:

```bash
python scripts/train_models.py --symbol <TICKER>
```

#### Outputs
- Trained weights are saved natively to `data/models/{model_name}/`
- Technical performance reports land in `docs/reports/`
- AI Model Cards (`model_card.json`) are automatically updated in the model directories with hyperparameters and empirical test metrics.

## Avoiding Leakage

The `TechnicalAgent` has zero look-ahead bias by design:
- Target construction shifts close prices by `-1` (predicting $t+1$).
- Training loops split chronologically.
- At inference time (`predict()`), only the most recent complete feature window is fed to the models. Time metadata is strictly stamped post-computation.
