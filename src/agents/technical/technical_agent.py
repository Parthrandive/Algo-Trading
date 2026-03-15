import os
import json
import logging
from typing import Optional
from datetime import datetime, timezone

from src.agents.technical.schemas import TechnicalPrediction
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features
from src.agents.technical.models.arima_lstm import ArimaLstmHybrid
from src.agents.technical.models.cnn_pattern import CnnPatternClassifier
from src.agents.technical.models.garch_var import GarchVaRModel
from src.db.phase2_recorder import Phase2Recorder

logger = logging.getLogger(__name__)

class TechnicalAgent:
    """
    Technical Agent for consuming OHLCV data and producing market forecasts.
    Runs baseline models (ARIMA-LSTM, 2D CNN, GARCH).
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        models_dir: str = "data/models",
        persist_predictions: bool = True,
    ):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
        self.loader = DataLoader(self.db_url)
        self.models_dir = models_dir
        self.persist_predictions = persist_predictions
        self.phase2_recorder = Phase2Recorder(self.db_url) if persist_predictions else None
        
        # Load ARIMA-LSTM
        arima_dir = os.path.join(models_dir, "arima_lstm")
        with open(os.path.join(arima_dir, "hyperparams.json"), "r") as f:
            arima_hp = json.load(f)
        self.arima_lstm = ArimaLstmHybrid(
            arima_order=tuple(arima_hp["arima_order"]),
            lstm_hidden_size=arima_hp.get("lstm_hidden_size", 64),
            lstm_layers=arima_hp.get("lstm_layers", 1),
            window_size=arima_hp["window_size"]
        )
        self.arima_lstm.load(arima_dir)
        self.arima_features = arima_hp["feature_columns"]
        
        # Load CNN Pattern
        cnn_dir = os.path.join(models_dir, "cnn_pattern")
        with open(os.path.join(cnn_dir, "hyperparams.json"), "r") as f:
            cnn_hp = json.load(f)
        self.cnn = CnnPatternClassifier(
            window_size=cnn_hp["window_size"],
            neutral_threshold=cnn_hp.get("neutral_threshold", 0.001)
        )
        self.cnn.load(cnn_dir)
        
        # Load GARCH hyperparams (for fitting on the fly)
        garch_dir = os.path.join(models_dir, "garch_var")
        try:
            with open(os.path.join(garch_dir, "training_meta.json"), "r") as f:
                garch_meta = json.load(f)
                garch_hp = garch_meta.get("hyperparameters", {})
        except FileNotFoundError:
            garch_hp = {"window_size": 252, "dist": "normal"}
            
        self.garch = GarchVaRModel(
            window_size=garch_hp.get("window_size", 252),
            dist=garch_hp.get("dist", "normal")
        )

    def predict(
        self,
        symbol: str,
        *,
        limit: int = 300,
        data_snapshot_id: str | None = None,
    ) -> Optional[TechnicalPrediction]:
        """
        Produce a technical prediction for the given symbol using recent market history.
        """
        # 1. Fetch data
        try:
            df = self.loader.load_historical_bars(symbol, limit=limit)
            if df.empty or len(df) < 20:
                logger.warning(f"Not enough data for {symbol}.")
                return None
                
            # Engineer features for models
            is_forex = symbol.endswith("=X")
            df_feat = engineer_features(df, is_forex=is_forex)
            required_core_cols = set(self.cnn.feature_columns) | {"close"}
            missing_core = sorted(col for col in required_core_cols if col not in df_feat.columns)
            if missing_core:
                logger.warning(f"Missing required feature columns for {symbol}: {missing_core}")
                return None

            # Ignore unrelated/all-NaN columns from upstream payloads (e.g., optional DB columns).
            # ARIMA feature columns may drift across model versions and are handled as best-effort at inference.
            arima_present_cols = {col for col in self.arima_features if col in df_feat.columns}
            clean_subset = sorted(required_core_cols | arima_present_cols)
            df_feat = df_feat.dropna(subset=clean_subset).reset_index(drop=True)
            if len(df_feat) < max(self.arima_lstm.window_size, self.cnn.window_size):
                logger.warning(f"Not enough data after feature engineering for {symbol}.")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch or process data for {symbol}: {e}")
            return None

        # 2. Run ARIMA-LSTM
        try:
            price_forecast = self.arima_lstm.predict(df_feat, target_col='close')
        except Exception as e:
            logger.error(f"ARIMA-LSTM prediction failed: {e}")
            price_forecast = 0.0
            
        # 3. Run CNN
        try:
            direction, probs = self.cnn.predict(df_feat)
            confidence = probs.get(direction, 0.0)
        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            direction = "neutral"
            confidence = 0.0
            
        # 4. Run GARCH
        try:
            self.garch.fit(df_feat, price_col='close')
            risk_metrics = self.garch.forecast_risk(confidence_levels=(0.95, 0.99))
            vol_est = risk_metrics.get("volatility_forecast", 0.0)
            var_95 = risk_metrics.get("parametric_var_95", 0.0)
            var_99 = risk_metrics.get("parametric_var_99", 0.0)
            es_95 = risk_metrics.get("parametric_es_95", 0.0)
            es_99 = risk_metrics.get("parametric_es_99", 0.0)
        except Exception as e:
            logger.error(f"GARCH fitting failed: {e}")
            vol_est = var_95 = var_99 = es_95 = es_99 = 0.0
            
        now = datetime.now(timezone.utc)

        prediction = TechnicalPrediction(
            symbol=symbol,
            timestamp=now,
            price_forecast=float(price_forecast),
            direction=direction, # type: ignore
            volatility_estimate=float(vol_est),
            var_95=float(var_95),
            var_99=float(var_99),
            es_95=float(es_95),
            es_99=float(es_99),
            confidence=float(confidence),
            model_id="ensemble_arima_cnn_garch_v1.0"
        )

        if self.phase2_recorder is not None:
            try:
                self.phase2_recorder.save_technical_prediction(
                    prediction,
                    data_snapshot_id=data_snapshot_id,
                )
            except Exception as exc:
                logger.warning(f"Failed to persist technical prediction for {symbol}: {exc}")

        return prediction
