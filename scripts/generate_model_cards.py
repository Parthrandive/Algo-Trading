import os
import json
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_arima_lstm_card():
    card = {
        "model_id": "arima_lstm_hybrid_v1",
        "version": "1.0",
        "owner": "Technical Agent",
        "algorithm": "ARIMA-LSTM Hybrid",
        "description": "ARIMA for linear trend forecasting combined with LSTM for non-linear residual prediction.",
        "hyperparameters": {},
        "features": [],
        "metrics": {}
    }
    
    base_dir = "data/models/arima_lstm"
    
    # Load hyperparams
    hp_path = os.path.join(base_dir, "hyperparams.json")
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            hp = json.load(f)
            card["hyperparameters"] = {k: v for k, v in hp.items() if k != "feature_columns"}
            card["features"] = hp.get("feature_columns", [])
            
    # Load training meta
    meta_path = os.path.join(base_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            card["metrics"]["final_train_loss"] = meta.get("final_train_loss")
            card["metrics"]["final_val_loss"] = meta.get("final_val_loss")
            card["training_date"] = meta.get("timestamp")
            card["training_symbol"] = meta.get("symbol")
            
    with open(os.path.join(base_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=4)
    logger.info("Generated ARIMA-LSTM Model Card.")

def generate_cnn_pattern_card():
    card = {
        "model_id": "cnn_pattern_classifier_v1",
        "version": "1.0",
        "owner": "Technical Agent",
        "algorithm": "2D Convolutional Neural Network",
        "description": "Treats historical OHLCV windows as 1D/2D images to classify next-bar price direction (Up/Down/Neutral).",
        "hyperparameters": {},
        "features": [],
        "metrics": {}
    }
    
    base_dir = "data/models/cnn_pattern"
    
    # Load hyperparams
    hp_path = os.path.join(base_dir, "hyperparams.json")
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            hp = json.load(f)
            card["hyperparameters"] = {k: v for k, v in hp.items() if k != "feature_columns"}
            card["features"] = hp.get("feature_columns", [])
            
    # Load training meta
    meta_path = os.path.join(base_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            card["metrics"]["final_train_loss"] = meta.get("final_train_loss")
            card["metrics"]["final_val_loss"] = meta.get("final_val_loss")
            card["metrics"]["final_train_acc"] = meta.get("final_train_acc")
            card["metrics"]["final_val_acc"] = meta.get("final_val_acc")
            card["training_date"] = meta.get("timestamp")
            
    with open(os.path.join(base_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=4)
    logger.info("Generated CNN Pattern Model Card.")

def generate_garch_var_card():
    card = {
        "model_id": "garch_var_v1",
        "version": "1.0",
        "owner": "Technical Agent",
        "algorithm": "GARCH(1,1)",
        "description": "Volatility forecasting and parametric Value-at-Risk / Expected Shortfall modeling.",
        "hyperparameters": {},
        "features": ["close_returns"],
        "metrics": {}
    }
    
    base_dir = "data/models/garch_var"
    
    # Load training meta
    meta_path = os.path.join(base_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            card["hyperparameters"] = meta.get("hyperparameters", {})
            card["metrics"]["fit_convergence"] = meta.get("fit_results", {}).get("convergence_flag")
            card["training_date"] = meta.get("timestamp")
            
    with open(os.path.join(base_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=4)
    logger.info("Generated GARCH VaR Model Card.")

if __name__ == "__main__":
    generate_arima_lstm_card()
    generate_cnn_pattern_card()
    generate_garch_var_card()
    logger.info("All model cards generated successfully.")
