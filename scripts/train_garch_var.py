import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.technical.data_loader import DataLoader
from src.agents.technical.models.garch_var import GarchVaRModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set reproducibility seeds."""
    np.random.seed(seed)


def validate_data(df: pd.DataFrame) -> None:
    """Validate data quality before fitting GARCH."""
    if len(df) < 30:
        raise ValueError(f"Need at least 30 rows to fit GARCH. Got {len(df)}.")

    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column required for GARCH.")

    nan_pct = df['close'].isna().mean()
    if nan_pct > 0.05:
        raise ValueError(f"Column 'close' has {nan_pct:.1%} NaNs (max 5% allowed).")

    if df['close'].nunique() == 1:
        logger.warning("Target column 'close' is constant. Model cannot fit variance.")


def main():
    parser = argparse.ArgumentParser(description="Fit standalone GARCH VaR model.")
    parser.add_argument("--symbol", default="TATASTEEL.NS", help="Stock symbol to fit on")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to fetch from DB")
    parser.add_argument("--window-size", type=int, default=252, help="Trading days window size")
    parser.add_argument("--dist", default="normal", choices=["normal", "t", "skewstudent"], help="Return distribution model")
    parser.add_argument("--run-backtest", action="store_true", help="Run historical Kupiec POF backtest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data/models/garch_var/", help="Output directory for metadata")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Fetch Data
    logger.info(f"Fetching data for {args.symbol} from DB...")
    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    try:
        df = loader.load_historical_bars(args.symbol, limit=args.limit).dropna(subset=['close'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Validate Data
    logger.info(f"Loaded {len(df)} rows. Validating...")
    validate_data(df)

    # 3. Fit Model
    logger.info(f"Fitting GARCH(1,1) model (window={args.window_size}, dist={args.dist})...")
    model = GarchVaRModel(window_size=args.window_size, dist=args.dist)
    
    try:
        model.fit(df, price_col='close')
        logger.info(f"Fit complete. Convergence Flag: {model.last_convergence_flag} (0=success)")
    except Exception as e:
        logger.error(f"GARCH fit failed: {e}")
        sys.exit(1)

    # 4. Forecast Risk
    logger.info("Generating 1-step ahead risk forecasts...")
    risk = model.forecast_risk(confidence_levels=(0.95, 0.99))
    logger.info(f"Volatility Forecast: {risk['volatility_forecast']:.6f}")
    logger.info(f"Parametric VaR (95%): {risk['parametric_var_95']:.6f} | ES (95%): {risk['parametric_es_95']:.6f}")
    logger.info(f"Parametric VaR (99%): {risk['parametric_var_99']:.6f} | ES (99%): {risk['parametric_es_99']:.6f}")

    # 5. Optional Backtest
    backtest_metrics = None
    if args.run_backtest:
        logger.info("Running historical VaR backtest (Kupiec POF test)...")
        if len(df) < args.window_size + 50:
            logger.warning("Not enough data to run a meaningful rolling backtest. Skipping.")
        else:
            try:
                summary = model.backtest_var(df, price_col='close', confidence_levels=(0.95, 0.99), method="parametric")
                logger.info("--- 95% Confidence Level Backtest ---")
                logger.info(f"Breach Rate: {summary[95]['breach_rate']:.4f} (Expected: {summary[95]['expected_breach_rate']:.4f})")
                logger.info(f"P-Value: {summary[95]['p_value']:.4f}")
                
                logger.info("--- 99% Confidence Level Backtest ---")
                logger.info(f"Breach Rate: {summary[99]['breach_rate']:.4f} (Expected: {summary[99]['expected_breach_rate']:.4f})")
                logger.info(f"P-Value: {summary[99]['p_value']:.4f}")
                
                backtest_metrics = summary
            except Exception as e:
                logger.error(f"Backtest failed: {e}")

    # 6. Save Metadata (No weights saved for GARCH)
    logger.info(f"Saving metadata to {args.output_dir}")
    
    # Extract fitted params
    params = {}
    if model.fit_result:
        for name, value in model.fit_result.params.items():
            params[name] = float(value)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "input_rows": len(df),
        "hyperparameters": {
            "window_size": model.window_size,
            "dist": model.dist,
            "seed": args.seed
        },
        "fit_results": {
            "convergence_flag": model.last_convergence_flag,
            "params": params
        },
        "forecasts": risk
    }
    
    if backtest_metrics:
        meta["backtest"] = backtest_metrics
        
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    logger.info("GARCH run complete.")


if __name__ == "__main__":
    main()
