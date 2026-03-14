import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_command(command: list[str]) -> bool:
    """Run a shell command and stream output. Return True if successful."""
    cmd_str = " ".join(command)
    logger.info(f"Running: {cmd_str}")
    
    try:
        result = subprocess.run(command, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {cmd_str}")
        return False
    except Exception as e:
        logger.error(f"Failed to execute command '{cmd_str}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Unified Technical Agent Model Training Entrypoint.")
    parser.add_argument("--symbol", default="TATASTEEL.NS", help="Stock symbol to train all models on")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to fetch from DB")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Optional skip flags
    parser.add_argument("--skip-arima-lstm", action="store_true", help="Skip ARIMA-LSTM training")
    parser.add_argument("--skip-cnn-pattern", action="store_true", help="Skip CNN Pattern training")
    parser.add_argument("--skip-garch-var", action="store_true", help="Skip GARCH VaR fitting")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip Day 5 walk-forward backtest")
    parser.add_argument("--use-nse", action="store_true", help="Fetch data natively from NSE if DB is empty/unavailable")
    parser.add_argument("--interval", default="1d", help="Candle interval for all models, e.g. 1d, 1h")
    
    args = parser.parse_args()

    import os
    common_args = ["--symbol", args.symbol, "--seed", str(args.seed), "--interval", args.interval]
    if args.limit is not None:
        common_args.extend(["--limit", str(args.limit)])
    if args.use_nse:
        common_args.append("--use-nse")
        
    logger.info(f"=== Starting Unified Training for {args.symbol} at {args.interval} interval ===")

    # 1. ARIMA-LSTM
    if not args.skip_arima_lstm:
        script_path = os.path.join("scripts", "train_arima_lstm.py")
        cmd = ["python", script_path] + common_args
        if not run_command(cmd):
            sys.exit(1)
            
    # 2. CNN Pattern
    if not args.skip_cnn_pattern:
        script_path = os.path.join("scripts", "train_cnn_pattern.py")
        cmd = ["python", script_path] + common_args
        if not run_command(cmd):
            sys.exit(1)
            
    # 3. GARCH VaR
    if not args.skip_garch_var:
        script_path = os.path.join("scripts", "train_garch_var.py")
        cmd = ["python", script_path] + common_args + ["--run-backtest"]
        if not run_command(cmd):
            sys.exit(1)
            
    # 4. Backtest & Ablation
    if not args.skip_backtest:
        script_path = os.path.join("scripts", "run_backtest.py")
        cmd = ["python", script_path] + common_args
        if not run_command(cmd):
            sys.exit(1)
    # 5. Generate Model Cards
    logger.info("Generating Model Cards...")
    script_path = os.path.join("scripts", "generate_model_cards.py")
    cmd = ["python", script_path]
    run_command(cmd)

    logger.info("=== All requested training and backtesting completed successfully! ===")

if __name__ == "__main__":
    main()
