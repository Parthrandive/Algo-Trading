import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.sentinel.nsepython_client import NSEPythonClient
from src.agents.sentinel.yfinance_client import YFinanceClient


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def build_symbol_candidates(raw_symbol: str) -> list[str]:
    symbol = raw_symbol.strip().upper()
    if not symbol:
        return ["RELIANCE.NS"]

    if symbol.endswith(".NS"):
        base_symbol = symbol[:-3]
        return _dedupe_preserve_order([symbol, base_symbol])

    # For plain company symbols (e.g., TCS), try NSE suffix first.
    is_plain_ticker = symbol.replace("-", "").isalnum() and "." not in symbol and "=" not in symbol and "^" not in symbol
    if is_plain_ticker:
        return _dedupe_preserve_order([f"{symbol}.NS", symbol])

    return [symbol]


def resolve_historical_dir(symbol_candidates: list[str]) -> tuple[str, Path]:
    root = Path("data/silver/ohlcv")
    if not root.exists():
        return symbol_candidates[0], root / symbol_candidates[0]

    # Exact match first.
    for candidate in symbol_candidates:
        candidate_dir = root / candidate
        if candidate_dir.exists():
            return candidate, candidate_dir

    # Case-insensitive fallback.
    available_dirs = {entry.name.upper(): entry for entry in root.iterdir() if entry.is_dir()}
    for candidate in symbol_candidates:
        match = available_dirs.get(candidate.upper())
        if match is not None:
            return match.name, match

    return symbol_candidates[0], root / symbol_candidates[0]


def show_historical_data(symbol_candidates: list[str]) -> None:
    storage_symbol, base_dir = resolve_historical_dir(symbol_candidates)
    print(f"--- 1. Checking Historical Data (Silver Layer) for {storage_symbol} ---")

    if not base_dir.exists():
        print(f"No historical data found in {base_dir}")
        return

    files = sorted(base_dir.rglob("*.parquet"))
    if not files:
        print("No parquet files found.")
        return

    latest_files = files[-2:]
    print(f"Reading files: {[f.name for f in latest_files]}")

    dataframes = []
    for parquet_file in latest_files:
        try:
            dataframes.append(pd.read_parquet(parquet_file))
        except Exception as exc:
            print(f"Skipping unreadable file {parquet_file.name}: {exc}")

    if not dataframes:
        print("No readable parquet files available.")
        return

    df = pd.concat(dataframes, ignore_index=True)
    preferred_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    display_columns = [column for column in preferred_columns if column in df.columns]

    if not display_columns:
        print("Historical files loaded, but expected OHLCV columns are missing.")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"\nLast 10 records (spanning {len(latest_files)} days):")
    print(df.tail(10)[display_columns])


def choose_client_order(symbol: str) -> list[tuple[str, object]]:
    if symbol.endswith(".NS"):
        return [("NSEPython", NSEPythonClient()), ("YFinance", YFinanceClient())]
    return [("YFinance", YFinanceClient()), ("NSEPython", NSEPythonClient())]


def show_live_quote(symbol_candidates: list[str]) -> None:
    print("\n--- 2. Checking Real-Time Data (Live Quote) ---")

    errors: list[str] = []
    for candidate in symbol_candidates:
        print(f"Trying symbol: {candidate}")

        for client_name, client in choose_client_order(candidate):
            print(f"Fetching via {client_name}...")
            try:
                tick = client.get_stock_quote(candidate)
                print("\nSUCCESS: Received Live Tick")
                print(f"Symbol: {tick.symbol}")
                print(f"Price:  {tick.price}")
                print(f"Volume: {tick.volume}")
                print(f"Source: {tick.source_type}")
                print(f"Time:   {tick.timestamp}")
                return
            except Exception as exc:
                errors.append(f"{client_name} ({candidate}): {exc}")

    print("Failed to fetch live quote for all symbol/provider attempts.")
    for error in errors:
        print(f"- {error}")


def show_data() -> None:
    raw_symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    symbol_candidates = build_symbol_candidates(raw_symbol)

    show_historical_data(symbol_candidates)
    show_live_quote(symbol_candidates)


if __name__ == "__main__":
    show_data()
