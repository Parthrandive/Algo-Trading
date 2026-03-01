from src.agents.preprocessing.loader import MarketLoader
from tests.agents.preprocessing.test_cp4_deterministic_replay import mock_market_data

# Use a temporary directory
import tempfile
import pathlib
import sys

with tempfile.TemporaryDirectory() as temp_dir:
    path = mock_market_data(pathlib.Path(temp_dir))
    loader = MarketLoader()
    try:
        loader.load(path, "test_cp4_deterministic_baseline")
    except ValueError as e:
        print(f"Validation Error Captured: {e}")
        sys.exit(0)
