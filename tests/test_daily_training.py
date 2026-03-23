import pytest
from scripts.run_daily_training import _build_parser


def test_daily_training_parser():
    parser = _build_parser()
    
    # Test daily
    args = parser.parse_args(["--mode", "daily"])
    assert args.mode == "daily"
    assert args.symbols is None
    
    # Test weekly
    args = parser.parse_args(["--mode", "weekly", "--symbols", "RELIANCE.NS"])
    assert args.mode == "weekly"
    assert args.symbols == ["RELIANCE.NS"]
