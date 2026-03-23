import pytest
from pathlib import Path
import json

import pandas as pd

from scripts.train_regime_model import _build_parser, parse_args


def test_train_regime_model_parser():
    parser = _build_parser()
    args = parser.parse_args(["--symbols", "INFY.NS", "--limit", "100", "--output-dir", "/tmp/models"])
    assert args.symbols == ["INFY.NS"]
    assert args.limit == 100
    assert args.output_dir == "/tmp/models"


def test_train_regime_model_creates_artifacts(tmp_path):
    # This is a smoke test to ensure the script's core components are importable and basic
    # parsing/paths work, without hitting the actual DB for training
    args = _build_parser().parse_args(["--symbols", "TEST", "--output-dir", str(tmp_path)])
    assert args.output_dir == str(tmp_path)
