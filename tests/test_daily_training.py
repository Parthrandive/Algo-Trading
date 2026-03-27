import json

import pytest
import scripts.run_daily_training as daily_training
from scripts.run_daily_training import _build_parser


def test_daily_training_parser():
    parser = _build_parser()
    
    # Test daily
    args = parser.parse_args(["--mode", "daily"])
    assert args.mode == "daily"
    assert args.symbols is None
    assert args.arima_daily_filter_mode == "soft"
    assert args.arima_daily_up_penalty == pytest.approx(0.15)
    assert args.arima_class_threshold_min == pytest.approx(0.001)
    assert args.regime_min_gold_rows == 120
    assert args.cnn_standalone_symbols == "RELIANCE.NS,HDFCBANK.NS"
    assert args.xgb_arima_readiness_symbols == "RELIANCE.NS"
    assert args.xgb_arima_min_directional_accuracy == pytest.approx(0.50)
    assert args.xgb_daily_regime_filter_mode == "long_only_above_ma200"
    
    # Test weekly
    args = parser.parse_args(["--mode", "weekly", "--symbols", "RELIANCE.NS"])
    assert args.mode == "weekly"
    assert args.symbols == ["RELIANCE.NS"]


def test_weekly_pipeline_propagates_single_run_id(monkeypatch):
    captured_cmds: list[list[str]] = []

    def _fake_run_cmd(cmd: list[str]) -> bool:
        captured_cmds.append(cmd)
        return True

    monkeypatch.setattr(daily_training, "run_cmd", _fake_run_cmd)
    monkeypatch.setattr(
        daily_training,
        "_validate_artifact_contract",
        lambda symbols, run_id, interval="1h", xgb_optional_symbols=None: (True, []),
    )
    monkeypatch.setattr(
        daily_training,
        "_validate_retrain_evidence",
        lambda symbols, xgb_optional_symbols=None: (True, []),
    )

    run_id = "20260326T190000Z"
    args = daily_training._build_parser().parse_args([])
    ok = daily_training.run_training_pipeline(
        ["RELIANCE.NS", "TATASTEEL.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"],
        "weekly",
        run_id,
        args,
    )
    assert ok

    flattened = [" ".join(cmd) for cmd in captured_cmds]
    assert any("train_universe_arima_lstm.py" in cmd and f"--run-id {run_id}" in cmd for cmd in flattened)
    assert any("train_universe_arima_lstm.py" in cmd and "--daily-filter-mode soft" in cmd for cmd in flattened)
    assert any("train_universe_arima_lstm.py" in cmd and "--daily-up-penalty 0.15" in cmd for cmd in flattened)
    assert any("train_universe_arima_lstm.py" in cmd and "--class-threshold-min 0.001" in cmd for cmd in flattened)
    assert any("train_regime_model.py" in cmd and "--min-gold-rows 120" in cmd for cmd in flattened)
    assert any("train_universe_cnn.py" in cmd and f"--run-id {run_id}" in cmd for cmd in flattened)
    assert any("train_universe_xgboost.py" in cmd and f"--run-id {run_id}" in cmd for cmd in flattened)
    assert any("train_universe_xgboost.py" in cmd and f"--expected-run-id {run_id}" in cmd for cmd in flattened)
    assert any("train_universe_xgboost.py" in cmd and "--strict-artifact-match" in cmd for cmd in flattened)
    assert any("train_universe_xgboost.py" in cmd and "--daily-regime-filter-mode long_only_above_ma200" in cmd for cmd in flattened)
    assert any("diagnose_arima_regression.py" in cmd and f"--run-id {run_id}" in cmd for cmd in flattened)


def test_validate_artifact_contract_passes_for_five_symbols(tmp_path, monkeypatch):
    symbols = ["RELIANCE.NS", "TATASTEEL.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"]
    run_id = "20260326T190500Z"
    layers = ["arima_lstm", "cnn_pattern", "xgboost"]

    for layer in layers:
        for symbol in symbols:
            sym_dir = tmp_path / "data" / "models" / layer / symbol.replace(".", "_")
            sym_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_id": run_id,
                "interval": "1h",
                "symbol_canonical": symbol,
                "source_script": f"train_universe_{layer}.py",
                "feature_schema_version": "technical_features_v1",
                "split_counts": {"train": 100, "val": 20, "test": 20},
            }
            (sym_dir / "training_meta.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(daily_training, "PROJECT_ROOT", tmp_path)
    ok, errors = daily_training._validate_artifact_contract(symbols, run_id, interval="1h")
    assert ok
    assert errors == []


def test_validate_artifact_contract_rejects_empty_splits(tmp_path, monkeypatch):
    symbol = "RELIANCE.NS"
    run_id = "20260326T191000Z"
    layers = ["arima_lstm", "cnn_pattern", "xgboost"]
    for layer in layers:
        sym_dir = tmp_path / "data" / "models" / layer / symbol.replace(".", "_")
        sym_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "interval": "1h",
            "symbol_canonical": symbol,
            "source_script": f"train_universe_{layer}.py",
            "feature_schema_version": "technical_features_v1",
            "split_counts": {"train": 100, "val": 0, "test": 20},
        }
        (sym_dir / "training_meta.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(daily_training, "PROJECT_ROOT", tmp_path)
    ok, errors = daily_training._validate_artifact_contract([symbol], run_id, interval="1h")
    assert not ok
    assert any("empty split counts" in err for err in errors)


def test_validate_retrain_evidence_passes(tmp_path, monkeypatch):
    symbols = ["RELIANCE.NS"]
    canonical = symbols[0].replace(".", "_")
    for layer in ("arima_lstm", "cnn_pattern", "xgboost"):
        sym_dir = tmp_path / "data" / "models" / layer / canonical
        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "training_meta.json").write_text(json.dumps({"metrics": {"ok": True}}), encoding="utf-8")
    (tmp_path / "data" / "models" / "xgboost" / canonical / "evaluation_bundle.json").write_text(
        json.dumps({"ok": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_training, "PROJECT_ROOT", tmp_path)
    ok, errors = daily_training._validate_retrain_evidence(symbols)
    assert ok
    assert errors == []
