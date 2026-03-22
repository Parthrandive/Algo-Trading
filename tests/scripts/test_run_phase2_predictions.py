from __future__ import annotations

from types import SimpleNamespace

from scripts import run_phase2_predictions


def test_resolve_symbols_to_score_prefers_explicit_symbols(monkeypatch):
    args = SimpleNamespace(
        symbols=["USDINR=X", "RELIANCE.NS", "USDINR=X"],
        interval="1h",
        db_url=None,
        technical_limit=300,
        regime_limit=800,
    )

    def _unexpected_discovery(**kwargs):
        raise AssertionError("auto-discovery should not run when explicit symbols are provided")

    monkeypatch.setattr(run_phase2_predictions, "discover_training_symbols", _unexpected_discovery)

    resolved = run_phase2_predictions._resolve_symbols_to_score(args)

    assert resolved == ["USDINR=X", "RELIANCE.NS"]


def test_resolve_symbols_to_score_uses_discovery_when_symbols_are_omitted(monkeypatch):
    args = SimpleNamespace(
        symbols=None,
        interval="1h",
        db_url="postgresql://example",
        technical_limit=300,
        regime_limit=800,
    )

    class _FakeLoader:
        def load_historical_bars(self, *args, **kwargs):
            raise AssertionError("validator should not be called by this fake discovery path")

    monkeypatch.setattr(run_phase2_predictions, "DataLoader", lambda *_args, **_kwargs: _FakeLoader())
    monkeypatch.setattr(
        run_phase2_predictions,
        "discover_training_symbols",
        lambda **kwargs: SimpleNamespace(active_symbols=["RELIANCE.NS", "TATASTEEL.NS"]),
    )

    resolved = run_phase2_predictions._resolve_symbols_to_score(args)

    assert resolved == ["RELIANCE.NS", "TATASTEEL.NS"]
