"""
Tests for the latency instrumentation module.
"""

from __future__ import annotations

import time

import pytest

from src.utils.latency import LatencyReport, LatencyStore, StageStats, timed


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the global LatencyStore between tests."""
    LatencyStore.reset()
    yield
    LatencyStore.reset()


class TestLatencyStore:
    def test_record_and_stats(self):
        store = LatencyStore.instance()
        store.record("agent_a", "stage_1", 0.1)
        store.record("agent_a", "stage_1", 0.2)
        store.record("agent_a", "stage_1", 0.3)

        stats = store.stats("agent_a", "stage_1")
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(0.2, abs=0.01)
        assert stats["p50"] >= 0.1

    def test_stats_empty_key(self):
        store = LatencyStore.instance()
        stats = store.stats("no_agent", "no_stage")
        assert stats["count"] == 0
        assert stats["p50"] == 0.0

    def test_all_keys(self):
        store = LatencyStore.instance()
        store.record("a", "s1", 0.1)
        store.record("b", "s2", 0.2)
        keys = store.all_keys()
        assert ("a", "s1") in keys
        assert ("b", "s2") in keys

    def test_percentiles_known_data(self):
        store = LatencyStore.instance()
        # 100 samples: 0.01, 0.02, ..., 1.00
        for i in range(1, 101):
            store.record("test", "pct", i / 100.0)

        stats = store.stats("test", "pct")
        assert stats["count"] == 100
        assert stats["p50"] == pytest.approx(0.50, abs=0.02)
        assert stats["p95"] == pytest.approx(0.95, abs=0.02)
        assert stats["p99"] == pytest.approx(0.99, abs=0.02)

    def test_singleton(self):
        s1 = LatencyStore.instance()
        s2 = LatencyStore.instance()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        s1 = LatencyStore.instance()
        LatencyStore.reset()
        s2 = LatencyStore.instance()
        assert s1 is not s2


class TestTimedDecorator:
    def test_timed_records_duration(self):
        @timed("test_agent", "test_stage")
        def slow_fn():
            time.sleep(0.05)
            return 42

        result = slow_fn()
        assert result == 42

        stats = LatencyStore.instance().stats("test_agent", "test_stage")
        assert stats["count"] == 1
        assert stats["p50"] >= 0.04  # At least ~50ms

    def test_timed_preserves_function_name(self):
        @timed("a", "b")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_timed_records_on_exception(self):
        @timed("err_agent", "err_stage")
        def failing_fn():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_fn()

        stats = LatencyStore.instance().stats("err_agent", "err_stage")
        assert stats["count"] == 1

    def test_timed_multiple_calls_accumulate(self):
        @timed("multi", "calls")
        def noop():
            pass

        for _ in range(5):
            noop()

        stats = LatencyStore.instance().stats("multi", "calls")
        assert stats["count"] == 5


class TestLatencyReport:
    def test_from_store_empty(self):
        report = LatencyReport.from_store()
        assert report.stages == []

    def test_from_store_with_data(self):
        store = LatencyStore.instance()
        store.record("agent_x", "stage_y", 0.123)
        store.record("agent_x", "stage_y", 0.456)

        report = LatencyReport.from_store(store)
        assert len(report.stages) == 1
        assert report.stages[0].agent == "agent_x"
        assert report.stages[0].stage == "stage_y"
        assert report.stages[0].count == 2

    def test_to_json(self):
        store = LatencyStore.instance()
        store.record("a", "b", 0.1)
        report = LatencyReport.from_store(store)
        json_str = report.to_json()
        assert '"agent": "a"' in json_str
        assert '"stage": "b"' in json_str

    def test_to_markdown(self):
        store = LatencyStore.instance()
        store.record("sentinel", "ingest", 0.12345)
        report = LatencyReport.from_store(store)
        md = report.to_markdown()
        assert "# Latency Report" in md
        assert "sentinel" in md
        assert "ingest" in md
        assert "P50" in md
