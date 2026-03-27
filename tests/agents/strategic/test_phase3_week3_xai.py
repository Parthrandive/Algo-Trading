from __future__ import annotations

from src.agents.strategic.xai_attribution import OperationalMetricsBoard, PnLAttributionEngine, XAILogger


def test_xai_logger_tracks_top_k_and_coverage():
    logger = XAILogger()
    logger.mark_trade_seen("t1")
    logger.mark_trade_seen("t2")

    explanation = logger.log_trade(
        trade_id="t1",
        symbol="RELIANCE.NS",
        feature_contributions={
            "orderbook_imbalance": 0.42,
            "sentiment_z_t": 0.1,
            "regime_state": -0.25,
            "consensus_confidence": 0.2,
            "var_95": -0.05,
            "queue_pressure": 0.09,
        },
        agent_contributions={"technical": 0.4, "regime": 0.2, "sentiment": 0.1, "consensus": 0.3},
        signal_family_contributions={"microstructure": 0.5, "risk": 0.2, "consensus": 0.3},
    )

    assert explanation.trade_id == "t1"
    assert len(explanation.top_feature_contributions) == 5
    assert logger.coverage() == 0.5
    assert logger.coverage_ok() is False


def test_pnl_attribution_aggregates_all_dimensions():
    pnl = PnLAttributionEngine()
    pnl.add_event(
        trade_id="a1",
        symbol="RELIANCE.NS",
        sector="energy",
        agent="technical",
        signal_family="momentum",
        realized_pnl=1500.0,
    )
    pnl.add_event(
        trade_id="a2",
        symbol="TCS.NS",
        sector="it",
        agent="regime",
        signal_family="macro",
        realized_pnl=-500.0,
    )

    totals = pnl.totals()
    assert totals.total_pnl == 1000.0
    assert totals.by_agent["technical"] == 1500.0
    assert totals.by_signal_family["macro"] == -500.0
    assert totals.by_sector["it"] == -500.0


def test_operational_metrics_board_reports_expected_fields():
    board = OperationalMetricsBoard()
    board.add_decision_staleness(1.2)
    board.add_feature_lag(0.6)
    board.increment_mode_switch()
    board.increment_ood_trigger()
    board.increment_kill_switch_false_positive()
    board.record_mttr(45.0)

    snapshot = board.snapshot()
    assert snapshot["decision_staleness_avg_s"] == 1.2
    assert snapshot["feature_lag_avg_s"] == 0.6
    assert snapshot["mode_switch_frequency"] == 1
    assert snapshot["ood_trigger_rate"] == 1
    assert snapshot["kill_switch_false_positives"] == 1
    assert snapshot["mttr_avg_s"] == 45.0
