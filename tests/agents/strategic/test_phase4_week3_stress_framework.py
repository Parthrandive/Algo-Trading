from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from src.agents.strategic.schemas import RiskMode
from src.agents.strategic.week4 import (
    StressScenarioDefinition,
    StressTestEngine,
)


def test_week3_scenario_library_contains_required_entries() -> None:
    engine = StressTestEngine()
    library = engine.scenario_library()
    scenario_ids = {item.scenario_id for item in library}

    assert scenario_ids == {
        "rbi_surprise_rate_hike",
        "inr_flash_move",
        "liquidity_drought",
        "gfc_2008",
        "taper_tantrum_2013",
        "covid_crash_2020",
        "correlation_inversion",
        "frozen_constituent_prices",
        "multi_asset_liquidity_vacuum",
        "data_poisoning_feed_freeze",
    }
    types = {item.scenario_type for item in library}
    assert {"historical", "historical_synthetic", "synthetic", "impossible"}.issubset(types)


def test_week3_capacity_replays_expand_1x_2x_3x_and_force_2x_cap() -> None:
    engine = StressTestEngine(capacity_impact_cap_bps=25.0)
    definitions = (
        StressScenarioDefinition(
            scenario_id="liquidity_drought",
            scenario_type="synthetic",
            key_test="impact-capacity stress",
        ),
    )
    scenarios = list(engine.build_capacity_replays(definitions))

    assert [item.capacity_multiplier for item in scenarios] == [1.0, 2.0, 3.0]

    scenarios[-1] = replace(
        scenarios[-1],
        impact_bps=32.0,
        protective_mode=RiskMode.REDUCE_ONLY,
        expected_min_mode=RiskMode.REDUCE_ONLY,
    )
    report = engine.evaluate(scenarios)

    assert report.passed is False
    assert report.hard_cap_forced is True
    assert report.capacity_hard_cap_multiplier == 2.0
    assert any("impact_cap_breach" in item for item in report.failure_reasons)
    assert "global:capacity_hard_cap_forced_2x" in report.failure_reasons


def test_week3_nightly_ci_scenarios_cover_full_library() -> None:
    engine = StressTestEngine()
    scenarios = engine.build_nightly_ci_scenarios()

    assert len(scenarios) == 30
    multipliers = {item.capacity_multiplier for item in scenarios}
    assert multipliers == {1.0, 2.0, 3.0}
    ids = {item.scenario_id.rsplit("_", maxsplit=1)[0] for item in scenarios}
    assert ids == {definition.scenario_id for definition in engine.scenario_library()}


def test_week3_impossible_scenarios_enforce_core_safety_checks() -> None:
    engine = StressTestEngine()
    scenarios = [
        replace(
            engine.build_capacity_replays(
                (
                    StressScenarioDefinition(
                        scenario_id="correlation_inversion",
                        scenario_type="impossible",
                        key_test="variance integrity",
                    ),
                )
            )[0],
            variance_defined=False,
        ),
        replace(
            engine.build_capacity_replays(
                (
                    StressScenarioDefinition(
                        scenario_id="frozen_constituent_prices",
                        scenario_type="impossible",
                        key_test="zero vs missing",
                    ),
                )
            )[0],
            zero_vs_missing_distinguished=False,
        ),
        replace(
            engine.build_capacity_replays(
                (
                    StressScenarioDefinition(
                        scenario_id="data_poisoning_feed_freeze",
                        scenario_type="synthetic",
                        key_test="safe mode",
                    ),
                )
            )[0],
            feed_integrity_uncertain=True,
            safe_mode_engaged=False,
        ),
    ]

    report = engine.evaluate(scenarios)

    assert report.passed is False
    assert any("variance_invalid" in item for item in report.failure_reasons)
    assert any("zero_missing_confused" in item for item in report.failure_reasons)
    assert any("safe_mode_not_engaged" in item for item in report.failure_reasons)


def test_week3_quarterly_review_template_captures_actions() -> None:
    engine = StressTestEngine(capacity_impact_cap_bps=25.0)
    scenarios = engine.build_capacity_replays(engine.scenario_library()[:1])
    stressed = [
        replace(scenarios[0], snapback_ticks=80, max_snapback_ticks=30),
        replace(scenarios[1], impact_bps=5.0),
        replace(scenarios[2], impact_bps=31.0),
    ]
    report = engine.evaluate(stressed)
    review = engine.generate_quarterly_review(
        report=report,
        owner="risk_owner",
        reviewer="risk_reviewer",
        previous_failure_count=1,
        generated_at=datetime(2026, 7, 1, 9, 0, tzinfo=UTC),
    )

    assert review.quarter_label == "2026-Q3"
    assert review.owner == "risk_owner"
    assert review.reviewer == "risk_reviewer"
    assert review.hard_cap_forced is True
    assert review.hard_cap_multiplier == 2.0
    assert "rerun_failed_scenarios" in review.required_actions
    assert "enforce_capacity_hard_cap_2x" in review.required_actions
    assert "review_snapback_tuning_manually" in review.required_actions
