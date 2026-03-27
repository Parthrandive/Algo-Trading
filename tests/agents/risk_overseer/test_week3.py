from __future__ import annotations

from src.agents.risk_overseer import (
    DEFAULT_STRESS_SCENARIOS,
    RiskOverseerConfig,
    RiskStressTestFramework,
    StressReplayObservation,
)
from src.agents.strategic.schemas import RiskMode


def test_week3_default_library_contains_all_mandatory_scenarios() -> None:
    scenario_ids = {scenario.scenario_id for scenario in DEFAULT_STRESS_SCENARIOS}

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


def test_week3_capacity_breach_at_3x_forces_2x_hard_cap() -> None:
    framework = RiskStressTestFramework()
    observations = [
        StressReplayObservation(
            scenario_id="rbi_surprise_rate_hike",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=10.0,
            capacity_multiplier=1.0,
        ),
        StressReplayObservation(
            scenario_id="inr_flash_move",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=15.0,
            capacity_multiplier=2.0,
        ),
        StressReplayObservation(
            scenario_id="liquidity_drought",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=31.0,
            capacity_multiplier=3.0,
        ),
        StressReplayObservation(
            scenario_id="gfc_2008",
            protective_mode=RiskMode.CLOSE_ONLY,
            impact_bps=12.0,
        ),
        StressReplayObservation(
            scenario_id="taper_tantrum_2013",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=11.0,
        ),
        StressReplayObservation(
            scenario_id="covid_crash_2020",
            protective_mode=RiskMode.CLOSE_ONLY,
            impact_bps=16.0,
        ),
        StressReplayObservation(
            scenario_id="correlation_inversion",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=0.0,
        ),
        StressReplayObservation(
            scenario_id="frozen_constituent_prices",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=0.0,
        ),
        StressReplayObservation(
            scenario_id="multi_asset_liquidity_vacuum",
            protective_mode=RiskMode.CLOSE_ONLY,
            impact_bps=0.0,
        ),
        StressReplayObservation(
            scenario_id="data_poisoning_feed_freeze",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=0.0,
        ),
    ]

    report = framework.evaluate(observations)

    assert report.passed is False
    assert report.recommended_capacity_cap_multiplier == 2.0
    assert any("impact_cap_breach" in reason for reason in report.failure_reasons)
    capacity_3x = next(item for item in report.capacity_results if item.multiplier == 3.0)
    assert capacity_3x.passed is False


def test_week3_impossible_scenarios_require_safe_mode_and_valid_variance() -> None:
    framework = RiskStressTestFramework(
        config=RiskOverseerConfig(stress_max_snapback_ticks=20)
    )
    observations = [
        StressReplayObservation(
            scenario_id="rbi_surprise_rate_hike",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=10.0,
        ),
        StressReplayObservation(
            scenario_id="inr_flash_move",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=12.0,
        ),
        StressReplayObservation(
            scenario_id="liquidity_drought",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=14.0,
        ),
        StressReplayObservation(
            scenario_id="gfc_2008",
            protective_mode=RiskMode.CLOSE_ONLY,
            impact_bps=10.0,
        ),
        StressReplayObservation(
            scenario_id="taper_tantrum_2013",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=10.0,
        ),
        StressReplayObservation(
            scenario_id="covid_crash_2020",
            protective_mode=RiskMode.CLOSE_ONLY,
            impact_bps=12.0,
        ),
        StressReplayObservation(
            scenario_id="correlation_inversion",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=0.0,
            risk_variance_valid=False,
        ),
        StressReplayObservation(
            scenario_id="frozen_constituent_prices",
            protective_mode=RiskMode.REDUCE_ONLY,
            impact_bps=0.0,
            zero_vs_missing_distinguished=False,
        ),
        StressReplayObservation(
            scenario_id="multi_asset_liquidity_vacuum",
            protective_mode=RiskMode.NORMAL,
            impact_bps=0.0,
            feed_integrity_uncertain=True,
            snapback_ticks=22,
        ),
        StressReplayObservation(
            scenario_id="data_poisoning_feed_freeze",
            protective_mode=RiskMode.NORMAL,
            impact_bps=0.0,
            feed_integrity_uncertain=True,
        ),
    ]

    report = framework.evaluate(observations)

    assert report.passed is False
    assert "correlation_inversion:invalid_risk_variance" in report.failure_reasons
    assert "frozen_constituent_prices:zero_missing_confused" in report.failure_reasons
    assert "multi_asset_liquidity_vacuum:feed_integrity_safe_mode_missing" in report.failure_reasons
    assert "data_poisoning_feed_freeze:feed_integrity_safe_mode_missing" in report.failure_reasons
    assert report.snapback_alert_scenarios == ("multi_asset_liquidity_vacuum",)


def test_week3_quarterly_review_template_captures_actions() -> None:
    framework = RiskStressTestFramework()
    report = framework.evaluate(
        [
            StressReplayObservation(
                scenario_id="rbi_surprise_rate_hike",
                protective_mode=RiskMode.REDUCE_ONLY,
                impact_bps=10.0,
            )
        ],
        require_full_library=True,
    )

    review = framework.build_quarterly_review(
        report,
        quarter_label="2026-Q2",
        owner="risk_owner",
        reviewer="review_partner",
    )
    markdown = review.to_markdown()

    assert review.passed is False
    assert "Rerun missing scenarios" in markdown
    assert "Investigate and sign off all stress failures" in markdown
    assert "2026-Q2" in markdown
