from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.agents.risk_overseer.config import RiskOverseerConfig
from src.agents.strategic.schemas import RiskMode
from src.agents.strategic.week4 import StressScenarioResult, StressTestEngine


class StressScenarioCategory(str, Enum):
    HISTORICAL = "historical"
    SYNTHETIC = "synthetic"
    IMPOSSIBLE = "impossible"


class StressInjectorType(str, Enum):
    REPLAY = "replay"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class StressScenarioDefinition:
    scenario_id: str
    category: StressScenarioCategory
    injector: StressInjectorType
    expected_min_mode: RiskMode
    description: str
    labels: tuple[str, ...] = ()
    replay_source: str | None = None


@dataclass(frozen=True)
class StressReplayObservation:
    scenario_id: str
    protective_mode: RiskMode
    impact_bps: float
    capacity_multiplier: float = 1.0
    snapback_ticks: int = 0
    crashed: bool = False
    data_corruption: bool = False
    zero_vs_missing_distinguished: bool = True
    feed_integrity_uncertain: bool = False
    risk_variance_valid: bool = True
    notes: str | None = None


@dataclass(frozen=True)
class CapacityStressResult:
    multiplier: float
    scenarios_run: int
    max_impact_bps: float
    passed: bool


@dataclass(frozen=True)
class RiskStressTestReport:
    passed: bool
    library_version: str
    scenarios_run: int
    missing_scenarios: tuple[str, ...]
    unknown_scenarios: tuple[str, ...]
    failure_count: int
    failure_reasons: tuple[str, ...]
    snapback_alert_scenarios: tuple[str, ...]
    recommended_capacity_cap_multiplier: float
    capacity_results: tuple[CapacityStressResult, ...]


@dataclass(frozen=True)
class QuarterlyStressReview:
    quarter_label: str
    owner: str
    reviewer: str
    library_version: str
    passed: bool
    scenarios_run: int
    recommended_capacity_cap_multiplier: float
    missing_scenarios: tuple[str, ...]
    required_actions: tuple[str, ...]

    def to_markdown(self) -> str:
        actions = "\n".join(f"- {item}" for item in self.required_actions) or "- No action required."
        missing = ", ".join(self.missing_scenarios) if self.missing_scenarios else "None"
        return (
            f"# Quarterly Stress Review: {self.quarter_label}\n\n"
            f"- Owner: {self.owner}\n"
            f"- Reviewer: {self.reviewer}\n"
            f"- Library Version: {self.library_version}\n"
            f"- Passed: {'YES' if self.passed else 'NO'}\n"
            f"- Scenarios Run: {self.scenarios_run}\n"
            f"- Missing Scenarios: {missing}\n"
            f"- Recommended Capacity Cap: {self.recommended_capacity_cap_multiplier:.1f}x\n\n"
            "## Required Actions\n"
            f"{actions}\n"
        )


DEFAULT_STRESS_SCENARIOS: tuple[StressScenarioDefinition, ...] = (
    StressScenarioDefinition(
        scenario_id="rbi_surprise_rate_hike",
        category=StressScenarioCategory.HISTORICAL,
        injector=StressInjectorType.REPLAY,
        expected_min_mode=RiskMode.REDUCE_ONLY,
        description="Replay RBI surprise rate-hike windows for INR and rates shock behavior.",
        labels=("rbi", "rates", "macro"),
        replay_source="historical_analogue",
    ),
    StressScenarioDefinition(
        scenario_id="inr_flash_move",
        category=StressScenarioCategory.HISTORICAL,
        injector=StressInjectorType.REPLAY,
        expected_min_mode=RiskMode.REDUCE_ONLY,
        description="Replay INR flash-move windows and hedge trigger continuity.",
        labels=("fx", "flash_move"),
        replay_source="historical_analogue",
    ),
    StressScenarioDefinition(
        scenario_id="liquidity_drought",
        category=StressScenarioCategory.SYNTHETIC,
        injector=StressInjectorType.SYNTHETIC,
        expected_min_mode=RiskMode.REDUCE_ONLY,
        description="Synthetic liquidity drought to validate impact and participation controls.",
        labels=("liquidity", "impact"),
    ),
    StressScenarioDefinition(
        scenario_id="gfc_2008",
        category=StressScenarioCategory.HISTORICAL,
        injector=StressInjectorType.REPLAY,
        expected_min_mode=RiskMode.CLOSE_ONLY,
        description="Replay 2008 GFC crisis slices for survival and drawdown containment.",
        labels=("2008", "gfc"),
        replay_source="historical_analogue",
    ),
    StressScenarioDefinition(
        scenario_id="taper_tantrum_2013",
        category=StressScenarioCategory.HISTORICAL,
        injector=StressInjectorType.REPLAY,
        expected_min_mode=RiskMode.REDUCE_ONLY,
        description="Replay 2013 taper tantrum regime transition and macro shock response.",
        labels=("2013", "macro", "regime"),
        replay_source="historical_analogue",
    ),
    StressScenarioDefinition(
        scenario_id="covid_crash_2020",
        category=StressScenarioCategory.HISTORICAL,
        injector=StressInjectorType.REPLAY,
        expected_min_mode=RiskMode.CLOSE_ONLY,
        description="Replay Feb-Apr 2020 COVID crash correlation and protective mode behavior.",
        labels=("2020", "covid", "correlation"),
        replay_source="historical_analogue",
    ),
    StressScenarioDefinition(
        scenario_id="correlation_inversion",
        category=StressScenarioCategory.IMPOSSIBLE,
        injector=StressInjectorType.SYNTHETIC,
        expected_min_mode=RiskMode.REDUCE_ONLY,
        description="Impossible correlation inversion drill for risk-model coherence.",
        labels=("impossible", "correlation"),
    ),
    StressScenarioDefinition(
        scenario_id="frozen_constituent_prices",
        category=StressScenarioCategory.IMPOSSIBLE,
        injector=StressInjectorType.SYNTHETIC,
        expected_min_mode=RiskMode.REDUCE_ONLY,
        description="Frozen constituent prices drill to separate zero-volume from missing data.",
        labels=("impossible", "market_data"),
    ),
    StressScenarioDefinition(
        scenario_id="multi_asset_liquidity_vacuum",
        category=StressScenarioCategory.IMPOSSIBLE,
        injector=StressInjectorType.SYNTHETIC,
        expected_min_mode=RiskMode.CLOSE_ONLY,
        description="Simultaneous multi-asset liquidity vacuum requiring safe mode activation.",
        labels=("impossible", "liquidity"),
    ),
    StressScenarioDefinition(
        scenario_id="data_poisoning_feed_freeze",
        category=StressScenarioCategory.SYNTHETIC,
        injector=StressInjectorType.SYNTHETIC,
        expected_min_mode=RiskMode.REDUCE_ONLY,
        description="Feed freeze and data poisoning drill for degraded-mode activation.",
        labels=("feed_freeze", "data_quality"),
    ),
)


class RiskStressTestFramework:
    """
    Week 3 automation for the Risk Overseer's stress library.

    The framework reuses the strategic stress assertions while layering on
    mandatory library coverage, impossible-scenario checks, and 3x capacity
    hard-cap enforcement.
    """

    def __init__(
        self,
        *,
        config: RiskOverseerConfig | None = None,
        stress_engine: StressTestEngine | None = None,
        scenario_library: Sequence[StressScenarioDefinition] | None = None,
    ) -> None:
        self.config = config or RiskOverseerConfig()
        self.stress_engine = stress_engine or StressTestEngine(
            capacity_impact_cap_bps=self.config.capacity_impact_cap_bps
        )
        self.scenario_library = tuple(scenario_library or DEFAULT_STRESS_SCENARIOS)

    def evaluate(
        self,
        observations: Sequence[StressReplayObservation],
        *,
        require_full_library: bool = True,
    ) -> RiskStressTestReport:
        library_by_id = {scenario.scenario_id: scenario for scenario in self.scenario_library}
        observed_ids = {item.scenario_id for item in observations}

        missing_scenarios = ()
        if require_full_library:
            missing_scenarios = tuple(sorted(set(library_by_id) - observed_ids))
        unknown_scenarios = tuple(sorted(observed_ids - set(library_by_id)))

        base_results = tuple(self._to_week4_result(item, library_by_id) for item in observations)
        base_report = self.stress_engine.evaluate(base_results)

        extra_failures: list[str] = []
        snapback_alerts: list[str] = []
        for item in observations:
            scenario = library_by_id.get(item.scenario_id)
            if item.feed_integrity_uncertain and _mode_rank(item.protective_mode) < _mode_rank(RiskMode.REDUCE_ONLY):
                extra_failures.append(f"{item.scenario_id}:feed_integrity_safe_mode_missing")
            if item.scenario_id == "correlation_inversion" and not item.risk_variance_valid:
                extra_failures.append(f"{item.scenario_id}:invalid_risk_variance")
            if item.snapback_ticks > self.config.stress_max_snapback_ticks:
                snapback_alerts.append(item.scenario_id)
            if scenario is not None and scenario.category == StressScenarioCategory.IMPOSSIBLE:
                if item.crashed:
                    extra_failures.append(f"{item.scenario_id}:impossible_scenario_crash")
                if item.data_corruption:
                    extra_failures.append(f"{item.scenario_id}:impossible_scenario_data_corruption")

        failure_reasons = list(base_report.failure_reasons)
        failure_reasons.extend(extra_failures)
        failure_reasons.extend(f"missing_scenario:{item}" for item in missing_scenarios)
        failure_reasons.extend(f"unknown_scenario:{item}" for item in unknown_scenarios)
        failure_reasons = list(dict.fromkeys(failure_reasons))

        capacity_results = self._build_capacity_results(observations)
        recommended_capacity_cap = self._recommended_capacity_cap(observations)

        return RiskStressTestReport(
            passed=len(failure_reasons) == 0,
            library_version=self.config.stress_scenario_library_version,
            scenarios_run=len(observations),
            missing_scenarios=missing_scenarios,
            unknown_scenarios=unknown_scenarios,
            failure_count=len(failure_reasons),
            failure_reasons=tuple(failure_reasons),
            snapback_alert_scenarios=tuple(sorted(set(snapback_alerts))),
            recommended_capacity_cap_multiplier=recommended_capacity_cap,
            capacity_results=capacity_results,
        )

    def build_quarterly_review(
        self,
        report: RiskStressTestReport,
        *,
        quarter_label: str,
        owner: str,
        reviewer: str,
    ) -> QuarterlyStressReview:
        actions: list[str] = []
        if report.missing_scenarios:
            actions.append(f"Rerun missing scenarios: {', '.join(report.missing_scenarios)}.")
        if report.snapback_alert_scenarios:
            actions.append(
                "Review slow snapback clips for "
                f"{', '.join(report.snapback_alert_scenarios)} before changing smoothing or exposure rules."
            )
        if report.recommended_capacity_cap_multiplier < max(self.config.capacity_stress_multipliers):
            actions.append(
                f"Enforce a {report.recommended_capacity_cap_multiplier:.1f}x hard capacity cap until 3x impact retest passes."
            )
        if report.failure_reasons:
            actions.append(
                f"Investigate and sign off all stress failures ({len(report.failure_reasons)} findings) before the next gate review."
            )
        if not actions:
            actions.append("No corrective action required; sign and archive this quarter's stress review.")

        return QuarterlyStressReview(
            quarter_label=quarter_label,
            owner=owner,
            reviewer=reviewer,
            library_version=report.library_version,
            passed=report.passed,
            scenarios_run=report.scenarios_run,
            recommended_capacity_cap_multiplier=report.recommended_capacity_cap_multiplier,
            missing_scenarios=report.missing_scenarios,
            required_actions=tuple(actions),
        )

    def _to_week4_result(
        self,
        item: StressReplayObservation,
        library_by_id: dict[str, StressScenarioDefinition],
    ) -> StressScenarioResult:
        scenario = library_by_id.get(item.scenario_id)
        expected_mode = scenario.expected_min_mode if scenario is not None else RiskMode.REDUCE_ONLY
        return StressScenarioResult(
            scenario_id=item.scenario_id,
            protective_mode=item.protective_mode,
            expected_min_mode=expected_mode,
            crashed=item.crashed,
            data_corruption=item.data_corruption,
            zero_vs_missing_distinguished=item.zero_vs_missing_distinguished,
            snapback_ticks=item.snapback_ticks,
            max_snapback_ticks=self.config.stress_max_snapback_ticks,
            capacity_multiplier=item.capacity_multiplier,
            impact_bps=item.impact_bps,
        )

    def _build_capacity_results(
        self,
        observations: Sequence[StressReplayObservation],
    ) -> tuple[CapacityStressResult, ...]:
        results: list[CapacityStressResult] = []
        for multiplier in self.config.capacity_stress_multipliers:
            matching = [item for item in observations if abs(item.capacity_multiplier - multiplier) < 1e-9]
            max_impact = max((item.impact_bps for item in matching), default=0.0)
            passed = max_impact <= self.config.capacity_impact_cap_bps or multiplier < 3.0
            results.append(
                CapacityStressResult(
                    multiplier=multiplier,
                    scenarios_run=len(matching),
                    max_impact_bps=max_impact,
                    passed=passed,
                )
            )
        return tuple(results)

    def _recommended_capacity_cap(
        self,
        observations: Sequence[StressReplayObservation],
    ) -> float:
        for item in observations:
            if item.capacity_multiplier >= 3.0 and item.impact_bps > self.config.capacity_impact_cap_bps:
                return self.config.capacity_hard_cap_multiplier
        return max(self.config.capacity_stress_multipliers)


def _mode_rank(mode: RiskMode) -> int:
    return {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }[mode]


__all__ = [
    "CapacityStressResult",
    "DEFAULT_STRESS_SCENARIOS",
    "QuarterlyStressReview",
    "RiskStressTestFramework",
    "RiskStressTestReport",
    "StressInjectorType",
    "StressReplayObservation",
    "StressScenarioCategory",
    "StressScenarioDefinition",
]
