from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

OBSERVATION_SCHEMA_VERSION = "1.0"
STRATEGIC_EXEC_CONTRACT_VERSION = "strat_exec_v1"
WEEK2_ACTION_EXPORT_VERSION = "week2_action_space_v1"
RUN_MANIFEST_VERSION = "phase3_run_manifest_v1"
STRATEGIC_POLICY_SNAPSHOT_VERSION = "phase3_policy_snapshot_v1"
STRATEGIC_ENSEMBLE_VERSION = "phase3_ensemble_v1"
STRATEGIC_EXECUTION_AUDIT_VERSION = "phase3_execution_audit_v1"

# Teacher inference is explicitly restricted to offline/slow-loop paths.
TEACHER_POLICY_TYPE = "teacher"
STUDENT_POLICY_TYPE = "student"
ALLOWED_TEACHER_LOOP_TYPES = {"offline", "slow"}

DEFAULT_SENTIMENT_LANE = "slow"
DEFAULT_MAX_SIGNAL_STALENESS = timedelta(hours=6)
DEFAULT_ACTION_EXPORT_DIR = Path("data/reports/phase3/week1")
DEFAULT_ACTION_EXPORT_FILE = "week2_action_space_export.jsonl"
DEFAULT_RUN_MANIFEST_FILE = "run_manifest.json"


@dataclass(frozen=True)
class StrategicAssemblerConfig:
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    sentiment_lane: str = DEFAULT_SENTIMENT_LANE
    max_signal_staleness: timedelta = DEFAULT_MAX_SIGNAL_STALENESS
    allow_market_level_sentiment_fallback: bool = True


@dataclass(frozen=True)
class EnvironmentCostConfig:
    initial_cash: float = 1_000_000.0
    brokerage_bps: float = 2.0
    slippage_bps: float = 5.0
    impact_bps_per_unit: float = 0.1


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: float = 2.0
    validation_months: float = 6.0
    test_months: float = 6.0
    min_train_rows: int = 500
    # Foundation defaults for testing/splits
    train_start: datetime = field(default_factory=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc))
    train_end: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    validation_start: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    validation_end: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    test_start: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    test_end: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))


@dataclass(frozen=True)
class PolicyFoundationConfig:
    policy_id: str
    algorithm: str
    version: str = "1.0"
    action_space: str = "continuous"
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    checkpoint_root: Path = field(default_factory=lambda: Path("/tmp/checkpoints"))


@dataclass(frozen=True)
class EnsembleConfig:
    temperature: float = 0.35
    diversity_penalty: float = 0.15
    minimum_policy_weight: float = 0.05
    confidence_floor: float = 0.05
    consensus_boost: float = 0.10
    crisis_confidence_haircut: float = 0.20
    divergence_hold_confidence_cap: float = 0.35
    version: str = STRATEGIC_ENSEMBLE_VERSION


@dataclass(frozen=True)
class DistillationConfig:
    student_policy_id: str = "strategic_student_v1"
    student_version: str = "0.1.0"
    agreement_threshold: float = 0.85
    crisis_agreement_threshold: float = 0.90
    drift_alert_threshold: float = 0.12
    fast_loop_latency_p99_ms: float = 8.0
    fast_loop_degrade_ms: float = 10.0


@dataclass(frozen=True)
class PolicySnapshotConfig:
    refresh_interval: timedelta = timedelta(minutes=10)
    snapshot_ttl: timedelta = timedelta(minutes=15)
    stale_after: timedelta = timedelta(minutes=12)
    required_quality_status: str = "pass"


@dataclass(frozen=True)
class DeliberationConfig:
    target_latency_ms: int = 250
    bypass_volatility_threshold: float = 2.0
    max_policy_search_ms: int = 500
    refresh_on_improvement_only: bool = True


@dataclass(frozen=True)
class PortfolioConfig:
    normal_notional_cap: float = 50_000_000.0
    validated_notional_cap: float = 100_000_000.0
    max_exposure_per_symbol: float = 0.15
    max_leverage: float = 1.0
    max_single_position_fraction: float = 0.15
    max_sector_fraction: float = 0.30
    max_correlation: float = 0.85
    uncertainty_size_floor: float = 0.10
    uncertainty_size_ceiling: float = 1.00
    participation_limit_large_cap: float = 0.05
    participation_limit_mid_liquidity: float = 0.03
    participation_limit_fx: float = 0.08
    participation_limit_gold: float = 0.04
    high_volatility_cap_fraction: float = 0.40
    extreme_volatility_cap_fraction: float = 0.15


@dataclass(frozen=True)
class ExecutionConfig:
    default_order_type: str = "LIMIT"
    max_partial_fill_count: int = 5
    slippage_alert_bps: float = 20.0
    routing_health_failure_limit: int = 3
    circuit_breaker_reduce_factor: float = 0.50
    audit_version: str = STRATEGIC_EXECUTION_AUDIT_VERSION


@dataclass(frozen=True)
class ImpactMonitorConfig:
    cooldown_seconds: int = 300
    hysteresis_clean_fills: int = 2
    reduction_step_fraction: float = 0.25
    min_size_multiplier: float = 0.10
    slippage_alert_bps: float = 20.0


@dataclass(frozen=True)
class RiskBudgetConfig:
    false_trigger_acceptance_limit: float = 0.20
    false_trigger_rolling_days: int = 30
    normal_cap: float = 1.00
    elevated_cap: float = 0.70
    high_cap: float = 0.40
    extreme_cap: float = 0.15


@dataclass(frozen=True)
class OrderBookFeatureConfig:
    top_n_levels: int = 5
    staleness_threshold_seconds: int = 2
    degraded_downweight: float = 0.50


@dataclass(frozen=True)
class LatencyDisciplineConfig:
    p99_target_ms: float = 8.0
    degrade_threshold_ms: float = 10.0
    restore_consecutive_windows: int = 3
    regression_guard_ms: float = 0.5


@dataclass(frozen=True)
class PromotionGateConfig:
    min_crisis_agreement: float = 0.80
    require_latency_gate: bool = True
    require_rollback_readiness: bool = True
    require_non_regression: bool = True


@dataclass(frozen=True)
class XAIConfig:
    top_k_features: int = 5
    min_coverage: float = 0.80
