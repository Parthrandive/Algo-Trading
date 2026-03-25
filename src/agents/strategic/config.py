from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

OBSERVATION_SCHEMA_VERSION = "1.0"
STRATEGIC_EXEC_CONTRACT_VERSION = "strat_exec_v1"
WEEK2_ACTION_EXPORT_VERSION = "week2_action_space_v1"
RUN_MANIFEST_VERSION = "phase3_run_manifest_v1"

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
