from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

OBSERVATION_SCHEMA_VERSION = "1.0"
OBSERVATION_MAPPING_VERSION = "phase3_week1_v1"
DEFAULT_ALIGNMENT_TOLERANCE_SECONDS = 300


@dataclass(frozen=True)
class ObservationAssemblerConfig:
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    mapping_version: str = OBSERVATION_MAPPING_VERSION
    alignment_tolerance_seconds: int = DEFAULT_ALIGNMENT_TOLERANCE_SECONDS
    fill_value: float = 0.0


@dataclass(frozen=True)
class EnvironmentCostConfig:
    brokerage_bps: float = 10.0
    slippage_bps: float = 5.0
    impact_bps_per_unit: float = 2.0
    annualization_factor: int = 252
    initial_cash: float = 1_000_000.0


@dataclass(frozen=True)
class WalkForwardConfig:
    train_start: str = "2019-01-01"
    train_end: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"


@dataclass(frozen=True)
class PolicyFoundationConfig:
    policy_id: str
    algorithm: str
    action_space: str = "continuous"
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    offline_only: bool = True
    is_teacher_policy: bool = True
    checkpoint_root: Path = field(default_factory=lambda: Path("data/models/rl_teachers"))
