from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
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
