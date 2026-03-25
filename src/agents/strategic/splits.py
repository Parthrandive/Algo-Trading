from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.agents.strategic.config import WalkForwardConfig
from src.agents.strategic.schemas import RLTrainingRunRecord


def _to_utc_datetime(value: datetime) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def build_walk_forward_mask(frame: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["split"] = "out_of_window"
    train_mask = result["timestamp"].between(config.train_start, config.train_end)
    val_mask = result["timestamp"].between(config.validation_start, config.validation_end)
    test_mask = result["timestamp"].between(config.test_start, config.test_end)
    result.loc[train_mask, "split"] = "train"
    result.loc[val_mask, "split"] = "validation"
    result.loc[test_mask, "split"] = "test"
    return result


def build_planned_training_run(policy_id: str, config: WalkForwardConfig, *, reward_name: str) -> RLTrainingRunRecord:
    return RLTrainingRunRecord(
        policy_id=policy_id,
        started_at=datetime.now(UTC),
        status="planned",
        train_start=_to_utc_datetime(config.train_start),
        train_end=_to_utc_datetime(config.train_end),
        validation_start=_to_utc_datetime(config.validation_start),
        validation_end=_to_utc_datetime(config.validation_end),
        test_start=_to_utc_datetime(config.test_start),
        test_end=_to_utc_datetime(config.test_end),
        reward_name=reward_name,
        metrics={"status": "foundation_only"},
        params={"training_enabled": False},
        notes="Week 1 foundation scaffold only. No training executed.",
    )
