from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.agents.strategic.config import WalkForwardConfig
from src.agents.strategic.schemas import RLTrainingRunRecord


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
        train_start=pd.Timestamp(config.train_start, tz="UTC").to_pydatetime(),
        train_end=pd.Timestamp(config.train_end, tz="UTC").to_pydatetime(),
        validation_start=pd.Timestamp(config.validation_start, tz="UTC").to_pydatetime(),
        validation_end=pd.Timestamp(config.validation_end, tz="UTC").to_pydatetime(),
        test_start=pd.Timestamp(config.test_start, tz="UTC").to_pydatetime(),
        test_end=pd.Timestamp(config.test_end, tz="UTC").to_pydatetime(),
        reward_name=reward_name,
        metrics={"status": "foundation_only"},
        params={"training_enabled": False},
        notes="Week 1 foundation scaffold only. No training executed.",
    )
