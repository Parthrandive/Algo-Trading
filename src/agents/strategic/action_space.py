from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.agents.strategic.schemas import StrategicToExecutiveContract, Week2ActionSpaceRecord


def to_week2_record(contract: StrategicToExecutiveContract) -> Week2ActionSpaceRecord:
    return Week2ActionSpaceRecord(
        timestamp=contract.timestamp,
        symbol=contract.symbol,
        policy_id=contract.policy_id,
        policy_type=contract.policy_type.value,
        loop_type=contract.loop_type.value,
        action=contract.action,
        action_size=contract.action_size,
        confidence=contract.confidence,
        observation_id=contract.observation_id,
        snapshot_id=contract.snapshot_id,
        observation_schema_version=contract.observation_schema_version,
        quality_status="pass",
        decision_reason=contract.decision_reason,
    )


def export_week2_action_space(path: Path, actions: Iterable[StrategicToExecutiveContract]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for contract in actions:
            record = to_week2_record(contract)
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")
    return path
