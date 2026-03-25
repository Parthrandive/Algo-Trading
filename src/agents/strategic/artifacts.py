from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agents.strategic.config import RUN_MANIFEST_VERSION


class Phase3RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = RUN_MANIFEST_VERSION
    run_id: str
    phase: str = "phase_3"
    component: str = "strategic"
    started_at_utc: datetime
    finished_at_utc: datetime
    symbols: list[str]
    observation_schema_version: str
    contract_version: str
    export_schema_version: str
    rows_materialized: int = 0
    actions_generated: int = 0
    code_hash: str | None = None
    dataset_snapshot: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def resolve_code_hash() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    sha = proc.stdout.strip()
    return sha or None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_manifest(path: Path, manifest: Phase3RunManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
