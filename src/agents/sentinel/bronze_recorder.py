import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


class BronzeRecorder:
    """
    Persists raw source payloads in append-only JSONL files.
    Partition path: source_id/YYYY-MM-DD/HH/events.jsonl
    """

    def __init__(self, base_dir: str = "data/bronze"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_event(
        self,
        source_id: str,
        payload: dict[str, Any],
        event_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        schema_id: Optional[str] = None,
    ) -> Path:
        ingestion_ts = datetime.now(timezone.utc)
        effective_event_time = event_time or ingestion_ts
        if effective_event_time.tzinfo is None:
            effective_event_time = effective_event_time.replace(tzinfo=timezone.utc)
        else:
            effective_event_time = effective_event_time.astimezone(timezone.utc)

        target_dir = self.base_dir / source_id / effective_event_time.strftime("%Y-%m-%d") / effective_event_time.strftime("%H")
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / "events.jsonl"

        record = {
            "source_id": source_id,
            "symbol": symbol,
            "schema_id": schema_id,
            "event_timestamp_utc": effective_event_time.isoformat(),
            "ingestion_timestamp_utc": ingestion_ts.isoformat(),
            "payload": payload,
        }

        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str))
            f.write("\n")

        return file_path

    def save_batch(
        self,
        source_id: str,
        payloads: Iterable[dict[str, Any]],
        event_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        schema_id: Optional[str] = None,
    ) -> list[Path]:
        written_files: list[Path] = []
        for payload in payloads:
            written_files.append(
                self.save_event(
                    source_id=source_id,
                    payload=payload,
                    event_time=event_time,
                    symbol=symbol,
                    schema_id=schema_id,
                )
            )
        return written_files
