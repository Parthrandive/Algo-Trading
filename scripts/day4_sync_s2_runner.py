from __future__ import annotations

import argparse
import json
import numbers
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.preprocessing.loader import MacroLoader, MarketLoader
from src.agents.preprocessing.pipeline import PreprocessingPipeline
from src.agents.textual.textual_data_agent import TextualDataAgent
from src.schemas.market_data import CorporateAction

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def parse_trading_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def file_stem_date(path: Path) -> date | None:
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d").date()
    except ValueError:
        return None


def make_snapshot_id(now_utc: datetime | None = None) -> str:
    now = now_utc or datetime.now(UTC)
    return f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}_UTC"


def normalize_ratio(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace("/", ":")
    if not text:
        return None
    if ":" in text:
        left, right = text.split(":", 1)
        try:
            numerator = float(left)
            denominator = float(right)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return None


def bytes_to_gb(value: int) -> float:
    return float(value) / (1024**3)


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.to_datetime(value, utc=True).isoformat()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return to_json_safe(value.tolist())
    return value


def normalize_record_list(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_json_safe(record) for record in records]


def find_market_files_for_day(base_path: Path, trading_day: date) -> list[Path]:
    return sorted(
        path for path in base_path.glob("**/*.parquet") if file_stem_date(path) == trading_day
    )


def find_macro_files_as_of_day(base_path: Path, trading_day: date) -> list[Path]:
    selected: list[Path] = []
    for path in base_path.glob("**/*.parquet"):
        parsed = file_stem_date(path)
        if parsed is not None and parsed <= trading_day:
            selected.append(path)
    return sorted(selected)


def find_corp_files_as_of_day(base_path: Path, trading_day: date) -> list[Path]:
    selected: list[Path] = []
    for path in base_path.glob("**/*.parquet"):
        parsed = file_stem_date(path)
        if parsed is not None and parsed <= trading_day:
            selected.append(path)
    return sorted(selected)


def concat_parquet_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_corp_action_slice(
    corp_paths: list[Path],
    snapshot_id: str,
    cutoff_utc: datetime,
) -> tuple[pd.DataFrame, list[str]]:
    if not corp_paths:
        return pd.DataFrame(), []

    raw_df = concat_parquet_files(corp_paths)
    if raw_df.empty:
        return pd.DataFrame(), []

    validation_errors: list[str] = []
    validated_rows: list[dict[str, Any]] = []

    for row in raw_df.to_dict(orient="records"):
        try:
            validated = CorporateAction.model_validate(row)
            normalized = validated.model_dump(mode="json")
            normalized["action_type"] = str(normalized.get("action_type", "")).upper()
            normalized["ratio"] = normalize_ratio(normalized.get("ratio"))
            validated_rows.append(normalized)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(str(exc))

    if not validated_rows:
        return pd.DataFrame(), validation_errors

    corp_df = pd.DataFrame(validated_rows)
    corp_df["ex_date"] = pd.to_datetime(corp_df["ex_date"], utc=True)
    corp_df = corp_df[corp_df["ex_date"] <= cutoff_utc]
    corp_df["dataset_snapshot_id"] = snapshot_id
    corp_df = corp_df.sort_values(["symbol", "ex_date"]).reset_index(drop=True)
    return corp_df, validation_errors


def resolve_textual_silver_paths(trading_day: date) -> tuple[Path, Path]:
    year = f"{trading_day.year:04d}"
    month = f"{trading_day.month:02d}"
    base = PROJECT_ROOT / "data" / "silver" / "text"
    canonical_path = base / "canonical" / year / month / f"{trading_day.isoformat()}.parquet"
    sidecar_path = base / "sidecar" / year / month / f"{trading_day.isoformat()}.parquet"
    return canonical_path, sidecar_path


def load_textual_runtime_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "textual_data_agent_runtime_v1.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def blocked_source_id_prefixes(runtime_config: dict[str, Any]) -> tuple[str, ...]:
    # Prefix mapping for source_id formats produced by adapters.
    source_prefix_map: dict[str, tuple[str, ...]] = {
        "nse_news": ("nse_news_", "nse_fallback_"),
        "economic_times": ("et_news_",),
        "rbi_reports": ("rbi_pr_", "rbi_rss_", "rbi_report_", "rbi_rss_fallback_"),
        "earnings_transcripts": ("earnings_pdf_",),
        "x_posts": ("x_post_",),
    }
    blocked_sources = {
        str(entry.get("source_name", ""))
        for entry in runtime_config.get("source_allowlist", [])
        if not bool(entry.get("allowed", False))
    }
    prefixes: list[str] = []
    for source_name in blocked_sources:
        prefixes.extend(source_prefix_map.get(source_name, (f"{source_name}_",)))
    return tuple(prefixes)


def filter_blocked_text_sources(
    canonical_df: pd.DataFrame,
    sidecar_df: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not prefixes:
        return canonical_df, sidecar_df, 0

    removed = 0
    filtered_canonical = canonical_df
    filtered_sidecar = sidecar_df

    if not canonical_df.empty and "source_id" in canonical_df.columns:
        canonical_mask = ~canonical_df["source_id"].astype(str).str.startswith(prefixes)
        removed += int((~canonical_mask).sum())
        filtered_canonical = canonical_df.loc[canonical_mask].reset_index(drop=True)
    if not sidecar_df.empty and "source_id" in sidecar_df.columns:
        sidecar_mask = ~sidecar_df["source_id"].astype(str).str.startswith(prefixes)
        removed += int((~sidecar_mask).sum())
        filtered_sidecar = sidecar_df.loc[sidecar_mask].reset_index(drop=True)

    return filtered_canonical, filtered_sidecar, removed


def build_textual_replay_slice(trading_day: date) -> dict[str, Any]:
    day_start_utc = datetime.combine(trading_day, time.min, tzinfo=UTC)
    day_end_utc = datetime.combine(trading_day, time.max, tzinfo=UTC)
    runtime_config = load_textual_runtime_config()
    blocked_prefixes = blocked_source_id_prefixes(runtime_config)

    canonical_path, sidecar_path = resolve_textual_silver_paths(trading_day)
    if canonical_path.exists() and sidecar_path.exists():
        canonical_df = pd.read_parquet(canonical_path)
        sidecar_df = pd.read_parquet(sidecar_path)
        canonical_df, sidecar_df, removed_rows = filter_blocked_text_sources(
            canonical_df, sidecar_df, blocked_prefixes
        )
        if removed_rows > 0:
            canonical_df.to_parquet(canonical_path, index=False)
            sidecar_df.to_parquet(sidecar_path, index=False)
            source_mode = "persisted_silver_artifacts_filtered"
        else:
            source_mode = "persisted_silver_artifacts"
        batch_dict = {
            "canonical_records": normalize_record_list(canonical_df.to_dict(orient="records")),
            "sidecar_records": normalize_record_list(sidecar_df.to_dict(orient="records")),
        }
    else:
        agent = TextualDataAgent.from_default_components()
        batch = agent.run_once(as_of_utc=day_end_utc)

        canonical_records = []
        for record in batch.canonical_records:
            ts = pd.to_datetime(getattr(record, "timestamp"), utc=True)
            if day_start_utc <= ts <= day_end_utc:
                canonical_records.append(record)

        sidecar_records = []
        for record in batch.sidecar_records:
            ts = pd.to_datetime(record.ingestion_timestamp_utc, utc=True)
            if day_start_utc <= ts <= day_end_utc:
                sidecar_records.append(record)

        batch_dict = {
            "canonical_records": normalize_record_list(
                [record.model_dump(mode="json") for record in canonical_records]
            ),
            "sidecar_records": normalize_record_list(
                [record.model_dump(mode="json") for record in sidecar_records]
            ),
        }

        canonical_df = pd.DataFrame(batch_dict["canonical_records"])
        sidecar_df = pd.DataFrame(batch_dict["sidecar_records"])
        canonical_df, sidecar_df, removed_rows = filter_blocked_text_sources(
            canonical_df, sidecar_df, blocked_prefixes
        )
        batch_dict = {
            "canonical_records": normalize_record_list(canonical_df.to_dict(orient="records")),
            "sidecar_records": normalize_record_list(sidecar_df.to_dict(orient="records")),
        }
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_df.to_parquet(canonical_path, index=False)
        sidecar_df.to_parquet(sidecar_path, index=False)
        source_mode = "live_adapter_fetch_filtered" if removed_rows > 0 else "live_adapter_fetch"

    canonical_records = batch_dict["canonical_records"]
    sidecar_records = batch_dict["sidecar_records"]

    sidecar_type_checks: list[dict[str, Any]] = []
    for sidecar in sidecar_records:
        confidence = sidecar.get("confidence")
        ttl_seconds = sidecar.get("ttl_seconds")
        manipulation_risk = sidecar.get("manipulation_risk_score")
        sidecar_type_checks.append(
            {
                "source_id": str(sidecar.get("source_id", "")),
                "confidence_is_float": isinstance(confidence, numbers.Real)
                and not isinstance(confidence, bool),
                "ttl_seconds_is_int": isinstance(ttl_seconds, numbers.Integral)
                and not isinstance(ttl_seconds, bool),
                "manipulation_risk_score_is_float": isinstance(manipulation_risk, numbers.Real)
                and not isinstance(manipulation_risk, bool),
            }
        )

    all_sidecar_types_valid = all(
        check["confidence_is_float"]
        and check["ttl_seconds_is_int"]
        and check["manipulation_risk_score_is_float"]
        for check in sidecar_type_checks
    )

    return {
        "canonical_records": canonical_records,
        "sidecar_records": sidecar_records,
        "batch_dict": batch_dict,
        "sidecar_type_checks": sidecar_type_checks,
        "all_sidecar_types_valid": all_sidecar_types_valid,
        "source_mode": source_mode,
        "blocked_source_id_prefixes": list(blocked_prefixes),
        "canonical_silver_path": str(canonical_path),
        "sidecar_silver_path": str(sidecar_path),
    }


def normalize_dtype(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int64"
    if pd.api.types.is_float_dtype(dtype):
        return "float64"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    first = non_null.iloc[0]
    if isinstance(first, bool):
        return "bool"
    if isinstance(first, int) and not isinstance(first, bool):
        return "int64"
    if isinstance(first, float):
        return "float64"
    if isinstance(first, (datetime, pd.Timestamp)):
        return "datetime"
    if isinstance(first, str):
        column_name = str(series.name).lower()
        if "timestamp" in column_name or column_name.endswith("_date"):
            parsed = pd.to_datetime(non_null, utc=True, errors="coerce", format="mixed")
            if len(parsed) > 0 and float(parsed.notna().mean()) >= 0.95:
                return "datetime"
        return "string"
    if isinstance(first, list):
        return "array"
    if isinstance(first, dict):
        return "object"
    return "string"


def schema_map(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    return {column: normalize_dtype(df[column]) for column in df.columns}


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_day4_sync(
    trading_day: date,
    market_path: Path,
    macro_path: Path,
    corp_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    cutoff_utc = datetime.combine(trading_day, time(23, 59, 59), tzinfo=UTC)
    snapshot_id = make_snapshot_id()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    market_files = find_market_files_for_day(market_path, trading_day)
    macro_files = find_macro_files_as_of_day(macro_path, trading_day)
    corp_files = find_corp_files_as_of_day(corp_path, trading_day)

    if not market_files:
        raise RuntimeError(
            f"No market files found for trading day {trading_day.isoformat()} under {market_path}"
        )
    if not macro_files:
        raise RuntimeError(
            f"No macro files found on or before {trading_day.isoformat()} under {macro_path}"
        )

    market_slice_df = concat_parquet_files(market_files)
    macro_slice_df = concat_parquet_files(macro_files)
    corp_slice_df, corp_validation_errors = build_corp_action_slice(
        corp_paths=corp_files,
        snapshot_id=snapshot_id,
        cutoff_utc=cutoff_utc,
    )

    market_slice_path = artifacts_dir / f"market_slice_{trading_day.isoformat()}.parquet"
    macro_slice_path = artifacts_dir / f"macro_slice_asof_{trading_day.isoformat()}.parquet"
    corp_slice_path = artifacts_dir / f"corp_slice_asof_{trading_day.isoformat()}.parquet"

    market_slice_df.to_parquet(market_slice_path, index=False)
    macro_slice_df.to_parquet(macro_slice_path, index=False)
    if not corp_slice_df.empty:
        corp_slice_df.to_parquet(corp_slice_path, index=False)

    market_loader = MarketLoader()
    macro_loader = MacroLoader()

    loaded_market_df = market_loader.load(str(market_slice_path), snapshot_id)
    loaded_macro_df = macro_loader.load(str(macro_slice_path), snapshot_id)

    pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
    gold_a = pipeline.replay_snapshot(
        market_source_path=str(market_slice_path),
        macro_source_path=str(macro_slice_path),
        snapshot_id=snapshot_id,
        cutoff_date=cutoff_utc.isoformat(),
        corporate_action_path=str(corp_slice_path) if corp_slice_path.exists() else None,
    )
    gold_b = pipeline.replay_snapshot(
        market_source_path=str(market_slice_path),
        macro_source_path=str(macro_slice_path),
        snapshot_id=snapshot_id,
        cutoff_date=cutoff_utc.isoformat(),
        corporate_action_path=str(corp_slice_path) if corp_slice_path.exists() else None,
    )
    deterministic_replay = gold_a.output_hash == gold_b.output_hash
    gold_df = pd.DataFrame(gold_a.records)

    textual_slice = build_textual_replay_slice(trading_day)
    canonical_df = pd.DataFrame(textual_slice["batch_dict"]["canonical_records"])
    sidecar_df = pd.DataFrame(textual_slice["batch_dict"]["sidecar_records"])
    if canonical_df.empty and sidecar_df.empty:
        raise RuntimeError(
            "No real textual records were produced for the selected trading day. "
            "Run textual ingestion first or verify feed connectivity."
        )
    canonical_df["dataset_snapshot_id"] = snapshot_id
    sidecar_df["dataset_snapshot_id"] = snapshot_id

    textual_canonical_slice_path = artifacts_dir / f"textual_canonical_{trading_day.isoformat()}.parquet"
    textual_sidecar_slice_path = artifacts_dir / f"textual_sidecar_{trading_day.isoformat()}.parquet"
    canonical_df.to_parquet(textual_canonical_slice_path, index=False)
    sidecar_df.to_parquet(textual_sidecar_slice_path, index=False)

    market_snapshot_value = (
        loaded_market_df["dataset_snapshot_id"].iloc[0]
        if not loaded_market_df.empty and "dataset_snapshot_id" in loaded_market_df
        else None
    )
    macro_snapshot_value = (
        loaded_macro_df["dataset_snapshot_id"].iloc[0]
        if not loaded_macro_df.empty and "dataset_snapshot_id" in loaded_macro_df
        else None
    )
    corp_snapshot_value = (
        corp_slice_df["dataset_snapshot_id"].iloc[0]
        if not corp_slice_df.empty and "dataset_snapshot_id" in corp_slice_df
        else snapshot_id
    )

    snapshot_cross_reference = [
        {
            "stream": "sentinel_ohlcv_silver",
            "dataset_snapshot_id": market_snapshot_value,
            "record_count": int(len(loaded_market_df)),
            "trace_ref": str(market_slice_path),
        },
        {
            "stream": "sentinel_corporate_actions_silver",
            "dataset_snapshot_id": corp_snapshot_value,
            "record_count": int(len(corp_slice_df)),
            "trace_ref": str(corp_slice_path) if corp_slice_path.exists() else "none",
        },
        {
            "stream": "macro_indicator_tables",
            "dataset_snapshot_id": macro_snapshot_value,
            "record_count": int(len(loaded_macro_df)),
            "trace_ref": str(macro_slice_path),
        },
        {
            "stream": "textual_silver_plus_sidecar",
            "dataset_snapshot_id": snapshot_id,
            "record_count": int(len(canonical_df) + len(sidecar_df)),
            "trace_ref": f"{textual_canonical_slice_path}|{textual_sidecar_slice_path}",
        },
        {
            "stream": "preprocessing_gold_features",
            "dataset_snapshot_id": gold_a.input_snapshot_id,
            "record_count": int(len(gold_df)),
            "trace_ref": gold_a.output_hash,
        },
    ]

    snapshot_alignment_pass = all(
        entry.get("dataset_snapshot_id") == snapshot_id for entry in snapshot_cross_reference
    )

    schema_tables: dict[str, pd.DataFrame] = {
        "gold_preprocessing_features": gold_df,
        "macro_indicator_table": loaded_macro_df,
        "text_silver_canonical": canonical_df,
        "text_sidecar_metadata": sidecar_df,
        "corp_actions_table": corp_slice_df,
    }
    schema_maps = {name: schema_map(df) for name, df in schema_tables.items()}

    pairwise_compatibility: list[dict[str, Any]] = []
    table_names = list(schema_maps.keys())
    for index, left_name in enumerate(table_names):
        for right_name in table_names[index + 1 :]:
            left_map = schema_maps[left_name]
            right_map = schema_maps[right_name]
            shared_columns = sorted(set(left_map.keys()).intersection(right_map.keys()))
            conflicts = [
                column
                for column in shared_columns
                if left_map.get(column) != right_map.get(column)
            ]
            pairwise_compatibility.append(
                {
                    "left_table": left_name,
                    "right_table": right_name,
                    "shared_column_count": len(shared_columns),
                    "conflict_columns": conflicts,
                    "status": "compatible" if not conflicts else "conflict",
                }
            )

    schema_conflict_count = sum(
        1 for item in pairwise_compatibility if item["status"] == "conflict"
    )

    textual_canonical_bytes = len(
        json.dumps(textual_slice["batch_dict"]["canonical_records"]).encode("utf-8")
    )
    textual_sidecar_bytes = len(
        json.dumps(textual_slice["batch_dict"]["sidecar_records"]).encode("utf-8")
    )
    gold_payload_bytes = len(json.dumps(gold_a.records, default=str).encode("utf-8"))

    current_volume = {
        "sentinel_ohlcv_gb": bytes_to_gb(market_slice_path.stat().st_size),
        "sentinel_corp_actions_gb": bytes_to_gb(corp_slice_path.stat().st_size)
        if corp_slice_path.exists()
        else 0.0,
        "macro_indicator_gb": bytes_to_gb(macro_slice_path.stat().st_size),
        "text_canonical_gb": bytes_to_gb(textual_canonical_bytes),
        "text_sidecar_gb": bytes_to_gb(textual_sidecar_bytes),
        "preprocessing_gold_gb": bytes_to_gb(gold_payload_bytes),
    }
    current_volume["total_input_streams_gb"] = (
        current_volume["sentinel_ohlcv_gb"]
        + current_volume["sentinel_corp_actions_gb"]
        + current_volume["macro_indicator_gb"]
        + current_volume["text_canonical_gb"]
        + current_volume["text_sidecar_gb"]
    )

    roadmap_targets = [
        {
            "checkpoint": "Q2 2026",
            "target_total_input_gb_per_day": 1.0,
            "focus": "Stabilize replay determinism and complete partner text replay automation.",
        },
        {
            "checkpoint": "Q3 2026",
            "target_total_input_gb_per_day": 5.0,
            "focus": "Scale to full NSE universe and add textual partner production feed.",
        },
        {
            "checkpoint": "Q4 2026",
            "target_total_input_gb_per_day": 20.0,
            "focus": "Introduce warm-tier compaction, catalog partitioning, and retention automation.",
        },
        {
            "checkpoint": "Q1 2027",
            "target_total_input_gb_per_day": 75.0,
            "focus": "Phase 2 throughput readiness with replay SLA and quarterly backfill drills.",
        },
    ]

    gold_files_exist = any((PROJECT_ROOT / "data" / "gold").glob("**/*"))
    defects: list[dict[str, Any]] = []
    if corp_validation_errors:
        defects.append(
            {
                "id": "DEF-001",
                "severity": "Medium",
                "title": "Corporate action records failed strict validation",
                "status": "Open",
                "owner": "Data Platform",
                "detail": f"{len(corp_validation_errors)} row(s) failed schema validation during replay prep.",
            }
        )
    if not gold_files_exist:
        defects.append(
            {
                "id": "DEF-002",
                "severity": "Medium",
                "title": "Gold replay output is in-memory only",
                "status": "Open",
                "owner": "Preprocessing",
                "detail": "Preprocessing replay returns TransformOutput hash and records but does not persist to data/gold.",
            }
        )
    local_checks_pass = (
        deterministic_replay
        and snapshot_alignment_pass
        and textual_slice["all_sidecar_types_valid"]
        and schema_conflict_count == 0
    )
    decision = "GO_WITH_CONDITIONS" if local_checks_pass else "BLOCKED"

    result = {
        "trading_day": trading_day.isoformat(),
        "cutoff_utc": cutoff_utc.isoformat(),
        "dataset_snapshot_id": snapshot_id,
        "deterministic_replay": deterministic_replay,
        "gold_hash": gold_a.output_hash,
        "snapshot_alignment_pass": snapshot_alignment_pass,
        "snapshot_cross_reference": snapshot_cross_reference,
        "schema_maps": schema_maps,
        "pairwise_compatibility": pairwise_compatibility,
        "schema_conflict_count": schema_conflict_count,
        "sidecar_type_checks": textual_slice["sidecar_type_checks"],
        "all_sidecar_types_valid": textual_slice["all_sidecar_types_valid"],
        "volumes_gb": current_volume,
        "roadmap_targets": roadmap_targets,
        "defects": defects,
        "decision": decision,
        "paths": {
            "market_slice": str(market_slice_path),
            "macro_slice": str(macro_slice_path),
            "corp_slice": str(corp_slice_path) if corp_slice_path.exists() else "",
            "textual_canonical_slice": str(textual_canonical_slice_path),
            "textual_sidecar_slice": str(textual_sidecar_slice_path),
        },
        "textual_source_mode": textual_slice["source_mode"],
    }
    return result


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trading_day = result["trading_day"]

    replay_report_path = output_dir / f"cross_agent_replay_report_{trading_day}.md"
    schema_matrix_path = output_dir / f"schema_compatibility_matrix_{trading_day}.md"
    roadmap_path = output_dir / "data_scale_roadmap_v1.md"
    decision_log_path = output_dir / f"s2_decision_log_{trading_day}.md"
    defect_tracker_path = output_dir / "defect_tracker_day4_sync_s2.md"
    json_summary_path = output_dir / f"sync_s2_summary_{trading_day}.json"

    snapshot_rows = [
        [
            entry["stream"],
            str(entry["dataset_snapshot_id"]),
            str(entry["record_count"]),
            entry["trace_ref"],
        ]
        for entry in result["snapshot_cross_reference"]
    ]
    snapshot_table = markdown_table(
        snapshot_rows,
        headers=["Stream", "dataset_snapshot_id", "Record Count", "Trace Reference"],
    )

    sidecar_rows = [
        [
            check["source_id"],
            str(check["confidence_is_float"]),
            str(check["ttl_seconds_is_int"]),
            str(check["manipulation_risk_score_is_float"]),
        ]
        for check in result["sidecar_type_checks"]
    ]
    sidecar_table = markdown_table(
        sidecar_rows,
        headers=[
            "source_id",
            "confidence float",
            "ttl_seconds int",
            "manipulation_risk_score float",
        ],
    )

    replay_report = "\n".join(
        [
            f"# Cross-Agent Replay Report (Day 4 Sync S2) - {trading_day}",
            "",
            f"- **Run date (UTC)**: {datetime.now(UTC).isoformat()}",
            f"- **Trading day replayed**: {trading_day}",
            f"- **Cutoff**: {result['cutoff_utc']}",
            f"- **dataset_snapshot_id**: `{result['dataset_snapshot_id']}`",
            f"- **Preprocessing Gold hash**: `{result['gold_hash']}`",
            f"- **Deterministic replay (A == B)**: `{result['deterministic_replay']}`",
            "",
            "## Stream Coverage",
            snapshot_table,
            "",
            "## Snapshot ID Cross-Reference",
            f"- **Alignment pass**: `{result['snapshot_alignment_pass']}`",
            "",
            "## Text Sidecar Metadata Validation",
            sidecar_table,
            "",
            f"- **All required sidecar fields typed correctly**: `{result['all_sidecar_types_valid']}`",
            f"- **Textual replay source mode**: `{result.get('textual_source_mode', 'unknown')}`",
            "",
            "## Status",
            f"- **S2 decision status**: `{result['decision']}`",
            "- **Note**: Textual slice is generated from live/cached adapters and persisted in artifact parquet files.",
            "",
        ]
    )
    replay_report_path.write_text(replay_report, encoding="utf-8")

    schema_rows = []
    for entry in result["pairwise_compatibility"]:
        schema_rows.append(
            [
                entry["left_table"],
                entry["right_table"],
                str(entry["shared_column_count"]),
                ", ".join(entry["conflict_columns"]) if entry["conflict_columns"] else "-",
                entry["status"],
            ]
        )
    schema_table = markdown_table(
        schema_rows,
        headers=[
            "Left Table",
            "Right Table",
            "Shared Columns",
            "Conflict Columns",
            "Status",
        ],
    )

    table_schema_sections: list[str] = []
    for table_name, mapping in result["schema_maps"].items():
        rows = [[column, dtype] for column, dtype in sorted(mapping.items())]
        section_table = markdown_table(rows, headers=["Column", "Type"]) if rows else "_Empty_"
        table_schema_sections.extend([f"### {table_name}", section_table, ""])

    schema_doc = "\n".join(
        [
            f"# Schema Compatibility Matrix - {trading_day}",
            "",
            f"- **dataset_snapshot_id**: `{result['dataset_snapshot_id']}`",
            f"- **Conflict pair count**: `{result['schema_conflict_count']}`",
            "",
            "## Pairwise Matrix",
            schema_table,
            "",
            "## Per-Table Field Maps",
            "",
            *table_schema_sections,
        ]
    )
    schema_matrix_path.write_text(schema_doc, encoding="utf-8")

    volumes = result["volumes_gb"]
    current_rows = [
        ["Sentinel OHLCV", f"{volumes['sentinel_ohlcv_gb']:.8f}"],
        ["Sentinel Corporate Actions", f"{volumes['sentinel_corp_actions_gb']:.8f}"],
        ["Macro Indicators", f"{volumes['macro_indicator_gb']:.8f}"],
        ["Text Canonical", f"{volumes['text_canonical_gb']:.8f}"],
        ["Text Sidecar", f"{volumes['text_sidecar_gb']:.8f}"],
        ["Preprocessing Gold (derived output)", f"{volumes['preprocessing_gold_gb']:.8f}"],
        ["Total Input Streams", f"{volumes['total_input_streams_gb']:.8f}"],
    ]
    current_table = markdown_table(current_rows, headers=["Stream", "Current Daily Volume (GB)"])

    roadmap_rows = [
        [
            target["checkpoint"],
            f"{target['target_total_input_gb_per_day']:.2f}",
            target["focus"],
        ]
        for target in result["roadmap_targets"]
    ]
    roadmap_table = markdown_table(
        roadmap_rows,
        headers=["Checkpoint", "Target Total Input GB/Day", "Focus"],
    )

    roadmap_doc = "\n".join(
        [
            "# Data Scale Roadmap v1",
            "",
            f"- **Baseline trading day**: {trading_day}",
            f"- **Snapshot**: `{result['dataset_snapshot_id']}`",
            "",
            "## Current Volume Baseline",
            current_table,
            "",
            "## Phase 2 Targets and Quarterly Checkpoints",
            roadmap_table,
            "",
            "## Assumptions",
            "- Baseline reflects replay slice artifacts generated for Day 4 Sync S2.",
            "- Targets represent scaling milestones for full-universe and partner textual production onboarding.",
            "",
        ]
    )
    roadmap_path.write_text(roadmap_doc, encoding="utf-8")

    defect_rows = [
        [
            defect["id"],
            defect["severity"],
            defect["status"],
            defect["owner"],
            defect["title"],
            defect["detail"],
        ]
        for defect in result["defects"]
    ]
    defect_table = markdown_table(
        defect_rows,
        headers=["ID", "Severity", "Status", "Owner", "Title", "Detail"],
    )
    defect_doc = "\n".join(
        [
            "# Day 4 Sync S2 Defect Tracker",
            "",
            f"- **Trading day**: {trading_day}",
            f"- **Snapshot**: `{result['dataset_snapshot_id']}`",
            "",
            defect_table,
            "",
        ]
    )
    defect_tracker_path.write_text(defect_doc, encoding="utf-8")

    decision_doc = "\n".join(
        [
            f"# S2 Decision Log - {trading_day}",
            "",
            f"- **Decision timestamp (UTC)**: {datetime.now(UTC).isoformat()}",
            f"- **dataset_snapshot_id**: `{result['dataset_snapshot_id']}`",
            f"- **Decision**: `{result['decision']}`",
            "",
            "## Decision Inputs",
            f"- Deterministic replay: `{result['deterministic_replay']}`",
            f"- Snapshot ID alignment across streams: `{result['snapshot_alignment_pass']}`",
            f"- Text sidecar typing checks: `{result['all_sidecar_types_valid']}`",
            f"- Schema conflict pairs: `{result['schema_conflict_count']}`",
            f"- Textual source mode: `{result.get('textual_source_mode', 'unknown')}`",
            "",
            "## Condition Notes",
            "- Gold persistence path is pending; replay evidence currently uses output hash + record payload.",
            "",
            "## Artifact Links",
            f"- Replay report: `{replay_report_path}`",
            f"- Schema matrix: `{schema_matrix_path}`",
            f"- Data scale roadmap v1: `{roadmap_path}`",
            f"- Defect tracker: `{defect_tracker_path}`",
            "",
        ]
    )
    decision_log_path.write_text(decision_doc, encoding="utf-8")

    json_summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Day 4 Sync S2 runner: cross-agent replay, schema matrix, roadmap, and decision log."
    )
    parser.add_argument("--trading-day", default="2026-03-05", help="Trading day in YYYY-MM-DD format")
    parser.add_argument(
        "--market-path",
        default=str(PROJECT_ROOT / "data" / "silver" / "ohlcv"),
        help="Path to Sentinel OHLCV Silver data",
    )
    parser.add_argument(
        "--macro-path",
        default=str(PROJECT_ROOT / "data" / "silver" / "macro"),
        help="Path to Macro Silver indicator tables",
    )
    parser.add_argument(
        "--corp-path",
        default=str(PROJECT_ROOT / "data" / "e2e_test" / "silver" / "corporate_actions"),
        help="Path to Corporate Actions Silver data",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "docs" / "reports" / "day4_sync_s2"),
        help="Directory for Day 4 deliverables",
    )
    args = parser.parse_args()

    trading_day = parse_trading_day(args.trading_day)
    result = run_day4_sync(
        trading_day=trading_day,
        market_path=Path(args.market_path),
        macro_path=Path(args.macro_path),
        corp_path=Path(args.corp_path),
        output_dir=Path(args.output_dir),
    )
    write_outputs(output_dir=Path(args.output_dir), result=result)

    print(json.dumps(
        {
            "status": "ok",
            "decision": result["decision"],
            "trading_day": result["trading_day"],
            "dataset_snapshot_id": result["dataset_snapshot_id"],
            "output_dir": str(Path(args.output_dir)),
        },
        indent=2,
    ))
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
