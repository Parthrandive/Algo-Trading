"""
Day 13 Observability Report
Builds lag/failover/parser/tracing metrics from Day 13 E2E outputs.
"""

from pathlib import Path
import json
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_trace_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    print("Generating Observability Report...")

    data_dir = PROJECT_ROOT / "data" / "e2e_test"
    silver_ohlcv_dir = data_dir / "silver" / "ohlcv"
    silver_corp_dir = data_dir / "silver" / "corporate_actions"
    metrics_file = data_dir / "metrics" / "ingest_metrics.json"
    traces_file = data_dir / "traces" / "ingest_trace.jsonl"

    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "observability_dashboard.md"

    ohlcv_files = list(silver_ohlcv_dir.rglob("*.parquet"))
    corp_files = list(silver_corp_dir.rglob("*.parquet"))
    if not ohlcv_files and not corp_files:
        print("ERROR: No Silver data found. Run test_day13_e2e.py first.")
        sys.exit(1)

    ohlcv_df = pd.concat([pd.read_parquet(file) for file in ohlcv_files], ignore_index=True) if ohlcv_files else pd.DataFrame()
    corp_df = pd.concat([pd.read_parquet(file) for file in corp_files], ignore_index=True) if corp_files else pd.DataFrame()

    lag_max = lag_min = lag_avg = 0.0
    if not ohlcv_df.empty and "timestamp" in ohlcv_df and "ingestion_timestamp_utc" in ohlcv_df:
        ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], utc=True)
        ohlcv_df["ingestion_timestamp_utc"] = pd.to_datetime(ohlcv_df["ingestion_timestamp_utc"], utc=True)
        ohlcv_df["lag_seconds"] = (ohlcv_df["ingestion_timestamp_utc"] - ohlcv_df["timestamp"]).dt.total_seconds()
        lag_max = _safe_float(ohlcv_df["lag_seconds"].max())
        lag_min = _safe_float(ohlcv_df["lag_seconds"].min())
        lag_avg = _safe_float(ohlcv_df["lag_seconds"].mean())

    total_records = int(len(ohlcv_df) + len(corp_df))
    fallback_records = 0
    if not ohlcv_df.empty and "source_type" in ohlcv_df:
        fallback_records += int((ohlcv_df["source_type"] == "fallback_scraper").sum())
    if not corp_df.empty and "source_type" in corp_df:
        fallback_records += int((corp_df["source_type"] == "fallback_scraper").sum())
    fallback_pct = (fallback_records / total_records * 100.0) if total_records else 0.0

    metrics = _load_json(metrics_file)
    trace_rows = _load_trace_rows(traces_file)
    failed_trace_rows = [row for row in trace_rows if row.get("status") != "success"]
    parser_failures = int(metrics.get("parser_failures", 0))
    avg_ingest_latency_ms = _safe_float(metrics.get("avg_ingest_latency_ms"))
    max_ingest_latency_ms = _safe_float(metrics.get("max_ingest_latency_ms"))

    symbols = []
    if not ohlcv_df.empty and "symbol" in ohlcv_df:
        symbols.extend(ohlcv_df["symbol"].dropna().astype(str).tolist())
    if not corp_df.empty and "symbol" in corp_df:
        symbols.extend(corp_df["symbol"].dropna().astype(str).tolist())
    unique_symbols = sorted(set(symbols))

    report = f"""# Week 2 Observability Dashboard

## Data Pipeline Metrics

- **Total Records Processed (Silver):** {total_records}
- **OHLCV Records:** {len(ohlcv_df)}
- **Corporate Action Records:** {len(corp_df)}
- **Unique Symbols:** {len(unique_symbols)}
- **Symbols Present:** {", ".join(unique_symbols) if unique_symbols else "none"}

## Failover & Resilience

- **Primary Source Records:** {total_records - fallback_records}
- **Fallback Source Records:** {fallback_records}
- **Fallback Percentage:** {fallback_pct:.2f}%

## Latency & Lag

- **Average Ingest Latency (ms):** {avg_ingest_latency_ms:.2f}
- **Max Ingest Latency (ms):** {max_ingest_latency_ms:.2f}
- **OHLCV Maximum Lag (seconds):** {lag_max:.2f}
- **OHLCV Minimum Lag (seconds):** {lag_min:.2f}
- **OHLCV Average Lag (seconds):** {lag_avg:.2f}

## Parser + Tracing

- **Parser Failures:** {parser_failures}
- **Trace Spans Recorded:** {len(trace_rows)}
- **Trace Spans Failed:** {len(failed_trace_rows)}
- **Trace File:** `{traces_file}`

## Source Artifacts

- **Metrics File:** `{metrics_file}`
- **Generated At (UTC):** {metrics.get("generated_at_utc", "unknown")}
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"Report generated successfully at {report_path}")


if __name__ == "__main__":
    main()
