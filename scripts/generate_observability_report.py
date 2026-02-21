"""
Day 13 Observability Report
Calculates lag and failover metrics from Silver data.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("Generating Observability Report...")
    
    silver_dir = PROJECT_ROOT / "data" / "e2e_test" / "silver"
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "observability_dashboard.md"
    
    if not silver_dir.exists():
        print("ERROR: Silver data not found. Run test_day13_e2e.py first.")
        sys.exit(1)
        
    silver_files = list(silver_dir.rglob("*.parquet"))
    
    if not silver_files:
        print("ERROR: No parquet files found in silver directory.")
        sys.exit(1)
        
    dfs = [pd.read_parquet(f) for f in silver_files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Calculate Lag (Ingestion UTC - Event Timestamp UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ingestion_timestamp_utc'] = pd.to_datetime(df['ingestion_timestamp_utc'], utc=True)
    
    df['lag_seconds'] = (df['ingestion_timestamp_utc'] - df['timestamp']).dt.total_seconds()
    
    max_lag = df['lag_seconds'].max()
    min_lag = df['lag_seconds'].min()
    avg_lag = df['lag_seconds'].mean()
    
    # Failovers
    total_records = len(df)
    fallback_records = len(df[df['source_type'] == 'fallback_scraper'])
    fallback_pct = (fallback_records / total_records) * 100 if total_records > 0 else 0
    
    report = f"""# Week 2 Observability Dashboard

## Data Pipeline Metrics

- **Total Records Processed (Silver):** {total_records}
- **Unique Symbols:** {df['symbol'].nunique()}
- **Symbols Present:** {', '.join(df['symbol'].unique())}

## Failover & Resilience

- **Primary Source Records:** {total_records - fallback_records}
- **Fallback Source Records:** {fallback_records}
- **Fallback Percentage:** {fallback_pct:.2f}%

## Latency & Lag

*Note: For historical data batch ingestion, lag represents the time since the historical event occurred.*

- **Maximum Lag (seconds):** {max_lag:.2f}
- **Minimum Lag (seconds):** {min_lag:.2f}
- **Average Lag (seconds):** {avg_lag:.2f}

"""

    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"Report generated successfully at {report_path}")

if __name__ == "__main__":
    main()
