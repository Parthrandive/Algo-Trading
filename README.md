# Algo-Trading System

Multi-Agent AI Trading System for Indian Markets (NSE/F&O/MCX).

## Documentation

- **Master Plan**: [Multi_Agent_AI_Trading_System_Plan_Updated.md](docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md)
- **Phase 1 Execution Plan**: [Phase_1_Data_Orchestration_Execution_Plan.md](docs/plans/Phase_1_Data_Orchestration_Execution_Plan.md)
- **Architecture Decisions**: [docs/architecture/](docs/architecture/)
- **Governance & Policies**: [docs/governance/](docs/governance/)

## Project Structure

- `src/`: Source code for agents and utilities.
- `tests/`: Unit and integration tests.
- `docs/`: Documentation, plans, and architectural records.
- `scripts/`: Utility scripts (e.g., smoke tests).
- `configs/`: Configuration files.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Tests**:
    ```bash
    pytest
    ```

3.  **Run Smoke Test**:
    ```bash
    python3 scripts/smoke_test.py
    ```

## Docker Setup

Use this path if you want your partner to avoid local `pip` installation and run everything through the container.

1.  **Build and start app container**:
    ```bash
    docker compose up --build -d
    ```

2.  **(Optional) Start local DB container**:
    ```bash
    docker compose -f docker-compose.db.yml up -d
    ```

3.  **Verify required Python packages inside container**:
    ```bash
    docker compose exec algo-trading python -c "import yfinance, nsepython, psycopg2; print('deps ok')"
    ```

4.  **Run scripts from inside container**:
    ```bash
    docker compose exec algo-trading python scripts/show_latest_data.py RELIANCE.NS --auto-fetch-missing-history --days 1
    ```

## Data Utility CLIs

### 1) Show Latest Data

Use `scripts/show_latest_data.py` to read local Silver history and optionally fetch missing history on demand.

```bash
python3 scripts/show_latest_data.py [SYMBOL] [options]
```

Examples:

```bash
# Local history + live quote
python3 scripts/show_latest_data.py TCS.NS

# Auto-fetch missing historical window before display
python3 scripts/show_latest_data.py INFY --auto-fetch-missing-history --days 7

# Explicit historical window (UTC), skip live quote, return JSON summary
python3 scripts/show_latest_data.py TATASTEEL.NS --auto-fetch-missing-history --from 2026-02-01 --to 2026-02-20 --no-live --json
```

Key options:

- `--auto-fetch-missing-history`: Fetch missing/gapped history before display.
- `--days N`: Backfill `N` days when local history is missing (default `7`).
- `--from YYYY-MM-DD`: Backfill start date (UTC). Mutually exclusive with `--days`.
- `--to YYYY-MM-DD`: Backfill end date (UTC). Default is now.
- `--interval {1h,1d}`: Historical interval for on-demand fetch.
- `--no-live`: Skip live quote fetch.
- `--json`: Emit JSON summary in addition to console output.

### 2) Bulk Historical Backfill

Use `scripts/backfill_historical.py` for multi-symbol historical ingestion for training datasets.

```bash
python3 scripts/backfill_historical.py [options]
```

Examples:

```bash
# Backfill two symbols for the last 30 days
python3 scripts/backfill_historical.py --symbols HDFCBANK.NS,ITC.NS --days 30

# Backfill from universe file with bounded workers and custom report/checkpoint paths
python3 scripts/backfill_historical.py --universe ./symbols.csv --from 2025-01-01 --to 2026-02-20 --workers 4 --checkpoint /tmp/backfill_checkpoint.json --report /tmp/backfill_report.json

# Force refresh and stop on first failure
python3 scripts/backfill_historical.py --symbols RELIANCE.NS,TCS.NS --from 2025-01-01 --force-refresh --fail-fast
```

Key options:

- `--symbols S1,S2,...`: Comma-separated symbols.
- `--universe path.csv|path.txt`: Universe file input (CSV/TXT).
- `--days N` or `--from YYYY-MM-DD`: Backfill window start control.
- `--to YYYY-MM-DD`: Backfill window end (UTC).
- `--interval {1h,1d}`: Ingest interval.
- `--workers N`: Worker threads (`1` to `16`).
- `--skip-recent-hours N`: Skip fetch if local data is already fresh.
- `--resume` / `--no-resume`: Use or ignore checkpoint state.
- `--force-refresh`: Ignore freshness checks and refetch full window.
- `--continue-on-error` / `--fail-fast`: Failure mode.
- `--max-failures N`: Stop once failure count reaches `N`.
- `--write-bronze` / `--no-write-bronze`: Enable/disable Bronze writes during backfill.
- `--checkpoint PATH`: Checkpoint JSON path.
- `--report PATH`: Report JSON path.

### Exit Codes

Both data scripts follow this convention:

- `0`: Success.
- `1`: Partial failure (at least one symbol failed in batch workflows).
- `2`: CLI usage/validation error.
- `3`: Fatal runtime error.
- `130`: Interrupted (`Ctrl+C`).
