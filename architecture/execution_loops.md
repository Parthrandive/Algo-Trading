# Execution Loop Architecture

## 1. Fast Loop (Execution-Critical)
**Purpose**: Make order/risk decisions only from already-validated state.

### Constraints
- **Inputs Allowed**:
    - In-memory/cache reads of *latest* technical features.
    - Risk state (positions, orders).
    - Provenance & quality flags.
- **Inputs NOT Allowed**:
    - Live API calls (HTTP/Websocket init).
    - Database scans/heavy queries.
    - File I/O.
    - Heavy model inference or retraining.
- **Latency Target**: `p99 <= 10 ms` (Hardware/Network capability permitting).
    - **Hard Cap**: If processing exceeds limit, auto-trip to **Degrade Mode**.

### Behavior
- **Non-Blocking**: Always reads latest published snapshot in O(1). **Never** waits for Slow Loop.
- **Fail-Safe**: If snapshot is `stale`, `expired`, or `quality-fail`, automatically shifts to **REDUCE_ONLY** or **CLOSE_ONLY** advisory mode.

## 2. Slow Loop (Context + Updates)
**Purpose**: Ingestion, normalization, feature recompute, fallback handling, quality checks, backfills.

### Constraints
- **Execution**: Runs asynchronously. Off the critical path.
- **Latency**: Typical ~100–500 ms+.
- **Blocking**: **Never** blocks the Fast Loop thread/process.

### Behavior
- **Publication**: Publishes atomic snapshots for Fast Loop consumption.
- **Failure**: Failure here degrades system mode and raises alerts but **cannot pause** the Fast Loop.

## 3. Cross-Loop Protocol

### Atomic Handoff
- Snapshot updates must be atomic (e.g., pointer swap `current_snapshot_id`).
- Fast Loop sees either *Old Consistent State* or *New Consistent State*, never partial state.

### Payload Schema
All cross-loop data payloads must carry:

| Field | Description |
| :--- | :--- |
| `snapshot_id` | Unique UUID/Sequence ID for the state. |
| `generated_at` | Timestamp of calculation completion. |
| `expires_at` | Timestamp validity limit (TTL). |
| `schema_version` | Version of the data contract. |
| `quality_status` | Enum (`pass`, `warn`, `fail`). |
| `source_type` | Provenance tag (e.g., `official_api`). |
