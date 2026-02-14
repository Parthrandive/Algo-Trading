# Execution Loop Architecture

## 1. Fast Loop (Execution-Critical)
**Purpose**: Make order/risk decisions only from already-validated state.

### Constraints
- **Inputs Allowed**:
    - In-memory/cache reads of *latest* technical features.
    - Risk state (positions, orders).
    - Provenance & quality flags.
    - Distilled student-policy inference outputs with pre-approved deterministic runtime profile.
    - Lightweight microstructure signals (including order-book imbalance) from pre-published snapshots.
- **Inputs NOT Allowed**:
    - Live API calls (HTTP/Websocket init).
    - Database scans/heavy queries.
    - File I/O.
    - Heavy model inference or retraining.
    - Teacher policy inference, online training, or calibration.
- **Latency Target**: Stretch `p99 <= 8 ms` on execution-critical compute stages.
    - **Hard Cap**: If processing exceeds `p99 > 10 ms`, auto-trip to **Degrade Mode**.

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
- **Teacher Workloads**: Expensive modeling, research inference, and retraining stay in Slow Loop or offline jobs only.
- **Implementation Choice**: Critical parsing/routing components may run in Rust/C++ services if they preserve contract and observability requirements.

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

## 4. Latency Engineering Rules

- **Release Gates**: Every execution-path release must pass stretch `p99 <= 8 ms`, enforce degrade safeguards above `10 ms`, and satisfy defined `p99.9` jitter limits in replay and peak-load simulations.
- **Model Compression**: Student artifacts must be compression-validated (distillation/quantization/pruning as applicable) before production promotion.
- **Hardware Scope**: FPGA acceleration is allowed only for execution-critical components with measured gain; non-critical services remain CPU/GPU.
- **Evidence Standard**: Rust/C++/FPGA paths are benchmark-gated only; fixed speedup multipliers are not assumed.
