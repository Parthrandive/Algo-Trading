# Preprocessing Agent — Detailed Implementation Plan (Days 2–7)

**Dates:** February 24 – March 1, 2026
**Linked to:** `Week_3_Preprocessing_Backlog.md`
**Status:** I/O Contract frozen (Day 1 completed).

---

## Architecture Overview

The Preprocessing Agent is designed as a deterministic DAG (Directed Acyclic Graph) of data transformations. Its purpose is to load raw/canonical Silver data (Macro indicators and OHLCV bars), apply deterministic alignment, normalize features, and export Gold-ready tensors with robust provenance.

### Core Pipeline Flow:
1. **Loaders:** Read Parquet/JSON batches from Silver with `dataset_snapshot_id`. Validate using strict schemas.
2. **Lag Alignment:** Shift macro-economic variables (e.g., CPI) forward in time based on their _publication_ date, preventing look-ahead bias against fast-moving market prices.
3. **Transform Graph:** Execute registered `TransformNode` objects (Z-score, Min-Max, Log Returns) based on a versioned `TransformConfig`.
4. **Reproducibility Layer:** Hash the final ordered output and store it as a `TransformOutput` record.

---

## Detailed Day-by-Day Implementation

### Day 2 (Tue) — Transform Graph Scaffold + Loaders

**Objective:** Build the foundational I/O structure and DAG definitions.

1. **`src/agents/preprocessing/loader.py`**
   - Create `MacroLoader` class with `.load(directory_path, snapshot_id)` returning a Pandas DataFrame.
   - Enforce schema validation against `MacroIndicator` schema `v1.1`.
   - Create `MarketLoader` class validating against `Bar` schema `v1.0`.

2. **`src/agents/preprocessing/transform_graph.py`**
   - Define abstract `TransformNode` class with properties: `name`, `version`, `input_schema_version`, and a core `transform(df: pd.DataFrame) -> pd.DataFrame` method.
   - Implement `TransformGraph` class that acts as a dependency injection container. Allows registering nodes and executing them topologically.
   - Add schema tracking on the output DataFrame metadata.

### Day 3 (Wed) — Normalization Modules + Config Versioning

**Objective:** Implement standard transformations required for the ML model inputs.

1. **`src/agents/preprocessing/normalizers.py`**
   - Implement `ZScoreNormalizer(TransformNode)` using rolling standard deviation (configurable window, e.g., 30 days).
   - Implement `MinMaxNormalizer(TransformNode)` using rolling bounds.
   - Implement `LogReturnNormalizer(TransformNode)` using `np.log(price / price.shift(1))`.
   - Implement `DirectionalChangeDetector(TransformNode)` to track when trend thresholds are breached.

2. **`configs/transform_config_v1.json`**
   - Create the exact configuration JSON defining the sequence of nodes and their parameters.

3. **`tests/agents/preprocessing/test_transform_graph.py`**
   - Guarantee DAG execution correctness (Topological sorting).
   - Test versions: If config specifies ZScore v1.0, and the Node is v1.1, the graph must reject execution to ensure reproducible artifacts.

### Day 4 (Thu) — Lag-Alignment + Pipeline Wiring

**Objective:** Map slow-moving macro data correctly onto fast-moving price data without future leakage.

1. **`src/agents/preprocessing/lag_alignment.py`**
   - Implement `LagAligner` using Pandas `asof` merges.
   - Configure publication delay per macro tag (e.g., Inflation is reported mid-month for the previous month; the `LagAligner` shifts the date index to the release datetime, not the observed datetime).

2. **`src/agents/preprocessing/pipeline.py`**
   - Implement `PreprocessingPipeline` facade: `loader -> LagAligner -> TransformGraph -> Output`.
   - Implement `ReplaySupport` helper to build outputs deterministically from archived Silver paths.

### Day 5 (Fri) — Leakage + Reproducibility 🔒 CP2

**Objective:** Enforce Section 5.3 compliance regarding information control.

1. **`src/agents/preprocessing/reproducibility.py`**
   - Implement `ReproducibilityHasher` (SHA-256 over concatenated, sorted string representations of the output DataFrames augmented with the snapshot ID).

2. **`src/agents/preprocessing/leakage_test.py`**
   - Build `LeakageTestHarness` that injects synthetic future data into raw macro sets and verifies the `LagAligner` blocks it from reaching the aligned feature sets.

3. **CP2 Verification:**
   - Execute test harness (`tests/agents/preprocessing/test_leakage.py` & `test_reproducibility.py`).
   - Create `docs/plans/CP2_Pipeline_Readiness.md`.

### Day 6 (Sat) — Consume Macro Samples 🔒 CP3 (Sync Gate A)

**Objective:** End-to-end integration mapping macro test data into the pipeline.

1. **Mock Data Creation / Partner Consumption:**
   - Read JSON files from `data/macro_samples/` created by the Macro pipeline.
   - Run through `MacroLoader` list validation (`TypeAdapter(list[MacroIndicator])`).
   - Ensure the new enum (`RBI_BULLETIN` count markers) are appropriately loaded and feature-aligned.

2. **CP3 Verification:**
   - Validate success of execution paths and log metrics.
   - Document outcomes in `docs/plans/CP3_Sync_Gate_A.md`.

### Day 7 (Sun) — Deterministic Replay + Sign-off 🔒 CP4

**Objective:** Section 5.5 final validation.

1. **Replay Validation:**
   - Define a fixed snapshot date.
   - Re-run the entire `PreprocessingPipeline` using the specific snapshot three times.
   - Compare SHA-256 hashes computationally.

2. **Phase 1 Sign Off:**
   - Produce `docs/plans/CP4_Week_3_Signoff.md` confirming readiness for textual preprocessing and Phase 2 Gold ingestion.
