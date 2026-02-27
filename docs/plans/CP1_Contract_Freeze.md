# CP1: Contract Freeze

**Date:** February 23, 2026
**Component:** Preprocessing Agent & Macro Monitor Agent

## Evidence of Contract Freeze

### 1. Frozen Schema
- `src/schemas/macro_data.py` has been updated to `schema_version = "1.1"`.
- `MacroIndicatorType` now includes `CPI, WPI, IIP, GDP, INR_USD, BRENT_CRUDE, US_10Y, INDIA_10Y, FII_FLOW, DII_FLOW, REPO_RATE, FX_RESERVES, INDIA_US_10Y_SPREAD, RBI_BULLETIN`.

### 2. Preprocessing Data Schema
- `src/schemas/preprocessing_data.py` created with `TransformConfig`, `TransformOutput`, and `PreprocessingContract` models.

### 3. Preprocessing I/O Contract
- `configs/preprocessing_contract_v1.json` committed.
- Accepts: `MacroIndicator v1.1`, `Bar v1.0`, and `Tick v1.0`.
- Defines format for `dataset_snapshot_id` and hash generation rules for determinism (Section 5.5).

### 4. Feature Approval Workflow (Section 5.3)
To ensure all feature engineering is deterministic and reproducible, the following workflow is enforced before any feature promotes to production:

1. **Proposal:** 
   - Define feature mathematical logic, input data required, and expected lag (e.g., waiting for CPI publication).
   - Define normalization rules (e.g., Z-Score with 30-day window).
2. **Offline Evaluation:** 
   - Test implementation in local notebook.
   - Run leakage tests and confirm valid timestamp alignment without look-ahead bias.
3. **Shadow Mode:** 
   - Implement as a `TransformNode` in the Preprocessing DAG.
   - Test continuously against live incoming Silver data to catch runtime anomalies, missing fields, or compute spikes.
4. **Promote:** 
   - Feature output hash is verified as stable.
   - Leakage tests pass (100%).
   - Added to Phase 2 Gold-ready artifacts.
