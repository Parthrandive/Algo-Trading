# Day 4 Sync S2 Defect Tracker

- **Trading day**: 2026-03-05
- **Snapshot**: `snapshot_20260305_185928_UTC`

| ID | Severity | Status | Owner | Title | Detail |
| --- | --- | --- | --- | --- | --- |
| DEF-001 | Medium | Open | Data Platform | Corporate action records failed strict validation | 16 row(s) failed schema validation during replay prep. |
| DEF-002 | Medium | Open | Preprocessing | Gold replay output is in-memory only | Preprocessing replay returns TransformOutput hash and records but does not persist to data/gold. |
