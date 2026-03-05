# Data Scale Roadmap v1

- **Baseline trading day**: 2026-03-05
- **Snapshot**: `snapshot_20260305_182828_UTC`

## Current Volume Baseline
| Stream | Current Daily Volume (GB) |
| --- | --- |
| Sentinel OHLCV | 0.00001030 |
| Sentinel Corporate Actions | 0.00000996 |
| Macro Indicators | 0.00000801 |
| Text Canonical | 0.00000102 |
| Text Sidecar | 0.00000063 |
| Preprocessing Gold (derived output) | 0.00001114 |
| Total Input Streams | 0.00002993 |

## Phase 2 Targets and Quarterly Checkpoints
| Checkpoint | Target Total Input GB/Day | Focus |
| --- | --- | --- |
| Q2 2026 | 1.00 | Stabilize replay determinism and complete partner text replay automation. |
| Q3 2026 | 5.00 | Scale to full NSE universe and add textual partner production feed. |
| Q4 2026 | 20.00 | Introduce warm-tier compaction, catalog partitioning, and retention automation. |
| Q1 2027 | 75.00 | Phase 2 throughput readiness with replay SLA and quarterly backfill drills. |

## Assumptions
- Baseline reflects replay slice artifacts generated for Day 4 Sync S2.
- Targets represent scaling milestones for full-universe and partner textual production onboarding.
