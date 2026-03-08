# CI Benchmark Evidence Checklist

Purpose: standardize objective gate checks for execution-path and model promotions.

## 1. Gate Metadata

- Pipeline run ID:
- Commit SHA:
- Branch:
- Candidate artifact ID:
- Baseline artifact ID:
- Target environment: `paper-trading` | `production`
- Change category: `model` | `execution` | `risk` | `infra`

## 2. Required CI Checks

- [ ] Unit and integration test suite passes.
- [ ] Schema compatibility and contract tests pass.
- [ ] Deterministic inference or transform test passes.
- [ ] Replay benchmark job completes and publishes artifact.
- [ ] Correctness parity check vs baseline produced.

## 3. Threshold Evaluation (For Execution-Path Changes)

- [ ] Fast Loop latency metrics (p50, p95, p99, p99.9, jitter) recorded.
- [ ] Fast Loop stretch target (`p99 <= 8ms`) status recorded.
- [ ] Hard safety rule (`degrade if p99 > 10ms`) status recorded.
- [ ] No critical risk-control regression detected.

## 4. Artifact Paths

- Replay benchmark artifact:
- Latency summary artifact:
- Correctness parity artifact:
- Paper-trading results artifact (if applicable):

## 5. Gate Outcome

- Result: `GO` | `NO-GO`
- Blocking reasons (if NO-GO):
- Reviewer (Owner):
- Timestamp:

## 6. Manual Follow-Ups (If Required)

- [ ] Paper-trading validation period completed for model/execution changes.
- [ ] Post-deploy monitoring plan documented.
- [ ] Rollback trigger and rollback runbook verified.
