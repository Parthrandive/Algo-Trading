# CI Benchmark Evidence Checklist

Purpose: standardize objective gate checks for execution-path and model promotions.

## 1. Gate Metadata

- Pipeline run ID:
- Commit SHA:
- Branch:
- Candidate artifact ID:
- Baseline artifact ID:
- Target environment: `staging` | `production`
- Change category: `model` | `execution` | `risk` | `infra`

## 2. Required CI Checks

- [ ] Unit and integration test suite passes.
- [ ] Schema compatibility and contract tests pass.
- [ ] Deterministic inference or transform test passes.
- [ ] Replay benchmark job completes and publishes artifact.
- [ ] Peak-load benchmark job completes and publishes artifact.
- [ ] Fast Loop latency metrics produced: p50, p95, p99, p99.9, jitter.
- [ ] Tail-latency gate evaluation produced (pass or fail).
- [ ] Degrade-path safety check produced for latency breach behavior.
- [ ] Correctness parity check vs baseline produced.

## 3. Threshold Evaluation

- [ ] Fast Loop stretch target (`p99 <= 8ms`) status recorded.
- [ ] Hard safety rule (`degrade if p99 > 10ms`) status recorded.
- [ ] p99.9 and jitter thresholds status recorded.
- [ ] Non-regression decision on slippage or impact metrics recorded (if available in shadow pipeline).
- [ ] No critical risk-control regression detected.

## 4. Artifact Paths

- Replay benchmark artifact:
- Peak-load benchmark artifact:
- Latency summary artifact:
- Correctness parity artifact:
- Degrade-path test artifact:
- Shadow A/B summary artifact (if applicable):

## 5. Gate Outcome

- Result: `GO` | `NO-GO`
- Blocking reasons (if NO-GO):
- Reviewer:
- Timestamp:

## 6. Manual Follow-Ups (If Required)

- [ ] Shadow A/B approval completed for major USD/INR or Gold changes.
- [ ] 48-hour post-deploy review owner assigned.
- [ ] Rollback trigger and rollback runbook verified.

