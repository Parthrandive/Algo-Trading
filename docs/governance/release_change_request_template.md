# Release Change Request Template (Evidence-First)

Use this template for any model, execution-path, or risk-control change proposed for paper trading or production.

## 1. Change Metadata

- Change ID:
- Date:
- Owner:
- Partner Reviewer:
- Phase(s) impacted:
- Environment target: `paper-trading` | `production`
- Scope: `USD/INR` | `NSE` | `Gold` | `Multi-asset`
- Change category: `model` | `execution` | `risk` | `data` | `infra`
- Major change: `yes` | `no`
- Linked issues or PRs:

## 2. Evidence-First Required Statements

- Expected impact statement:
  - `Expected impact: Unknown until controlled benchmark; target is non-regression plus measurable improvement.`
- Go criterion:
  - `Go criterion: all applicable phase gate checks pass.`
- No-go criterion:
  - `No-go criterion: any test, risk, or correctness gate fails.`

## 3. Baseline and Measurement Window

- Baseline build/model ID:
- Baseline measurement window:
- Candidate build/model ID:
- Candidate measurement window:
- Statistical method:
- Significance or confidence threshold:

## 4. Benchmark Evidence (Attach Artifacts)

- CI test suite result:
- Replay benchmark report attached:
- Fast Loop latency metrics attached (p50, p95, p99, p99.9, jitter):
- Correctness parity report attached:
- Paper-trading period and results (if applicable):

## 5. Risk and Rollback

- Pre-trade risk controls impacted:
- New failure modes introduced:
- Rollback trigger(s):
- Rollback plan:
- Kill-switch interaction notes:

## 6. Decision

- Decision: `GO` | `NO-GO`
- Owner sign-off:
- Partner cross-check:
- Decision timestamp:
- Notes:
