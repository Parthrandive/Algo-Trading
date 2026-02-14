# Release Change Request Template (Evidence-First)

Use this template for any model, execution-path, or risk-control change proposed for staging or production.

## 1. Change Metadata

- Change ID:
- Date:
- Owner:
- Reviewers:
- Pod:
- Phase(s) impacted:
- Environment target: `staging` | `production`
- Scope: `USD/INR` | `NSE` | `Gold` | `Multi-asset`
- Change category: `model` | `execution` | `risk` | `data` | `infra`
- Major change: `yes` | `no`
- Linked issues or PRs:

## 2. Evidence-First Required Statements

- Expected impact statement:
  - `Expected impact: Unknown until controlled benchmark/A-B; target is non-regression plus measurable improvement.`
- Go criterion:
  - `Go criterion: all applicable phase gate checks pass.`
- No-go criterion:
  - `No-go criterion: any latency, risk, or correctness gate fails.`

## 3. Baseline and Measurement Window

- Baseline build/model ID:
- Baseline measurement window:
- Candidate build/model ID:
- Candidate measurement window:
- Statistical method:
- Significance or confidence threshold:

## 4. Benchmark Evidence (Attach Artifacts)

- Replay benchmark report attached:
- Peak-load benchmark report attached:
- Fast Loop latency metrics attached (p50, p95, p99, p99.9, jitter):
- Degrade-path behavior test attached (`>10ms` safeguard):
- Correctness parity report attached:
- Observability parity report attached:
- Recovery or failover drill report attached:

## 5. Shadow A/B Evidence

- Shadow A/B required: `yes` | `no` (mandatory for major USD/INR or Gold changes)
- Control arm:
- Candidate arm:
- A/B window:
- Non-regression result on risk-adjusted metrics:
- Non-regression result on slippage and impact:
- Summary judgment:

## 6. Risk and Rollback

- Pre-trade risk controls impacted:
- New failure modes introduced:
- Rollback trigger(s):
- Rollback plan:
- Kill-switch interaction notes:

## 7. Decision

- Decision: `GO` | `NO-GO`
- Approver (Risk):
- Approver (Engineering):
- Approver (Trading or Product):
- Decision timestamp:
- Notes:

