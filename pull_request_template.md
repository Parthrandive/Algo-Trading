# Pull Request Template (Evidence-First)

Use this template for every PR. Incomplete PRs should not be approved.

Required governance references:
- `docs/governance/release_change_request_template.md`
- `docs/governance/ci_benchmark_evidence_checklist.md`

Direct links:
- [Release Change Request Template](./docs/governance/release_change_request_template.md)
- [CI Benchmark Evidence Checklist](./docs/governance/ci_benchmark_evidence_checklist.md)

## Submission Checklist (Required)

- [ ] I completed all required sections below.
- [ ] I linked or attached the completed release change request template.
- [ ] I linked or attached the completed CI benchmark evidence checklist.
- [ ] I confirmed this PR follows the evidence-first policy (no fixed uplift claims without controlled evidence).

## 1. Change Metadata (Required)

- Change category: `model` | `execution` | `risk` | `data` | `infra`
- Major change: `yes` | `no`
- Scope: `USD/INR` | `NSE` | `Gold` | `Multi-asset`
- Phase(s) impacted:
- Target environment: `staging` | `production`
- Linked issue(s):

## 2. Evidence-First Required Statements (Required)

- Expected impact statement:
  - `Expected impact: Unknown until controlled benchmark/A-B; target is non-regression plus measurable improvement.`
- Go criterion:
  - `Go criterion: all applicable phase gate checks pass.`
- No-go criterion:
  - `No-go criterion: any latency, risk, or correctness gate fails.`

## 3. Baseline vs Candidate Evidence (Required)

- Baseline artifact ID:
- Candidate artifact ID:
- Measurement window:
- Statistical method and threshold:
- Replay benchmark result:
- Peak-load benchmark result:
- Fast Loop metrics (p50, p95, p99, p99.9, jitter):
- Degrade-path safety result (`p99 > 10ms` behavior):
- Correctness parity result:
- Observability and recovery parity result:

## 4. Shadow A/B Status (Required for Major USD/INR or Gold Changes)

- Shadow A/B required: `yes` | `no`
- A/B window:
- Non-regression result (risk-adjusted metrics):
- Non-regression result (slippage/impact):
- Decision summary:

## 5. Risk, Rollback, and Ops (Required)

- Risk controls impacted:
- New failure modes:
- Rollback trigger(s):
- Rollback plan:
- 48-hour post-deploy owner:

## 6. Reviewer Gate (Required)

- [ ] Risk reviewer confirms gates are met.
- [ ] Engineering reviewer confirms benchmark evidence is complete.
- [ ] Trading/Product reviewer confirms deployment decision (`GO`/`NO-GO`).

