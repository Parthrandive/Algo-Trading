# Pull Request Template (Evidence-First)

Use this template for every PR. Incomplete PRs should not be merged.

## Submission Checklist (Required)

- [ ] I completed all required sections below.
- [ ] CI pipeline passes (unit tests, schema checks, linting).
- [ ] I confirmed this PR follows the evidence-first policy (no fixed uplift claims without benchmark evidence).

## 1. Change Metadata (Required)

- Change category: `model` | `execution` | `risk` | `data` | `infra`
- Major change: `yes` | `no`
- Scope: `USD/INR` | `NSE` | `Gold` | `Multi-asset`
- Phase(s) impacted:
- Target environment: `paper-trading` | `production`
- Linked issue(s):

## 2. Evidence-First Required Statements (Required)

- Expected impact statement:
  - `Expected impact: Unknown until controlled benchmark; target is non-regression plus measurable improvement.`
- Go criterion:
  - `Go criterion: all applicable CI and phase gate checks pass.`
- No-go criterion:
  - `No-go criterion: any test, risk, or correctness gate fails.`

## 3. Benchmark Evidence (Required)

- CI test suite result: `pass` | `fail`
- Replay benchmark result (if applicable):
- Schema compatibility check: `pass` | `fail` | `N/A`
- Paper-trading validation period and result (for model/execution changes):
- Correctness parity result (if modifying existing logic):

## 4. Risk and Rollback (Required)

- Risk controls impacted:
- New failure modes:
- Rollback trigger(s):
- Rollback plan:
- Post-deploy monitoring plan:

## 5. Review Gate (Required)

- [ ] Owner self-review: I have reviewed my own changes for correctness, test coverage, and risk.
- [ ] Partner cross-check: Partner has reviewed critical sections and confirms no obvious issues.
- [ ] Decision: `GO` | `NO-GO`
