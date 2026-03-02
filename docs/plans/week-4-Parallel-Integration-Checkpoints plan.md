# Week 4 Plan: Parallel Integration Checkpoints (Textual + GlobalMacroExogenous)

## Title and Purpose
Define joint execution checkpoints for the Week 4 parallel workstreams so both tracks stay decoupled during build but converge through measurable integration gates.

Related plans:
- [Week 4 Textual Data Agent Plan](./week-4-textual-data-agent-plan.md)
- [Week 4 GlobalMacroExogenous Agent Plan](./week-4-GlobalMacroExogenous-Agent%20plan.md)

## Owners and Responsibilities
- You: Textual Data Agent owner.
- Partner: GlobalMacroExogenous Agent owner.
- Shared responsibility: integration signoff, artifact validation, and decision logging at CP1/CP2/CP3.

## Integration Contract Between Workstreams
- No direct code coupling in Week 4.
- Exchange only versioned artifacts and contract reports.
- Shared join-key policy:
  - timestamps must be UTC-normalized for checkpoint comparison
  - provenance completeness must be explicit
  - quality status semantics must stay aligned (`pass`, `warn`, `fail`)

## Three Formal Gates
### CP1 Contract Freeze - Tuesday, March 3, 2026
Scope:
- Freeze schema IDs, output paths, scoring formulas, and compliance rule set.

Required artifacts:
- Contract markdown for both streams.
- Sample payloads (small seed packs).
- Field dictionary and validation rules.

Pass criteria:
- No unresolved schema-key conflicts.
- Provenance and compliance fields mapped for both streams.
- Artifact locations agreed and documented.

### CP2 Mid-Week Integration - Thursday, March 5, 2026
Scope:
- Validate both outputs can be consumed together in downstream staging assumptions.

Required artifacts:
- Schema validation report.
- Freshness report.
- Sidecar compatibility checklist.

Pass criteria:
- Both artifact families validate against declared contracts.
- UTC timestamp alignment checks pass.
- Fallback and stale handling behavior is documented and non-conflicting.

### CP3 Final Week Signoff - Sunday, March 8, 2026
Scope:
- Confirm end-to-end readiness for next-week implementation.

Required artifacts:
- Pass/fail gate table.
- Unresolved issues list.
- Recommended next sprint tasks with owners.

Pass criteria:
- All critical gate checks are green or formally waived with owner and deadline.
- Cross-stream integration notes are complete and signed.

## Meeting Template for Each Gate
- Inputs reviewed.
- Pass criteria checklist.
- Decision log.
- Blockers and owner.
- Deadline and rollback or fallback note.

## Pass Criteria Table
| Criterion | Textual Evidence | GlobalMacroExogenous Evidence | Gate Expectation |
| --- | --- | --- | --- |
| Data contract compliance | Canonical schema validation report | `ExogenousIndicator_v1.0` validation report | Zero untriaged contract breaks |
| Provenance completeness | Source and ingestion field coverage | Source and ingestion field coverage | 100% tagged accepted records |
| Timestamp alignment | UTC alignment evidence | UTC alignment evidence | Join-key consistency verified |
| Freshness and staleness behavior | TTL/freshness sidecar checks | Fresh/stale marker report | No semantic conflict |
| Compliance and licensing controls | Reject logs with reason codes | Reject logs with reason codes | Blocked sources auditable |
| Failure mode behavior | Text source outage handling | WorldMonitor outage fallback handling | Deterministic degrade path |
| Actionable next-step ownership | Open issues with owner/date | Open issues with owner/date | No unowned blocker |

## Test Cases and Scenarios
- Valid and invalid schema payload checks for both streams.
- No look-ahead timestamp leakage checks.
- Fresh vs stale vs expired input handling checks.
- Source outage fallback behavior checks.
- Cross-stream UTC timestamp alignment checks.
- Compliance rejection flow checks with reason logging.

## Escalation Rules
- Any failed gate blocks downstream live integration work.
- Independent stream progress may continue during remediation.
- No merge decision is allowed until failed gate criteria are closed or formally waived.
