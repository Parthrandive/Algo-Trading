# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-15
### Added
- `requirements.lock` for pinned reproducible dependency resolution from the active `venv`.
- `docs/architecture/architecture_decisions_v1.md` as the locked Day 2 architecture decision record.
- `docs/architecture/module_boundaries.md` with explicit module I/O boundaries and failure behavior.
- `docs/governance/data_contracts_v1.md` documenting frozen Day 3 contracts and mandatory provenance fields.
- `docs/governance/schema_compatibility_rules.md` defining backward/forward compatibility and version bump rules.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.0 with:
  - Optional Phase 4.5 hardware acceleration track (FPGA scope + deterministic benchmarks).
  - Market-making readiness module (inventory, hedging policy engine, quote-response simulation).
  - Research-to-production throughput KPIs and low-latency ML release gates.
  - Cross-functional pod ownership and 48-hour post-deploy accountability reviews.
- `docs/architecture/execution_loops.md` expanded with student-only Fast Loop inference, tail-latency release gates, and FPGA scope policy.
- `docs/architecture/storage_strategy.md` expanded with data-scale roadmap, hot/warm/cold access layers, and replay framework requirements.
- `docs/governance/mlops_governance.md` expanded with research throughput KPIs and low-latency ML guardrails.
- `docs/governance/runbook_v1.md` expanded with pod rotation ownership and 48-hour post-deploy review template.
- `docs/governance/sla_definitions.md` expanded with p99.9/jitter release-gate requirements.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.1 with:
  - Fast Loop stretch target tightened to p99 <= 8 ms with enforced degrade safeguards above 10 ms.
  - Optional market-making policy mode, order-book imbalance emphasis, and liquidity-rebate-aware reward variant.
  - Real-time slippage-impact sizing controls and live per-agent/per-signal PnL attribution requirements.
  - Mandatory shadow A/B evidence for major USD/INR and gold model changes.
  - Optional 15-30 minute online RL micro-update path gated to paper/shadow first.
  - Scenario-based COMEX shock limits for gold expansion and explicit prioritization tiers.
- `docs/architecture/execution_loops.md` tightened with p99 <= 8 ms stretch target, >10 ms degrade cap, and explicit Rust/C++ critical-path allowance.
- `docs/governance/mlops_governance.md` updated with lightweight inference guardrail, mandatory A/B for major FX-gold changes, and micro-update cadence controls.
- `docs/governance/sla_definitions.md` updated with dedicated Fast Loop decision compute SLO.
- `docs/governance/runbook_v1.md` updated with post-deploy PnL attribution health check.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.2 with:
  - Removal of speculative uplift language from executive summary.
  - Explicit no-assumed-speedup policy for Rust/C++/FPGA adoption.
  - New phase-by-phase go/no-go benchmark gate table (Phases 1, 2, 3, 4, 4.5, 5, and 17).
- `docs/architecture/execution_loops.md` updated with benchmark-only evidence standard for language/hardware acceleration paths.
- `docs/governance/mlops_governance.md` updated with evidence-first uplift rule (no fixed Sharpe/accuracy uplift claims without controlled A/B evidence).
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.3 with:
  - Formal `Decision Memo` section for IMC-inspired tech stack adoption.
  - Added `What to Fix for Rigor` checklist and `Recommended Wording Pattern` standard.
  - Evidence-first prioritization table with explicit go/no-go criteria for five initiatives.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.4 with:
  - Rigor and wording standards moved into core sections (`MLOps`, `Rollout`, and `Next Iteration`) instead of a standalone memo section.
  - Explicit anti-speculative approval rules (no fixed uplift claims without controlled evidence).
  - Standardized change-request fields for expected impact, go criterion, and no-go criterion.
- `docs/governance/release_change_request_template.md` added as mandatory evidence-first release submission template.
- `docs/governance/ci_benchmark_evidence_checklist.md` added as mandatory CI benchmark gate checklist.
- `docs/governance/mlops_governance.md` updated to require both templates in staging-to-production approval.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` rollout section updated to require completed governance templates for promotion requests.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.5 to reflect template-driven release governance operationalization.
- `pull_request_template.md` added with mandatory evidence-first PR fields and direct links to both governance templates.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.6 with implementation-freeze prioritization:
  - Tier 1 (next 4 to 8 weeks): impact or slippage monitor, dynamic vol-scaled risk budgets, order-book imbalance in Fast Loop, and tightened Fast Loop latency discipline.
  - Tier 2 (after three months paper trading): shadow A/B enforcement for major changes and basic crisis-weighted voting activation.
  - Tier 3 deferred: hardware or FPGA track, market-making module, online RL micro-updates, and full pod ownership model.
- `docs/governance/mlops_governance.md` updated to phase Shadow A/B enforcement after three months paper trading, with rationale-required fallback before activation.
- `docs/governance/runbook_v1.md` updated to current-cycle single owner plus reviewer operations (full pod rotation deferred).
- `docs/governance/release_change_request_template.md` updated with Tier 2 Shadow A/B activation wording.
- `docs/governance/ci_benchmark_evidence_checklist.md` updated with Tier 2 Shadow A/B activation wording.
- `pull_request_template.md` updated to mark Shadow A/B as Tier 2 enforcement with activation flag.
- `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md` upgraded to v1.3.7 with concrete Tier 1 execution backlog:
  - Real-time impact/slippage monitor tasks and completion checks.
  - Dynamic volatility-scaled risk budget tasks and completion checks.
  - Order-book imbalance Fast Loop tasks and completion checks.
  - Tightened Fast Loop latency discipline tasks and completion checks.
- Regenerated `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.pdf` and `docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated_readable.pdf` from the updated v1.3.7 markdown plan.

### Changed
- Canonical provenance fields in schemas now use `source_type`, `ingestion_timestamp_utc`, `ingestion_timestamp_ist`, and `quality_status` with backward-compatible input aliases.
- Added/updated schema tests to validate provenance aliases, UTC+IST timestamps, and stricter contract behavior.

## [0.1.0] - 2026-02-09
### Added
- Initial project structure (`src/`, `tests/`, `configs/`, `scripts/`, `docs/governance/`).
- `requirements.txt` with Day 1 dependencies (`pandas`, `numpy`, `nsepython`, `pydantic`, etc.).
- `docs/governance/source_inventory.md` defining data sources and integrity ranks.
- `CHANGELOG.md` for document control.
