# Phase 3 Week 1 Fast Loop Exclusion Note

Date: March 25, 2026  
Scope: Strategic Week 1 handoff safety note

## Decision
Teacher policies (SAC/PPO/TD3 placeholders and future trained teachers) are excluded from Fast Loop execution.

## Enforcement Points
- Contract schema validation in `src/agents/strategic/schemas.py` rejects:
  - `policy_type=teacher` with `loop_type=fast`.
- Placeholder dry-run command (`scripts/run_phase3_teacher_dry_run.py`) emits only slow-loop teacher actions.
- Week 2 export schema (`week2_action_space_v1`) enforces same exclusion.

## Rationale
- Execution/Fast Loop discipline requires student-policy-only inference in the execution-critical path.
- Teacher models are offline/slow-loop artifacts by design and may violate Fast Loop latency budgets.

## Week 2 Expectation
- Distilled student policy will be introduced in Week 2 and may be evaluated for Fast Loop eligibility under latency and agreement gates.
