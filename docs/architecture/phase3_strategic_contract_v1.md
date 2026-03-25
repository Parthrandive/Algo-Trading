# Phase 3 Strategic -> Strategic Executive Contract (v1)

Date: March 25, 2026  
Scope: Week 1 lock for Phase 3 handoff

## 1) Contract Decision
The final integration contract is `strategic_exec_v1`, implemented in:
- `src/agents/strategic/schemas.py` (`StrategicToExecutiveContract`)
- `src/agents/strategic/contracts.py` (`StrategicContractEnvelope`)

This contract is the only permitted payload from Week 1 strategic workers to downstream strategic_executive consumers.

## 2) Required Fields
- `contract_version`
- `timestamp`
- `symbol`
- `policy_id`
- `policy_type` (`teacher` or `student`)
- `loop_type` (`offline`, `slow`, `fast`)
- `action` (`buy`, `sell`, `hold`, `close`, `reduce`)
- `action_size` (0.0 to 1.0)
- `confidence` (0.0 to 1.0)
- `snapshot_id`
- `observation_schema_version`

## 3) Hard Safety Constraint
Teacher policies are blocked from Fast Loop usage by schema validation:
- if `policy_type=teacher`, then `loop_type=fast` is rejected.

This is aligned with AGENTS.md hard rules and Execution/Fast Loop skill guidance.

## 4) Compatibility Rules
- Observation payloads must use `observation_schema_version=1.0` for Week 1.
- Contract payloads must use `contract_version=strategic_exec_v1`.
- Any field addition/removal requires contract version bump.

## 5) Serialization
- Internal transport: Python dict / pydantic model
- External handoff: JSON objects validated against `StrategicToExecutiveContract`
