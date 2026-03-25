# Phase 3 RL Dependency Policy (Gymnasium / Stable-Baselines3)

Date: March 25, 2026  
Scope: Strategic Executive Week 1-2 implementation

## 1) Policy
- `gymnasium` and `stable-baselines3` are Phase 3 optional dependencies.
- They must not become hard requirements for Phase 1/2 pipelines.
- Week 1 dry-run and DB materialization workflows must run without RL framework imports.

## 2) Installation Boundary
- Base environment: keeps current `requirements.txt` unchanged for non-RL flows.
- RL training environment (Phase 3 training only): install RL extras in a dedicated env/profile.

## 3) Versioning Rules
- Pin `gymnasium` and `stable-baselines3` versions together in RL-specific install docs.
- Any version change requires:
  - changelog entry,
  - reproducibility check on at least one baseline training run,
  - update to run-manifest notes for impacted runs.

## 4) Runtime Guardrails
- Import RL frameworks lazily in policy training modules.
- No RL framework import is allowed in Fast Loop execution path.
- Teacher policies remain offline/slow-loop only; student policies are the only Fast Loop candidates.

## 5) CI Guidance
- Non-RL CI jobs must not fail if RL extras are not installed.
- RL-specific jobs should run in separate optional CI lanes gated by Phase 3 scope.
