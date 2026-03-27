# AGENTS.md — Multi-Agent AI Trading System
# Indian Market: NSE / USD-INR / MCX Gold | Plan v1.3.7

## MANDATORY FIRST STEP
Before writing any code, editing any file, or making any decision:
1. Read `trading-skills/INDEX.md`
2. Identify which skill(s) apply to the current task using the routing table
3. Read those specific SKILL.md files in full
4. Follow every rule in the skill — they are non-negotiable

## SKILL ROUTING (quick reference)
| Task type | Read this skill first |
|-----------|----------------------|
| Any data feed, ingestion, sentiment pipeline, preprocessing | `trading-skills/data-orchestration/SKILL.md` |
| Any model training, regime detection, consensus, sentiment model | `trading-skills/analyst-board/SKILL.md` |
| Any order placement, Fast Loop, latency, teacher-student inference | `trading-skills/execution-fastloop/SKILL.md` |
| Any risk control, kill switch, stress test, drawdown limit | `trading-skills/risk-overseer/SKILL.md` |
| Any model promotion, shadow testing, rollback, CI gate | `trading-skills/mlops-governance/SKILL.md` |
| Any order submission, universe filter, compliance, audit log | `trading-skills/compliance-universe/SKILL.md` |
| Any merge conflict, failing CI, PR checks red, branch behind | `trading-skills/green-prs/SKILL.md` |

## HARD RULES (always active, no exceptions)
- Risk Overseer rules override all other agents and all other skills
- Never place teacher policy inference in the Fast Loop execution path
- Never use fixed performance uplift claims (e.g. "will improve Sharpe by X") in any artifact
- Never enable Tier 3 items in the current cycle (FPGA, market-making, online RL)
- Fast Loop p99 must stay <= 8 ms; auto-degrade if > 10 ms
- Every order must pass all pre-trade compliance checks before submission
- All data sources must have documented licensing/public-availability justification
- Audit trail fields are mandatory on every trade event — never omit them

## PROJECT STATUS (v1.3.7)
- Phase 1 (Data Orchestration): COMPLETE
- Phase 2 (Analyst Board): COMPLETE; Technical Agent training ONGOING
- Tier 1 items: IN PROGRESS (implement now)
- Tier 2 items: after 3 months paper trading
- Tier 3 items: DEFERRED — do not implement

## GO-LIVE TARGETS (paper trading gates)
Sharpe >= 1.8 | Sortino >= 2.0 | MDD <= 8% | Win rate >= 52% |
Slippage <= model + 20 bps | Data uptime >= 99.5% | Zero compliance violations
