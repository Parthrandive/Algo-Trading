from __future__ import annotations

import argparse
import json
import sys
from statistics import mean
from time import perf_counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.strategic.action_space import export_week2_action_space
from src.agents.strategic.artifacts import Phase3RunManifest, resolve_code_hash, utc_now, write_manifest
from src.agents.strategic.config import (
    DEFAULT_ACTION_EXPORT_DIR,
    DEFAULT_ACTION_EXPORT_FILE,
    DEFAULT_RUN_MANIFEST_FILE,
    STRATEGIC_EXEC_CONTRACT_VERSION,
    WEEK2_ACTION_EXPORT_VERSION,
)
from src.agents.strategic.execution import ExecutionContext, ExecutionEngine, OrderRequest, OrderType
from src.agents.strategic.impact_monitor import FillEvent, InstrumentBucket
from src.agents.strategic.latency_discipline import BenchmarkEvidence, CILatencyBenchmarkGate
from src.agents.strategic.observation import ObservationAssembler
from src.agents.strategic.orderbook_features import OrderBookSnapshot
from src.agents.strategic.placeholder_teacher import generate_placeholder_teacher_actions
from src.agents.strategic.promotion_gates import PolicyStage, PromotionEvidence, RollbackDrillManager
from src.agents.strategic.risk_budgets import VolatilityReading
from src.agents.strategic.registry import register_placeholder_teacher_policies
from src.agents.strategic.schemas import ActionType, RiskMode
from src.agents.strategic.week4 import (
    PaperTradeDecision,
    StressScenarioResult,
    Week4Controller,
    build_week4_bundle,
)
from src.agents.strategic.week3 import build_week3_bundle
from src.db.connection import get_engine, get_session
from src.db.models import ConsensusSignalDB, RegimePredictionDB, TechnicalPredictionDB
from src.db.phase3_recorder import Phase3Recorder


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_symbols(value: str) -> list[str]:
    symbols = [item.strip() for item in value.split(",") if item.strip()]
    deduped: list[str] = []
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        deduped.append(symbol)
        seen.add(symbol)
    return deduped


def _resolve_latest_phase2_timestamp(symbols: list[str], database_url: str | None) -> datetime:
    engine = get_engine(database_url)
    session_factory = get_session(engine)
    latest_candidates: list[datetime] = []
    with session_factory() as session:
        for model in (TechnicalPredictionDB, RegimePredictionDB, ConsensusSignalDB):
            latest = session.query(func.max(model.timestamp)).filter(model.symbol.in_(symbols)).scalar()
            if isinstance(latest, datetime):
                if latest.tzinfo is None:
                    latest_candidates.append(latest.replace(tzinfo=timezone.utc))
                else:
                    latest_candidates.append(latest.astimezone(timezone.utc))

    if not latest_candidates:
        joined = ",".join(symbols)
        raise ValueError(
            f"No Phase 2 timestamps found for symbols [{joined}]. "
            "Generate Phase 2 signals first or provide explicit --start/--end."
        )
    return max(latest_candidates)


def _json_safe(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _bucket_for_symbol(symbol: str) -> InstrumentBucket:
    upper = symbol.upper()
    if "USDINR" in upper:
        return InstrumentBucket.USDINR
    if "GOLD" in upper:
        return InstrumentBucket.MCX_GOLD
    if upper.endswith(".NS"):
        return InstrumentBucket.LIQUID_LARGE_CAP
    return InstrumentBucket.MID_LIQUIDITY


def _cluster_for_symbol(symbol: str, default_cluster: str) -> str:
    upper = symbol.upper()
    if "USDINR" in upper:
        return "fx_usdinr"
    if "GOLD" in upper:
        return "mcx_gold"
    if upper.endswith(".NS"):
        return default_cluster
    return "nse_mid_liquidity"


def _build_orderbook_snapshot(symbol: str, timestamp: datetime, imbalance: float, queue_pressure: float, source_quality: str) -> OrderBookSnapshot:
    base_qty = 100.0
    normalized_imbalance = max(-0.95, min(0.95, float(imbalance)))
    bid_total = base_qty * (1.0 + normalized_imbalance)
    ask_total = base_qty * (1.0 - normalized_imbalance)
    bid_levels = tuple(max(1.0, bid_total / (level + 1)) for level in range(5))
    ask_levels = tuple(max(1.0, ask_total / (level + 1)) for level in range(5))
    bid_arrival = max(1.0, 100.0 * (1.0 + max(-0.95, min(0.95, float(queue_pressure)))))
    ask_arrival = max(1.0, 100.0 * (1.0 - max(-0.95, min(0.95, float(queue_pressure)))))
    return OrderBookSnapshot(
        symbol=symbol,
        timestamp=timestamp,
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        bid_arrival_rate=bid_arrival,
        ask_arrival_rate=ask_arrival,
        source_quality=source_quality,
    )


def _risk_mode_from_override(value: str | None) -> RiskMode:
    if not value:
        return RiskMode.NORMAL
    normalized = str(value).strip().lower()
    if normalized == "reduce_only":
        return RiskMode.REDUCE_ONLY
    if normalized == "close_only":
        return RiskMode.CLOSE_ONLY
    if normalized == "kill_switch":
        return RiskMode.KILL_SWITCH
    return RiskMode.NORMAL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DB-backed Phase 3 dry-run for placeholder teacher actions with optional "
            "Week 3 (Tier 1) and Week 4 (readiness) wiring."
        )
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. RELIANCE.NS,TCS.NS).")
    parser.add_argument("--start", required=True, help="ISO8601 start timestamp (UTC recommended).")
    parser.add_argument("--end", required=True, help="ISO8601 end timestamp (UTC recommended).")
    parser.add_argument("--database-url", default=None, help="Optional SQLAlchemy database URL override.")
    parser.add_argument("--policy-id", default="teacher_placeholder_v0", help="Placeholder teacher policy id.")
    parser.add_argument("--batch-size", type=int, default=500, help="Observation materialization batch size.")
    parser.add_argument("--output-dir", default=str(DEFAULT_ACTION_EXPORT_DIR), help="Artifact output directory.")
    parser.add_argument(
        "--action-export-file",
        default=DEFAULT_ACTION_EXPORT_FILE,
        help="Week 2 action-space export filename (JSONL).",
    )
    parser.add_argument(
        "--manifest-file",
        default=DEFAULT_RUN_MANIFEST_FILE,
        help="Run manifest filename.",
    )
    parser.add_argument(
        "--no-materialize-observations",
        action="store_true",
        help="Read from Phase 2 DB and build actions, but do not write observations table.",
    )
    parser.add_argument(
        "--write-decisions",
        action="store_true",
        help="Persist placeholder teacher actions to trade_decisions.",
    )
    parser.add_argument(
        "--register-teacher-model-cards",
        action="store_true",
        help="Optionally wire placeholder SAC/PPO/TD3 teachers into existing model registry tables.",
    )
    parser.add_argument(
        "--enable-week3",
        action="store_true",
        help="Run Week 3 Tier 1 wiring (impact/risk-budgets/orderbook/latency/promotion/XAI) and persist telemetry.",
    )
    parser.add_argument(
        "--week3-default-cluster",
        default="nse_large_cap",
        help="Default asset cluster used for volatility-scaled risk budget mapping.",
    )
    parser.add_argument(
        "--week3-sigma-baseline",
        type=float,
        default=0.02,
        help="Baseline sigma used by volatility-scaled risk budgets for the default cluster.",
    )
    parser.add_argument(
        "--week3-base-quantity",
        type=int,
        default=10_000,
        help="Reference quantity used to transform action_size into synthetic execution size.",
    )
    parser.add_argument(
        "--week3-adv",
        type=float,
        default=1_000_000.0,
        help="Synthetic ADV used for participation/impact checks in dry runs.",
    )
    parser.add_argument(
        "--week3-baseline-p99-ms",
        type=float,
        default=8.0,
        help="Baseline p99 used by latency CI benchmark artifact in dry runs.",
    )
    parser.add_argument(
        "--enable-week4",
        action="store_true",
        help="Run Week 4 readiness wiring (backtest/stress/paper/governance/gates) using dry-run artifacts.",
    )
    parser.add_argument(
        "--week4-outage-minutes",
        type=float,
        default=0.0,
        help="Synthetic paper-trading outage duration (minutes) to evaluate uptime handling.",
    )
    parser.add_argument(
        "--week4-crisis-agreement",
        type=float,
        default=0.86,
        help="Synthetic teacher-student crisis agreement used for Week 4 gate evidence.",
    )
    parser.add_argument(
        "--week4-blocking-defects",
        type=int,
        default=0,
        help="Open blocking defects count used by Week 4 GO/NO-GO evaluation.",
    )
    parser.add_argument(
        "--week4-compliance-violations",
        type=int,
        default=0,
        help="Critical compliance violations count used by Week 4 GO/NO-GO evaluation.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.database_url and ("<" in args.database_url or ">" in args.database_url):
        raise ValueError(
            "database-url contains placeholder text. Replace <user>/<pass>/<host>/<port>/<db> with actual values."
        )

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise ValueError("No symbols provided.")
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    if end < start:
        raise ValueError("end must be greater than or equal to start.")

    run_started = utc_now()
    run_id = f"phase3_week1_dry_run_{run_started.strftime('%Y%m%dT%H%M%SZ')}"
    output_root = Path(args.output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    assembler = ObservationAssembler(database_url=args.database_url)
    recorder = Phase3Recorder(database_url=args.database_url)

    # Build observations directly from real Phase 2 DB tables.
    observations = []
    per_symbol_counts: dict[str, int] = {}
    warn_count = 0
    for symbol in symbols:
        rows = assembler.build_symbol_observations(symbol=symbol, start=start, end=end)
        per_symbol_counts[symbol] = len(rows)
        warn_count += sum(1 for row in rows if row.quality_status != "pass")
        observations.extend(rows)
        if not args.no_materialize_observations and rows:
            for idx in range(0, len(rows), max(1, args.batch_size)):
                chunk = rows[idx : idx + max(1, args.batch_size)]
                recorder.save_observation_batch([item.model_dump(mode="python") for item in chunk])

    actions = generate_placeholder_teacher_actions(observations, policy_id=args.policy_id)

    rewards = []
    for action in actions:
        action_name = action.action.value
        directional_bonus = 1.0 if action_name in {"buy", "sell"} else 0.0
        reward_value = float(action.confidence) * directional_bonus - (0.01 * float(action.action_size))
        rewards.append(
            {
                "timestamp": action.timestamp,
                "symbol": action.symbol,
                "policy_id": action.policy_id,
                "reward_name": "ra_drl_composite_placeholder",
                "reward_value": reward_value,
                "components": {
                    "confidence": float(action.confidence),
                    "directional_bonus": directional_bonus,
                    "size_penalty": 0.01 * float(action.action_size),
                    "placeholder": True,
                },
            }
        )

    week3_enabled = bool(args.enable_week3)
    week3_decisions_payload: list[dict[str, object]] = [action.model_dump(mode="json") for action in actions]
    week3_summary: dict[str, object] = {}
    latency_gate_payload: dict[str, object] | None = None
    promotion_payload: list[dict[str, object]] = []
    rollback_payload: dict[str, object] | None = None
    week4_enabled = bool(args.enable_week4)
    week4_summary: dict[str, object] = {}
    week4_gate_payload: dict[str, object] | None = None
    week4_gonogo_payload: dict[str, object] | None = None

    if week3_enabled:
        sigma_baselines = {
            args.week3_default_cluster: float(args.week3_sigma_baseline),
            "nse_mid_liquidity": float(args.week3_sigma_baseline),
            "fx_usdinr": float(args.week3_sigma_baseline),
            "mcx_gold": float(args.week3_sigma_baseline),
        }
        bundle = build_week3_bundle(sigma_baseline_by_cluster=sigma_baselines)
        execution_engine = ExecutionEngine(impact_monitor=bundle.impact_monitor)
        ci_gate = CILatencyBenchmarkGate()

        impact_rows: list[dict[str, object]] = []
        risk_cap_rows: list[dict[str, object]] = []
        orderbook_rows: list[dict[str, object]] = []
        xai_rows: list[dict[str, object]] = []
        pnl_rows: list[dict[str, object]] = []

        for index, action in enumerate(actions):
            if index >= len(observations):
                break
            observation = observations[index]
            started_ns = perf_counter()
            decision_payload = action.model_dump(mode="json")
            decision_payload["is_placeholder"] = True

            orderbook_snapshot = _build_orderbook_snapshot(
                symbol=observation.symbol,
                timestamp=observation.timestamp,
                imbalance=observation.orderbook_imbalance or 0.0,
                queue_pressure=observation.queue_pressure or 0.0,
                source_quality=observation.quality_status,
            )
            orderbook_features = bundle.orderbook_features.compute(orderbook_snapshot)
            orderbook_rows.append(
                {
                    "timestamp": orderbook_features.timestamp,
                    "symbol": orderbook_features.symbol,
                    "quality_flag": orderbook_features.quality_flag.value,
                    "degraded": orderbook_features.degraded,
                    "degradation_reason": orderbook_features.degradation_reason,
                    "imbalance": orderbook_features.imbalance,
                    "queue_pressure": orderbook_features.queue_pressure,
                }
            )

            asset_cluster = _cluster_for_symbol(observation.symbol, args.week3_default_cluster)
            realized_vol = max(abs(float(observation.var_95)), abs(float(observation.es_95)))
            risk_cap = bundle.risk_budgets.update(
                VolatilityReading(
                    symbol=observation.symbol,
                    asset_cluster=asset_cluster,
                    realized_vol=realized_vol,
                    timestamp=observation.timestamp,
                )
            )
            risk_cap_rows.append(
                {
                    "timestamp": observation.timestamp,
                    "symbol": risk_cap.symbol,
                    "asset_cluster": risk_cap.asset_cluster,
                    "regime": risk_cap.regime.value,
                    "cap_fraction": risk_cap.cap_fraction,
                    "changed": risk_cap.changed,
                    "event_type": risk_cap.event_type,
                    "false_trigger_rate": risk_cap.false_trigger_rate,
                    "auto_adjustment_paused": risk_cap.auto_adjustment_paused,
                }
            )

            action_quantity = max(0, int(float(action.action_size) * int(args.week3_base_quantity)))
            model_slippage_bps = 10.0
            realized_slippage_bps = model_slippage_bps + (1.0 - float(action.confidence)) * 30.0
            impact_decision = None
            if action_quantity > 0 and action.action in {ActionType.BUY, ActionType.SELL, ActionType.REDUCE, ActionType.CLOSE}:
                impact_decision = bundle.impact_monitor.evaluate_fill(
                    FillEvent(
                        symbol=action.symbol,
                        bucket=_bucket_for_symbol(action.symbol),
                        quantity=action_quantity,
                        adv=float(args.week3_adv),
                        model_slippage_bps=model_slippage_bps,
                        realized_slippage_bps=realized_slippage_bps,
                        timestamp=action.timestamp,
                    )
                )
                impact_rows.append(
                    {
                        "timestamp": action.timestamp,
                        "symbol": impact_decision.symbol,
                        "bucket": impact_decision.bucket.value,
                        "event_type": impact_decision.event_type,
                        "breach": impact_decision.breach,
                        "participation_rate": impact_decision.participation_rate,
                        "slippage_delta_bps": impact_decision.slippage_delta_bps,
                        "impact_score": impact_decision.impact_score,
                        "size_multiplier": impact_decision.size_multiplier,
                        "cooldown_until": impact_decision.cooldown_until,
                        "risk_override": impact_decision.risk_override,
                        "reasons": impact_decision.reasons,
                    }
                )

                direction = "BUY" if action.action == ActionType.BUY else "SELL"
                execution_plan = execution_engine.plan_execution(
                    request=OrderRequest(
                        symbol=action.symbol,
                        direction=direction,
                        target_quantity=action_quantity,
                        target_notional=float(action_quantity) * float(max(observation.price_forecast, 1.0)),
                        confidence=float(action.confidence),
                        order_type=OrderType.LIMIT,
                        metadata={"placeholder": True, "week3_enabled": True},
                    ),
                    context=ExecutionContext(
                        timestamp=action.timestamp,
                        symbol=action.symbol,
                        current_price=float(max(observation.price_forecast, 1.0)),
                        orderbook_imbalance=float(orderbook_features.imbalance),
                        queue_pressure=float(orderbook_features.queue_pressure),
                        avg_volume_1h=float(args.week3_adv),
                    ),
                )
                decision_payload["action_size"] = round(
                    float(execution_plan.instruction.quantity) / float(max(1, args.week3_base_quantity)),
                    6,
                )
                decision_payload["risk_override"] = (
                    impact_decision.risk_override if impact_decision and impact_decision.risk_override else decision_payload.get("risk_override")
                )
                decision_payload["risk_override_reason"] = ",".join(impact_decision.reasons) if impact_decision else None
                decision_payload["metadata"] = {
                    **(decision_payload.get("metadata") or {}),
                    "compliance_passed": execution_plan.compliance.passed,
                    "routing_status": execution_plan.routing_status,
                    "estimated_slippage_bps": execution_plan.estimated_slippage_bps,
                    "quality_status": observation.quality_status,
                }

            trade_id = f"{run_id}:{index}"
            bundle.xai_logger.mark_trade_seen(trade_id)
            explanation = bundle.xai_logger.log_trade(
                trade_id=trade_id,
                symbol=action.symbol,
                timestamp=action.timestamp,
                feature_contributions={
                    "technical_confidence": float(observation.technical_confidence),
                    "consensus_confidence": float(observation.consensus_confidence),
                    "sentiment_z_t": float(observation.sentiment_z_t or 0.0),
                    "var_95": float(observation.var_95),
                    "es_95": float(observation.es_95),
                    "orderbook_imbalance": float(orderbook_features.imbalance),
                    "queue_pressure": float(orderbook_features.queue_pressure),
                },
                agent_contributions={
                    "technical": float(observation.technical_confidence),
                    "regime": float(observation.regime_transition_prob),
                    "sentiment": abs(float(observation.sentiment_score or 0.0)),
                    "consensus": float(observation.consensus_confidence),
                },
                signal_family_contributions={
                    "microstructure": abs(float(orderbook_features.imbalance)),
                    "consensus": float(observation.consensus_confidence),
                    "risk": abs(float(observation.var_95)),
                },
                metadata={"week3_enabled": True, "is_placeholder": True},
            )
            xai_rows.append(
                {
                    "trade_id": explanation.trade_id,
                    "timestamp": explanation.timestamp,
                    "symbol": explanation.symbol,
                    "top_feature_contributions": explanation.top_feature_contributions,
                    "agent_contributions": explanation.agent_contributions,
                    "signal_family_contributions": explanation.signal_family_contributions,
                    "metadata": explanation.metadata,
                }
            )

            reward_value = float(rewards[index]["reward_value"]) if index < len(rewards) else 0.0
            sector = "fx" if "USDINR" in action.symbol.upper() else "commodity" if "GOLD" in action.symbol.upper() else "equity"
            agent = "consensus" if action.confidence >= 0.5 else "technical"
            signal_family = "microstructure" if abs(orderbook_features.imbalance) > 0.2 else "consensus"
            bundle.pnl_attribution.add_event(
                trade_id=trade_id,
                symbol=action.symbol,
                sector=sector,
                agent=agent,
                signal_family=signal_family,
                realized_pnl=reward_value,
                timestamp=action.timestamp,
            )
            pnl_rows.append(
                {
                    "trade_id": trade_id,
                    "timestamp": action.timestamp,
                    "symbol": action.symbol,
                    "sector": sector,
                    "agent": agent,
                    "signal_family": signal_family,
                    "realized_pnl": reward_value,
                }
            )

            staleness_seconds = max(0.0, (run_started - observation.timestamp).total_seconds())
            feature_lag_seconds = max(0.0, (run_started - orderbook_features.timestamp).total_seconds())
            bundle.operational_metrics.add_decision_staleness(staleness_seconds)
            bundle.operational_metrics.add_feature_lag(feature_lag_seconds)
            if observation.crisis_mode or observation.agent_divergence:
                bundle.operational_metrics.increment_mode_switch()
            if abs(observation.regime_transition_prob) > 0.8:
                bundle.operational_metrics.increment_ood_trigger()
            if impact_decision and impact_decision.event_type == "IMPACT_BREACH" and not impact_decision.breach:
                bundle.operational_metrics.increment_kill_switch_false_positive()

            elapsed_ms = (perf_counter() - started_ns) * 1000.0
            bundle.latency_discipline.record_stage_latency("decision_path", elapsed_ms)
            decision_payload["decision_latency_ms"] = round(elapsed_ms, 6)
            week3_decisions_payload[index] = decision_payload

        latency_mode = bundle.latency_discipline.evaluate_mode(stage="decision_path")
        latency_summary = bundle.latency_discipline.summarize(stage="decision_path")
        recorder.save_fastloop_latency_event(
            {
                "timestamp": utc_now(),
                "stage": "decision_path",
                "mode": latency_mode.mode,
                "event_type": latency_mode.event_type,
                "reason": latency_mode.reason,
                "sample_count": latency_summary.count,
                "p50_ms": latency_summary.p50_ms,
                "p95_ms": latency_summary.p95_ms,
                "p99_ms": latency_summary.p99_ms,
                "p999_ms": latency_summary.p999_ms,
                "jitter_ms": latency_summary.jitter_ms,
            }
        )

        benchmark = ci_gate.evaluate(
            BenchmarkEvidence(
                replay_p99_ms=latency_summary.p99_ms,
                replay_p999_ms=latency_summary.p999_ms,
                peak_p99_ms=max(latency_summary.p99_ms, latency_summary.p95_ms * 1.1),
                peak_p999_ms=max(latency_summary.p999_ms, latency_summary.p99_ms * 1.05),
                baseline_p99_ms=float(args.week3_baseline_p99_ms),
                correctness_pass=True,
                degrade_path_pass=latency_mode.mode in {"normal", "degraded"},
            )
        )
        latency_gate_payload = {
            "run_id": run_id,
            "timestamp": utc_now(),
            "passed": benchmark.passed,
            "reasons": benchmark.reasons,
            "artifact": benchmark.artifact,
        }
        recorder.save_latency_benchmark_artifact(latency_gate_payload)

        average_reward = mean(float(item["reward_value"]) for item in rewards) if rewards else 0.0
        gate_evidence = PromotionEvidence(
            walk_forward_outperformance=average_reward >= 0.0,
            non_regression=True,
            crisis_slice_agreement=0.85,
            latency_gate_passed=benchmark.passed,
            rollback_ready=True,
            critical_compliance_violations=0,
        )
        shadow_decision = bundle.promotion_gates.transition(
            current_stage=PolicyStage.CANDIDATE,
            requested_stage=PolicyStage.SHADOW,
            evidence=gate_evidence,
        )
        champion_decision = bundle.promotion_gates.transition(
            current_stage=shadow_decision.to_stage,
            requested_stage=PolicyStage.CHAMPION,
            evidence=gate_evidence,
        )
        promotion_payload = [
            {
                "timestamp": shadow_decision.evaluated_at,
                "policy_id": args.policy_id,
                "from_stage": shadow_decision.from_stage.value,
                "to_stage": shadow_decision.to_stage.value,
                "approved": shadow_decision.approved,
                "reasons": shadow_decision.reasons,
                "evidence": {
                    "walk_forward_outperformance": gate_evidence.walk_forward_outperformance,
                    "non_regression": gate_evidence.non_regression,
                    "crisis_slice_agreement": gate_evidence.crisis_slice_agreement,
                    "latency_gate_passed": gate_evidence.latency_gate_passed,
                    "rollback_ready": gate_evidence.rollback_ready,
                },
            },
            {
                "timestamp": champion_decision.evaluated_at,
                "policy_id": args.policy_id,
                "from_stage": champion_decision.from_stage.value,
                "to_stage": champion_decision.to_stage.value,
                "approved": champion_decision.approved,
                "reasons": champion_decision.reasons,
                "evidence": {
                    "walk_forward_outperformance": gate_evidence.walk_forward_outperformance,
                    "non_regression": gate_evidence.non_regression,
                    "crisis_slice_agreement": gate_evidence.crisis_slice_agreement,
                    "latency_gate_passed": gate_evidence.latency_gate_passed,
                    "rollback_ready": gate_evidence.rollback_ready,
                },
            },
        ]
        for event in promotion_payload:
            recorder.save_promotion_gate_event(event)

        rollback_manager = RollbackDrillManager()
        rollback_manager.set_champion(f"{args.policy_id}_v1")
        rollback_manager.set_champion(f"{args.policy_id}_v2")
        rollback_result = rollback_manager.rollback(
            failed_model_id=f"{args.policy_id}_v2",
            started_at=run_started,
            ended_at=utc_now(),
        )
        rollback_payload = {
            "timestamp": rollback_result.timestamp,
            "failed_model_id": f"{args.policy_id}_v2",
            "reverted_to": rollback_result.reverted_to,
            "executed": rollback_result.executed,
            "mttr_seconds": rollback_result.mttr_seconds,
            "reasons": rollback_result.reasons,
        }
        recorder.save_rollback_drill_event(rollback_payload)
        bundle.operational_metrics.record_mttr(rollback_result.mttr_seconds)

        recorder.save_impact_event_batch(impact_rows)
        recorder.save_risk_cap_event_batch(risk_cap_rows)
        recorder.save_orderbook_feature_event_batch(orderbook_rows)
        recorder.save_xai_trade_explanation_batch(xai_rows)
        recorder.save_pnl_attribution_event_batch(pnl_rows)
        ops_snapshot = bundle.operational_metrics.snapshot()
        recorder.save_operational_metrics_snapshot(
            {
                "timestamp": ops_snapshot["updated_at"],
                "decision_staleness_avg_s": ops_snapshot["decision_staleness_avg_s"],
                "feature_lag_avg_s": ops_snapshot["feature_lag_avg_s"],
                "mode_switch_frequency": ops_snapshot["mode_switch_frequency"],
                "ood_trigger_rate": ops_snapshot["ood_trigger_rate"],
                "kill_switch_false_positives": ops_snapshot["kill_switch_false_positives"],
                "mttr_avg_s": ops_snapshot["mttr_avg_s"],
            }
        )
        week3_summary = {
            "impact_events": len(impact_rows),
            "risk_cap_events": len(risk_cap_rows),
            "orderbook_feature_events": len(orderbook_rows),
            "xai_rows": len(xai_rows),
            "pnl_rows": len(pnl_rows),
            "latency_mode": latency_mode.mode,
            "latency_p99_ms": latency_summary.p99_ms,
            "latency_gate_passed": benchmark.passed,
            "xai_coverage": bundle.xai_logger.coverage(),
        }

    if week4_enabled:
        week4_controller = Week4Controller(build_week4_bundle())

        reward_values = [float(item["reward_value"]) for item in rewards]
        if reward_values:
            split_at = max(1, len(reward_values) // 2)
            returns_by_fold = [reward_values[:split_at], reward_values[split_at:] or reward_values[:split_at]]
        else:
            returns_by_fold = [[]]

        leakage_rows = []
        for action, observation in zip(actions, observations):
            leakage_rows.append(
                {
                    "observation_timestamp": observation.timestamp,
                    "feature_timestamp": observation.timestamp,
                    "decision_timestamp": action.timestamp + timedelta(seconds=1),
                }
            )

        universe_rows = [
            {
                "symbol": symbol,
                "as_of": run_started,
                "point_in_time_version": "phase3_week4_pit_v1",
                "is_active": True,
                "delisted_at": None,
            }
            for symbol in symbols
        ]
        if symbols:
            universe_rows.append(
                {
                    "symbol": f"DELISTED_{symbols[0]}",
                    "as_of": run_started,
                    "point_in_time_version": "phase3_week4_pit_v1",
                    "is_active": False,
                    "delisted_at": run_started - timedelta(days=365),
                }
            )

        day1 = week4_controller.run_day1_backtesting(
            returns_by_fold=returns_by_fold,
            leakage_rows=leakage_rows,
            universe_rows=universe_rows,
        )

        day2 = week4_controller.run_day2_stress(
            [
                StressScenarioResult(
                    scenario_id="rbi_surprise_rate_hike",
                    protective_mode=RiskMode.REDUCE_ONLY,
                    expected_min_mode=RiskMode.REDUCE_ONLY,
                    snapback_ticks=9,
                    max_snapback_ticks=30,
                    capacity_multiplier=1.0,
                    impact_bps=18.0,
                ),
                StressScenarioResult(
                    scenario_id="liquidity_drought_3x",
                    protective_mode=RiskMode.CLOSE_ONLY,
                    expected_min_mode=RiskMode.REDUCE_ONLY,
                    snapback_ticks=14,
                    max_snapback_ticks=30,
                    capacity_multiplier=3.0,
                    impact_bps=22.0,
                ),
            ]
        )

        paper_decisions: list[PaperTradeDecision] = []
        for index, action in enumerate(actions):
            if index >= len(observations):
                break
            if action.action == ActionType.HOLD:
                continue
            observation = observations[index]
            decision_payload = week3_decisions_payload[index] if index < len(week3_decisions_payload) else {}
            risk_override = decision_payload.get("risk_override") if isinstance(decision_payload, dict) else None
            paper_decisions.append(
                PaperTradeDecision(
                    timestamp=action.timestamp,
                    symbol=action.symbol,
                    direction="BUY" if action.action == ActionType.BUY else "SELL",
                    quantity=max(1, int(float(action.action_size) * int(args.week3_base_quantity))),
                    confidence=float(action.confidence),
                    price=float(max(observation.price_forecast, 1.0)),
                    signal_source="strategic_ensemble",
                    model_version=f"{args.policy_id}_student_placeholder",
                    universe_version="phase3_week4_pit_v1",
                    data_source_tags=("primary_api",),
                    risk_mode=_risk_mode_from_override(risk_override if isinstance(risk_override, str) else None),
                    order_type=OrderType.LIMIT,
                )
            )
        day3 = week4_controller.run_day3_paper_trading(
            paper_decisions,
            outage_minutes=float(args.week4_outage_minutes),
            seed=11,
        )

        code_hash = resolve_code_hash() or "unknown"
        governance_models = [
            {
                "model_id": f"{args.policy_id}_artifact_{idx}",
                "version": "1.0.0",
                "training_data_snapshot_hash": f"{run_id}:dataset:{idx}",
                "code_hash": code_hash,
                "feature_schema_version": observations[0].observation_schema_version if observations else "1.0",
                "hyperparameters": {"seed": 42 + idx},
                "validation_metrics": {"sharpe": 2.0, "sortino": 2.2},
                "baseline_comparison": {"status": "non_regression"},
                "plan_version": "v1.3.7",
                "created_by": "owner",
                "reviewed_by": "partner",
            }
            for idx in range(6)
        ]
        rollback_for_week4 = week4_controller.bundle.governance.run_rollback_drill(
            champions=(f"{args.policy_id}_student_v1", f"{args.policy_id}_student_v2"),
            failed_model_id=f"{args.policy_id}_student_v2",
            started_at=run_started,
            ended_at=utc_now(),
        )
        day4 = week4_controller.run_day4_governance(
            models=governance_models,
            rollback_result=rollback_for_week4,
        )

        tier1_status = {
            "impact_monitor_functional": bool(week3_enabled),
            "risk_budgets_functional": bool(week3_enabled),
            "orderbook_imbalance_integrated": bool(week3_enabled),
            "latency_ci_gate_ready": bool(latency_gate_payload and latency_gate_payload.get("passed")),
        }
        evidence, assessment = week4_controller.run_day5_and_day6_gates(
            latency_p99_ms=float(week3_summary.get("latency_p99_ms", 0.0)),
            latency_p999_ms=float(week3_summary.get("latency_p99_ms", 0.0)),
            degrade_path_passed=bool(latency_gate_payload.get("passed")) if latency_gate_payload else bool(week3_enabled),
            crisis_slice_agreement=float(args.week4_crisis_agreement),
            rollback_result=rollback_for_week4,
            backtest_metrics=day1["walk_forward"].aggregate_metrics,
            observation_schema_version=observations[0].observation_schema_version if observations else "1.0",
            tier1_status=tier1_status,
            compliance_violations=int(args.week4_compliance_violations),
            blocking_defects=int(args.week4_blocking_defects),
        )

        week4_gate_payload = {
            "latency_gate_passed": evidence.latency_gate_passed,
            "crisis_agreement_passed": evidence.crisis_agreement_passed,
            "rollback_tested": evidence.rollback_tested,
            "walk_forward_passed": evidence.walk_forward_passed,
            "observation_schema_validated": evidence.observation_schema_validated,
            "tier1_operational": evidence.tier1_operational,
            "compliance_audit_complete": evidence.compliance_audit_complete,
            "blocking_defects": evidence.blocking_defects,
            "checks": evidence.checks,
        }
        week4_gonogo_payload = {
            "go": assessment.go,
            "checklist": assessment.checklist,
            "at_risk_items": assessment.at_risk_items,
        }
        week4_summary = {
            "backtest_target_pass": day1["walk_forward"].target_pass,
            "leakage_passed": day1["leakage"].passed,
            "survivorship_passed": day1["survivorship"].passed,
            "stress_passed": day2.passed,
            "paper_uptime_ratio": day3.uptime_ratio,
            "paper_audit_trail_complete": day3.audit_trail_complete,
            "governance_registry_complete": day4.registry_complete,
            "governance_reproducibility_ready": day4.reproducibility_ready,
            "governance_promotion_gate_passed": day4.promotion_gate_passed,
            "governance_rollback_passed": day4.rollback_passed,
            "gate_evidence": week4_gate_payload,
            "go_no_go": week4_gonogo_payload,
        }

    if args.write_decisions and actions:
        payload = []
        for row in week3_decisions_payload:
            row = dict(row)
            row["is_placeholder"] = True
            payload.append(row)
        recorder.save_trade_decision_batch(payload)

    if rewards:
        recorder.save_reward_log_batch(rewards)

    export_path = export_week2_action_space(output_root / args.action_export_file, actions)

    policy_ids = []
    training_runs_written = 0
    if args.register_teacher_model_cards:
        policy_ids = register_placeholder_teacher_policies(
            database_url=args.database_url,
            run_id=run_id,
            artifact_root=output_root,
            export_path=export_path,
            register_model_cards=True,
        )
        for policy_id in policy_ids:
            elapsed = max(0.0, (utc_now() - run_started).total_seconds())
            recorder.save_rl_training_run(
                {
                    "policy_id": policy_id,
                    "run_timestamp": run_started,
                    "training_start": start,
                    "training_end": end,
                    "episodes": max(1, len(observations)),
                    "total_steps": max(1, len(observations)),
                    "final_reward": 0.0,
                    "dataset_snapshot_id": run_id,
                    "code_hash": resolve_code_hash(),
                    "duration_seconds": elapsed,
                    "notes": "placeholder week1 dry-run training registration",
                }
            )
            training_runs_written += 1

    run_finished = utc_now()
    manifest = Phase3RunManifest(
        run_id=run_id,
        started_at_utc=run_started,
        finished_at_utc=run_finished,
        symbols=symbols,
        observation_schema_version=observations[0].observation_schema_version if observations else "1.0",
        contract_version=STRATEGIC_EXEC_CONTRACT_VERSION,
        export_schema_version=WEEK2_ACTION_EXPORT_VERSION,
        rows_materialized=0 if args.no_materialize_observations else len(observations),
        actions_generated=len(actions),
        code_hash=resolve_code_hash(),
        dataset_snapshot={
            "phase2_window": {"start": start.isoformat(), "end": end.isoformat()},
            "per_symbol_observation_rows": per_symbol_counts,
            "warn_quality_rows": warn_count,
        },
        artifacts={
            "action_export_jsonl": str(export_path),
            "write_decisions": bool(args.write_decisions),
            "materialized_observations": not args.no_materialize_observations,
            "reward_logs_written": len(rewards),
            "training_runs_written": training_runs_written,
            "registered_policy_ids": policy_ids,
            "week3_enabled": week3_enabled,
            "week3_summary": week3_summary,
            "week3_latency_gate": latency_gate_payload,
            "week3_promotion_events": promotion_payload,
            "week3_rollback_event": rollback_payload,
            "week4_enabled": week4_enabled,
            "week4_summary": week4_summary,
            "week4_gate_evidence": week4_gate_payload,
            "week4_go_no_go": week4_gonogo_payload,
        },
        notes=[
            "Teacher actions are placeholder-only and restricted to offline/slow-loop usage.",
            "Fast Loop exclusion for teacher inference remains enforced by contract schema.",
        ],
    )
    manifest_path = write_manifest(output_root / args.manifest_file, manifest)

    summary = {
        "run_id": run_id,
        "symbols": symbols,
        "rows_built": len(observations),
        "rows_materialized": 0 if args.no_materialize_observations else len(observations),
        "actions_generated": len(actions),
        "reward_logs_written": len(rewards),
        "training_runs_written": training_runs_written,
        "action_export": str(export_path),
        "manifest": str(manifest_path),
        "registered_policy_ids": policy_ids,
        "week3_enabled": week3_enabled,
        "week3_summary": week3_summary,
        "week3_latency_gate": latency_gate_payload,
        "week4_enabled": week4_enabled,
        "week4_summary": week4_summary,
        "week4_gate_evidence": week4_gate_payload,
        "week4_go_no_go": week4_gonogo_payload,
    }
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
