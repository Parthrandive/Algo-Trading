from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from src.agents.consensus import (
    AgentSignal,
    ConsensusAgent,
    ConsensusInput,
    ConsensusRegimeRiskLevel,
)


def _payload(*, sentiment_is_stale: bool = False, sentiment_is_missing: bool = False) -> ConsensusInput:
    return ConsensusInput(
        technical=AgentSignal(name="technical", score=0.62, confidence=0.78),
        regime=AgentSignal(name="regime", score=0.44, confidence=0.74),
        sentiment=AgentSignal(name="sentiment", score=0.18, confidence=0.71),
        volatility=0.18,
        macro_differential=0.12,
        rbi_signal=0.04,
        sentiment_quantile=0.59,
        crisis_probability=0.10,
        sentiment_is_stale=sentiment_is_stale,
        sentiment_is_missing=sentiment_is_missing,
        regime_ood_warning=False,
        regime_ood_alien=False,
        regime_risk_level=ConsensusRegimeRiskLevel.FULL_RISK,
        generated_at_utc=datetime.now(UTC),
    )


def test_consensus_run_meets_sub_10ms_target():
    agent = ConsensusAgent()
    payload = _payload()

    # Warm up once to avoid one-time initialization overhead skewing the timing.
    agent.run(payload)

    started = perf_counter()
    agent.run(payload)
    latency_ms = (perf_counter() - started) * 1000.0

    assert latency_ms < 10.0


def test_offline_consensus_pipeline_stays_out_of_execution_path():
    source_path = Path(__file__).resolve().parents[1] / "src" / "agents" / "consensus" / "offline_pipeline.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    forbidden_prefixes = (
        "src.agents.strategic",
        "src.agents.execution",
        "src.agents.trader_ensemble",
    )

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)

    offenders = [
        module
        for module in imported_modules
        if any(module.startswith(prefix) for prefix in forbidden_prefixes)
    ]

    assert offenders == []


def test_stale_sentiment_reduces_consensus_confidence():
    agent = ConsensusAgent()

    fresh = agent.run(_payload(sentiment_is_stale=False, sentiment_is_missing=False))
    stale = agent.run(_payload(sentiment_is_stale=True, sentiment_is_missing=False))
    missing = agent.run(_payload(sentiment_is_stale=True, sentiment_is_missing=True))

    assert stale.confidence < fresh.confidence
    assert missing.confidence < stale.confidence
