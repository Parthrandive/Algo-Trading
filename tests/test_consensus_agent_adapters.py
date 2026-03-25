from __future__ import annotations

import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.agents.consensus import ConsensusAgent, build_consensus_input_from_phase2_payload
from src.agents.consensus.schemas import ConsensusRegimeRiskLevel, ConsensusRiskMode
from src.db.models import Base, ModelCardDB
from src.db.phase2_recorder import Phase2Recorder


def _build_sqlite_recorder() -> tuple:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory, Phase2Recorder(engine=engine, session_factory=session_factory)


def test_build_consensus_input_from_phase2_payload_maps_fields():
    payload = {
        "technical": {"score": 0.4, "confidence": 0.7, "is_protective": False},
        "regime": {
            "score": 0.2,
            "confidence": 0.8,
            "is_protective": True,
            "risk_level": "neutral_cash",
            "ood_warning": True,
        },
        "sentiment": {
            "score": -0.1,
            "confidence": 0.65,
            "is_protective": False,
            "freshness_flag": "stale",
            "source_count": 3,
        },
        "context": {
            "volatility": 0.42,
            "macro_differential": 0.25,
            "rbi_signal": -0.1,
            "sentiment_quantile": 0.55,
            "crisis_probability": 0.4,
            "generated_at_utc": "2026-03-16T10:30:00+00:00",
        },
    }

    consensus_input = build_consensus_input_from_phase2_payload(payload)

    assert consensus_input.technical.score == 0.4
    assert consensus_input.regime.is_protective is True
    assert consensus_input.sentiment.confidence == 0.65
    assert consensus_input.volatility == 0.42
    assert consensus_input.crisis_probability == 0.4
    assert consensus_input.sentiment_is_stale is True
    assert consensus_input.sentiment_is_missing is False
    assert consensus_input.regime_ood_warning is True
    assert consensus_input.regime_risk_level == ConsensusRegimeRiskLevel.NEUTRAL_CASH


def test_consensus_agent_loads_runtime_config_defaults_and_runs():
    agent = ConsensusAgent.from_default_components()
    payload = {
        "technical": {"score": 0.6, "confidence": 0.75},
        "regime": {"score": 0.5, "confidence": 0.72},
        "sentiment": {"score": 0.1, "confidence": 0.66},
        "context": {
            "volatility": 0.2,
            "macro_differential": 0.1,
            "rbi_signal": 0.05,
            "sentiment_quantile": 0.61,
            "crisis_probability": 0.15,
            "generated_at_utc": "2026-03-16T10:30:00+00:00",
        },
    }

    consensus_input = build_consensus_input_from_phase2_payload(payload)
    result = agent.run(consensus_input)

    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.transition_score <= 1.0
    assert result.risk_mode in {
        ConsensusRiskMode.NORMAL,
        ConsensusRiskMode.REDUCED,
        ConsensusRiskMode.PROTECTIVE,
    }


def test_consensus_agent_registers_model_cards_from_latest_run(tmp_path):
    config_path = tmp_path / "consensus_runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "transition": {"volatility_switch_threshold": 0.35, "lstar_logistic_gain": 4.0},
                "routing": {"max_crisis_weight": 0.7, "safety_bias_boost": 0.1},
                "risk_modes": {
                    "divergence_warn_threshold": 0.45,
                    "divergence_protective_threshold": 0.75,
                    "reduced_mode_scale": 0.5,
                },
                "weights": {"technical_base": 0.42, "regime_base": 0.35, "sentiment_base": 0.23},
                "freshness": {
                    "stale_sentiment_weight_multiplier": 0.4,
                    "missing_sentiment_weight_multiplier": 0.05,
                    "stale_confidence_penalty": 0.05,
                    "missing_confidence_penalty": 0.1,
                },
                "ood": {
                    "warning_regime_weight_multiplier": 1.1,
                    "warning_technical_weight_multiplier": 0.9,
                    "warning_confidence_penalty": 0.15,
                    "alien_regime_weight_multiplier": 1.2,
                    "alien_technical_weight_multiplier": 0.8,
                    "alien_sentiment_weight_multiplier": 0.6,
                    "alien_confidence_penalty": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )

    training_runs_root = tmp_path / "reports" / "training_runs"
    run_dir = training_runs_root / "phase2_consensus_20260319_162445"
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "leakage_checks": {
                    "macro_asof_safe_violations": 0,
                    "text_timestamp_alignment_violations": 0,
                    "consensus_train_rows_from_oos_only": True,
                },
                "consensus_weighted_test": {"macro_f1": 0.15, "proxy_utility_after_costs": 0.001},
                "consensus_challenger_test": {"macro_f1": 0.30, "proxy_utility_after_costs": -0.002},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "report_path": str(run_dir / "phase2_consensus_report.md"),
                "metrics_path": str(metrics_path),
                "recommendation": {
                    "recommended_model": "challenger_lstar_estar_bayesian",
                    "reason": "Better macro_f1 in OOS replay.",
                },
            }
        ),
        encoding="utf-8",
    )

    session_factory, recorder = _build_sqlite_recorder()
    agent = ConsensusAgent.from_default_components(
        runtime_config_path=config_path,
        training_runs_root=training_runs_root,
        model_cards_root=tmp_path / "data" / "models",
        phase2_recorder=recorder,
    )

    cards = agent.register_model_cards()

    weighted_card = tmp_path / "data" / "models" / "consensus_weighted_v1" / "model_card.json"
    challenger_card = tmp_path / "data" / "models" / "consensus_challenger_v1" / "model_card.json"
    assert weighted_card.exists()
    assert challenger_card.exists()
    assert cards["consensus_weighted_v1"]["hyperparameters"]["weights"]["technical_base"] == 0.42
    assert cards["consensus_challenger_v1"]["baseline_comparison"]["recommended_model"] == "challenger_lstar_estar_bayesian"

    with session_factory() as session:
        stored_cards = session.execute(select(ModelCardDB).order_by(ModelCardDB.model_id)).scalars().all()

    assert [card.model_id for card in stored_cards] == [
        "consensus_challenger_v1",
        "consensus_weighted_v1",
    ]
