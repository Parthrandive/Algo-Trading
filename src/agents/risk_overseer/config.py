from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskOverseerConfig:
    student_drift_alert_threshold: float = 0.12
    teacher_divergence_threshold: float = 0.15
    max_drawdown_limit: float = 0.08
    daily_loss_limit: float = 0.03
    broker_rejection_rate_limit: float = 0.10
    max_recovery_steps_per_cycle: int = 1
    manual_operator_ack_required: bool = True
    service_name: str = "independent_risk_overseer"
    schema_version: str = "phase4_risk_overseer_v1"
