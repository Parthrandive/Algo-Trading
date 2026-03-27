from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class RiskOverseerConfig:
    student_drift_alert_threshold: float = 0.12
    teacher_divergence_threshold: float = 0.15
    max_drawdown_limit: float = 0.08
    daily_loss_limit: float = 0.03
    broker_rejection_rate_limit: float = 0.10
    max_recovery_steps_per_cycle: int = 1
    manual_operator_ack_required: bool = True
    full_crisis_vol_multiplier: float = 1.5
    full_crisis_liquidity_floor: float = 0.35
    full_crisis_confidence_floor: float = 0.35
    crisis_hysteresis_ticks: int = 3
    crisis_max_duration: timedelta = timedelta(minutes=30)
    crisis_weight_cap: float = 0.70
    full_crisis_exposure_cap: float = 0.15
    normal_exposure_cap: float = 1.00
    protective_exposure_cap: float = 0.40
    slow_crash_exposure_cap: float = 0.50
    feed_freeze_exposure_cap: float = 0.25
    ood_warning_exposure_cap: float = 0.75
    ood_alien_exposure_cap: float = 0.25
    divergence_hold_seconds: int = 1800
    divergence_major_agent_threshold: int = 2
    divergence_alignment_signals_required: int = 2
    rerisk_step_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    negative_sentiment_z_threshold: float = -2.5
    sentiment_price_mismatch_drop_threshold: float = -0.02
    slow_crash_drawdown_velocity_threshold: float = 0.04
    feed_freeze_staleness_seconds: int = 60
    stress_scenario_library_version: str = "phase4_week3_v1"
    capacity_stress_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    capacity_impact_cap_bps: float = 25.0
    capacity_hard_cap_multiplier: float = 2.0
    stress_max_snapback_ticks: int = 30
    service_name: str = "independent_risk_overseer"
    schema_version: str = "phase4_risk_overseer_v2"
