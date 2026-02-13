"""
System Degradation Configuration.
Defines operational states and limits.
"""
from enum import Enum

class SystemState(Enum):
    NORMAL = "NORMAL"
    DEGRADED_REDUCE_ONLY = "DEGRADED_REDUCE_ONLY"
    HALT_CRITICAL = "HALT_CRITICAL"

class ErrorThresholds:
    """Thresholds for error rates triggering degradation."""
    API_ERROR_RATE_WARN = 0.01  # 1%
    API_ERROR_RATE_CRITICAL = 0.05  # 5%

class RecoveryConfig:
    """Configuration for system recovery."""
    AUTO_RECOVERY_WAIT_TIME_S = 300  # 5 minutes stability required
    MANUAL_INTERVENTION_REQUIRED = [SystemState.HALT_CRITICAL]
