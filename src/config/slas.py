"""
System Service Level Agreements (SLAs) Configuration.
Defines constants for latency, freshness, and quality thresholds.
"""

class LatencySLA:
    """Latency thresholds in milliseconds."""
    TICK_TO_TRADE_P95_MS = 50
    TICK_TO_TRADE_MAX_MS = 100
    ORDER_EXECUTION_P95_MS = 200
    ORDER_EXECUTION_MAX_MS = 500
    INTERNAL_MSG_P95_MS = 5
    INTERNAL_MSG_MAX_MS = 10

class FreshnessSLA:
    """Data freshness thresholds in seconds."""
    MARKET_DATA_MAX_AGE_NORMAL_S = 1.0
    MARKET_DATA_MAX_AGE_DEGRADED_S = 3.0
    MARKET_DATA_HALT_THRESHOLD_S = 5.0
    
    POSITION_DATA_MAX_AGE_S = 5.0
    POSITION_DATA_BLOCK_THRESHOLD_S = 10.0

class QualitySLA:
    """Data quality thresholds."""
    MAX_MISSING_DATA_PCT = 0.0  # Strict 0% for critical data
    SIGMA_OUTLIER_THRESHOLD = 3.0

class SystemSLA:
    """System availability targets."""
    CORE_UPTIME_TARGET = 0.999
    INGESTION_UPTIME_TARGET = 0.995
