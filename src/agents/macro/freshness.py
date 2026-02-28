import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from src.db.connection import get_engine, get_session
from src.db.models import MacroIndicatorDB
from src.schemas.macro_data import MacroIndicatorType

logger = logging.getLogger(__name__)

class WebhookAlerter:
    """Stub for sending alerts to a webhook (e.g., Slack, PagerDuty)."""
    def send_alert(self, title: str, message: str, level: str = "warning") -> None:
        logger.warning("WEBHOOK ALERT [%s]: %s - %s", level.upper(), title, message)

class MacroFreshnessChecker:
    """
    Evaluates freshness SLAs for macro indicators.
    Identifies stale indicators, missing data, and generates a completeness report.
    """
    def __init__(self, config_path: str, database_url: Optional[str] = None):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.engine = get_engine(database_url)
        self.Session = get_session(self.engine)
        self.alerter = WebhookAlerter()
        
    def _get_latest_records(self) -> Dict[str, datetime]:
        """Fetch the most recent ingestion timestamp for each indicator from the DB."""
        latest_records = {}
        try:
            with self.Session() as session:
                results = (
                    session.query(
                        MacroIndicatorDB.indicator_name,
                        func.max(MacroIndicatorDB.ingestion_timestamp_utc)
                    )
                    .group_by(MacroIndicatorDB.indicator_name)
                    .all()
                )
                for name, dt in results:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    latest_records[name] = dt
        except Exception as e:
            logger.error("Failed to fetch latest records for freshness check: %s", e)
        return latest_records

    def generate_report(self, reference_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a Completeness and Freshness report.
        Evaluates SLA bounds from config.
        """
        now = reference_time or datetime.now(UTC)
        required_indicators = self.config.get("week3_required_publish_set", [])
        indicator_configs = self.config.get("indicator_configs", {})
        
        latest_records = self._get_latest_records()
        
        report = {
            "timestamp": now.isoformat(),
            "total_required": len(required_indicators),
            "healthy": 0,
            "stale": 0,
            "missing": 0,
            "details": {}
        }
        
        for ind_name in required_indicators:
            if ind_name not in indicator_configs:
                logger.warning("Indicator %s is required but missing from config", ind_name)
                continue
                
            sla_hours = indicator_configs[ind_name].get("freshness_window_hours", 24)
            
            if ind_name not in latest_records:
                # Missing completely
                report["missing"] += 1
                report["details"][ind_name] = {
                    "status": "MISSING",
                    "latency_hours": None,
                    "sla_hours": sla_hours
                }
                self.alerter.send_alert(
                    title=f"Missing Data: {ind_name}",
                    message=f"No records found for {ind_name}",
                    level="critical"
                )
                continue
                
            last_ingested = latest_records[ind_name]
            latency_hours = (now - last_ingested).total_seconds() / 3600.0
            
            if latency_hours > sla_hours:
                # Stale
                report["stale"] += 1
                report["details"][ind_name] = {
                    "status": "STALE",
                    "latency_hours": latency_hours,
                    "sla_hours": sla_hours,
                    "last_ingested": last_ingested.isoformat()
                }
                self.alerter.send_alert(
                    title=f"Stale Data: {ind_name}",
                    message=f"{ind_name} is stale. Latency: {latency_hours:.1f}h (SLA: {sla_hours}h)",
                    level="warning"
                )
            else:
                # Healthy/Fresh
                report["healthy"] += 1
                report["details"][ind_name] = {
                    "status": "FRESH",
                    "latency_hours": latency_hours,
                    "sla_hours": sla_hours,
                    "last_ingested": last_ingested.isoformat()
                }

        # Overall completion percentage calculation (healthy / total)
        total = report["total_required"]
        completion_pct = (report["healthy"] / total * 100) if total > 0 else 0
        report["completion_percentage"] = round(completion_pct, 1)
        
        if completion_pct < 95.0:
            self.alerter.send_alert(
                title="SLA Breach: Completeness < 95%",
                message=f"Only {completion_pct}% of indicators are fresh.",
                level="critical"
            )
            
        return report

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    checker = MacroFreshnessChecker("configs/macro_monitor_runtime_v1.json")
    print(json.dumps(checker.generate_report(), indent=2))
