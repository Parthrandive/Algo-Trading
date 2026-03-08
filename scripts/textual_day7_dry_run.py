import sys
import json
import logging
from pathlib import Path
from datetime import datetime, UTC

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.textual.textual_data_agent import TextualDataAgent
from src.schemas.text_sidecar import ComplianceStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Starting Day 7 Dry Run for Textual Data Agent")
    agent = TextualDataAgent.from_default_components()
    
    # Run a full batch
    start_time = datetime.now(UTC)
    logging.info(f"Running agent for {start_time}")
    batch = agent.run_once(as_of_utc=start_time)
    
    logging.info("Processing complete. Summarizing results...")
    
    # Analyze the batch
    canonical_records = batch.canonical_records
    sidecar_records = batch.sidecar_records
    
    total_canonical = len(canonical_records)
    total_sidecars = len(sidecar_records)
    
    # Analyze sidecars
    compliance_rejects = 0
    stale_ttl_breaches = 0
    fallback_sources = 0
    validation_failures = 0
    source_stats = {}
    quality_status_stats = {}
    
    for s in sidecar_records:
        src = s.source_type.value
        source_stats[src] = source_stats.get(src, 0) + 1
        
        if s.compliance_status == ComplianceStatus.REJECT:
            compliance_rejects += 1
            logging.info(f"Rejected: {s.source_id} - Reason: {s.compliance_reason}")
            
        if hasattr(s, "ttl_seconds") and s.ttl_seconds < 0:
            stale_ttl_breaches += 1
            
        if s.source_route_detail is not None and "fallback" in s.source_route_detail.value.lower():
            fallback_sources += 1
            
    for c in canonical_records:
        q = c.quality_status.value
        quality_status_stats[q] = quality_status_stats.get(q, 0) + 1
        
    validation_failures = total_sidecars - total_canonical
    
    report = {
        "timestamp": start_time.isoformat(),
        "total_canonical_records_produced": total_canonical,
        "total_sidecar_records_produced": total_sidecars,
        "source_distribution": source_stats,
        "quality_status_distribution": quality_status_stats,
        "metrics": {
            "compliance_rejects": compliance_rejects,
            "canonical_validation_drops": validation_failures,
            "stale_ttl_breaches": stale_ttl_breaches,
            "fallback_source_rates": fallback_sources
        }
    }
    
    report_path = Path(__file__).resolve().parents[1] / "logs" / "textual_day7_dry_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logging.info(f"Dry run complete. Report written to {report_path}")
    logging.info(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
