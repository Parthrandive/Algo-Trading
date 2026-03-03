from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.agents.textual.textual_data_agent import TextualDataAgent

def verify_day2():
    print("--- Starting Day 2 Verification ---")
    
    # Initialize Agent
    agent = TextualDataAgent.from_default_components()
    
    # Run ingestion
    batch = agent.run_once(as_of_utc=datetime.now(UTC))
    
    # Analyze results
    canonical_count = len(batch.canonical_records)
    sidecar_count = len(batch.sidecar_records)
    
    print(f"Total Canonical Records: {canonical_count}")
    print(f"Total Sidecar Records: {sidecar_count}")
    
    # Check for compliance rejection
    rejected_records = [s for s in batch.sidecar_records if s.compliance_status.value == "reject"]
    print(f"Rejected Records: {len(rejected_records)}")
    
    for r in rejected_records:
        print(f"  Rejected: {r.source_id} from {r.source_type} (Reason: {r.compliance_reason})")
        
    # Check for provenance and quality flags
    for s in batch.sidecar_records:
        if s.compliance_status.value == "allow":
            print(f"  Allowed: {s.source_id} | Route: {s.source_route_detail.value} | Flags: {s.quality_flags}")

    # Export to dict for summary
    exporter = agent.exporter
    summary = exporter.as_dict(batch)
    
    summary_path = Path("logs/day2_verification_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Verification summary saved to {summary_path}")
    print("--- Day 2 Verification Complete ---")

if __name__ == "__main__":
    verify_day2()
