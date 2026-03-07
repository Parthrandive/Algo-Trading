import argparse
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.preprocessing.pipeline import PreprocessingPipeline
from src.schemas.preprocessing_data import TransformOutput

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

def orchestrate_replay(cutoff_date: str) -> dict:
    # A single shared snapshot ID across all streams for the day
    # Representing the state of data as it stood precisely at cutoff_date
    replay_snapshot_id = f"cross_agent_replay_{uuid.uuid4().hex[:8]}"

    # In a full cross-agent environment, this invokes:
    # 1. Sentinel Replay (Bronze -> Silver OHLCV + Corp)
    # 2. Macro Replay (Bronze -> Silver Macro)
    # 3. Preprocessing (Silver -> Gold Deterministic Tensors)
    # 4. Textual Partner Replay (Bronze Text -> Canonical Silver -> Metadata Sidecar)
    
    # We orchestrate Preprocessing here as the ultimate sink of numeric streams
    pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
    
    market_path = str(PROJECT_ROOT / "data" / "silver" / "ohlcv")
    macro_path = str(PROJECT_ROOT / "data" / "silver" / "macro")
    
    print(f"Running Numeric Replay up to Cutoff: {cutoff_date}")
    output: TransformOutput = pipeline.replay_snapshot(
        market_source_path=market_path,
        macro_source_path=macro_path,
        snapshot_id=replay_snapshot_id,
        cutoff_date=cutoff_date,
        corporate_action_path=None 
    )

    return {
        "dataset_snapshot_id": replay_snapshot_id,
        "cutoff_date_event_time": cutoff_date,
        "numeric_gold_hash": output.output_hash,
        "status": "READY_FOR_TEXTUAL_SYNC"
    }

def main():
    parser = argparse.ArgumentParser(description="Cross Agent Replay Harness for Day 4 Sync.")
    parser.add_argument("--cutoff", default=datetime.now(timezone.utc).isoformat(), help="Cutoff date (ISO 8601)")
    
    args = parser.parse_args()
    
    print("--- 1. Orchestrating Multi-Stream Replay ---")
    try:
        report = orchestrate_replay(args.cutoff)
        print("\n--- 2. Snapshot Coordination Output ---")
        print(json.dumps(report, indent=2))
        print(f"\nSUCCESS: Snapshot {report['dataset_snapshot_id']} generated.")
        print("Please share the snapshot ID and cutoff with the textual partner for Sync S2.")
    except Exception as e:
        print(f"Harness failed: {e}")
        sys.exit(EXIT_FAILURE)

    sys.exit(EXIT_SUCCESS)

if __name__ == "__main__":
    main()
