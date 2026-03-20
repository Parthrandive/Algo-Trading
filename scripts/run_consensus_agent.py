from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agents.consensus import ConsensusAgent, build_consensus_input_from_phase2_payload


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Consensus Agent on a Phase-2 payload file.")
    parser.add_argument("--input", required=True, help="Path to phase-2 payload JSON")
    parser.add_argument("--output", required=False, help="Optional output path for consensus result JSON")
    args = parser.parse_args()

    payload_path = Path(args.input).resolve()
    payload = _read_json(payload_path)

    consensus_input = build_consensus_input_from_phase2_payload(payload)
    agent = ConsensusAgent.from_default_components()
    result = agent.run(consensus_input)
    result_dict = result.model_dump(mode="json")

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result_dict, handle, indent=2)
        print(f"Consensus output written to {output_path}")
        return

    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
