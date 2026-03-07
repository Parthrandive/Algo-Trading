import json
from pathlib import Path

from src.agents.textual.textual_data_agent import TextualDataAgent


def test_textual_agent_ingests_pdf_files_from_runtime_config(tmp_path: Path):
    rbi_pdf = tmp_path / "rbi_mpc_mar_2026.pdf"
    earnings_pdf = tmp_path / "infy_q3_2026.pdf"
    rbi_pdf.write_bytes(b"%PDF-1.4 RBI Bulletin Monetary Policy content sample")
    earnings_pdf.write_bytes(b"%PDF-1.4 INFY Earnings Transcript sample content")

    base_config_path = Path("configs/textual_data_agent_runtime_v1.json")
    runtime_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    runtime_config["pdf_input_paths"] = {
        "rbi_reports": [str(rbi_pdf)],
        "earnings_transcripts": [str(earnings_pdf)],
    }

    runtime_config_path = tmp_path / "textual_runtime_with_pdf_inputs.json"
    runtime_config_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")

    agent = TextualDataAgent.from_default_components(runtime_config_path)
    batch = agent.run_once()

    rbi_record = next(r for r in batch.canonical_records if r.source_id == "rbi_report_rbi_mpc_mar_2026")
    earnings_record = next(r for r in batch.canonical_records if r.source_id == "earnings_transcript_infy_q3_2026")

    assert "RBI Bulletin" in rbi_record.content
    assert earnings_record.symbol == "INFY"
    assert earnings_record.quarter == "Q3"
    assert earnings_record.year == 2026

    rbi_sidecar = next(s for s in batch.sidecar_records if s.source_id == "rbi_report_rbi_mpc_mar_2026")
    earnings_sidecar = next(s for s in batch.sidecar_records if s.source_id == "earnings_transcript_infy_q3_2026")

    assert "pdf_extraction_pass" in rbi_sidecar.quality_flags
    assert "pdf_extraction_pass" in earnings_sidecar.quality_flags
