import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.textual.adapters import RawTextRecord, EarningsTranscriptAdapter
from src.agents.textual.cleaners import TextCleaner
from src.schemas.text_data import SourceType
from src.schemas.text_sidecar import SourceRouteDetail

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_tests():
    logger.info("--- Starting Textual Agent Day 4 Feature Tests ---")
    
    cleaner = TextCleaner()
    now = datetime.now()

    # 1. Test Hinglish Detection & Transliteration
    logger.info("\n1. Testing Hinglish & Slang Cleaner...")
    sample_text = "Market aaj gira hai, bhai. Yeh ekdum fake hai, profit book karlo warna loss hoga."
    
    record = RawTextRecord(
        record_type="social_post",
        source_name="x_posts",
        source_id="test_hinglish_1",
        timestamp=now,
        content=sample_text,
        payload={"manipulation_risk_score": 0.1, "language": "en"},
        source_type=SourceType.SOCIAL_MEDIA,
        source_route_detail=SourceRouteDetail.OFFICIAL_FEED
    )
    
    cleaned_record = cleaner.clean(record)
    
    logger.info(f"Original: {sample_text}")
    logger.info(f"Cleaned:  {cleaned_record.content}")
    logger.info(f"Language Tag: {cleaned_record.payload.get('language')}")
    assert "fell" in cleaned_record.content, "Hinglish transliteration failed"
    assert "brother" in cleaned_record.content, "Hinglish transliteration failed"
    assert "book your profits" in cleaned_record.content, "Hinglish transliteration failed"
    assert cleaned_record.payload.get("language") == "code_mixed", "Did not tag as code_mixed"
    
    # 2. Test Slang/Scam Hooks
    logger.info("\n2. Testing Slang/Scam Detection...")
    scam_text = "This is a sure shot multibagger jackpot! Pump and dump incoming 100x guaranteed returns."
    
    scam_record = RawTextRecord(
        record_type="social_post",
        source_name="x_posts",
        source_id="test_scam_1",
        timestamp=now,
        content=scam_text,
        payload={"manipulation_risk_score": 0.2},
        source_type=SourceType.SOCIAL_MEDIA,
        source_route_detail=SourceRouteDetail.OFFICIAL_FEED
    )
    
    scam_cleaned = cleaner.clean(scam_record)
    flags = scam_cleaned.payload.get("quality_flags", [])
    new_risk_score = float(scam_cleaned.payload.get("manipulation_risk_score", 0.0))
    
    logger.info(f"Original: {scam_text}")
    logger.info(f"Flags Added: {flags}")
    logger.info(f"Old Risk Score: 0.2 -> New Risk Score: {new_risk_score:.2f}")
    assert "scam_slang_detected" in flags, "Scam slang was not flagged"
    assert new_risk_score > 0.2, "Manipulation risk score did not escalate"

    # 3. Test PDF Extraction Spot Check
    logger.info("\n3. Testing PDF Extraction in EarningsTranscriptAdapter...")
    try:
        adapter = EarningsTranscriptAdapter()
        records = adapter.fetch()
        if records:
            assert "pdfplumber" in sys.modules, "pdfplumber not imported"
            assert "pdf_extraction" in records[0].payload.get("quality_flags", []), "pdf_extraction flag missing"
            logger.info("PDF Dummy Extract Successful. Preview of text:")
            logger.info(records[0].content[:200] + "...")
        else:
            logger.error("Failed to fetch/extract dummy PDF.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"PDF Extraction Failed with exception: {e}")
        sys.exit(1)
        
    logger.info("\n--- Day 4 Features Verified Successfully ---")

if __name__ == "__main__":
    run_tests()
