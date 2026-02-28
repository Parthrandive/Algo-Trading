import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import TypeAdapter
from src.schemas.macro_data import MacroIndicator, MacroIndicatorType, SourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path("data/macro_samples")

def generate_sample(indicator: MacroIndicatorType, val: float, unit: str, period: str, days_ago: int = 1) -> MacroIndicator:
    now = datetime.now(UTC)
    obs_date = (now - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    return MacroIndicator(
        indicator_name=indicator,
        value=val,
        unit=unit,
        period=period,
        timestamp=obs_date,
        source_type=SourceType.OFFICIAL_API,
        ingestion_timestamp_utc=now,
        ingestion_timestamp_ist=now,
        schema_version="1.1",
        quality_status="pass",
    )

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Week 3 Required Set
    samples = {
        MacroIndicatorType.CPI: [(5.1, "%", "Monthly"), (5.0, "%", "Monthly")],
        MacroIndicatorType.WPI: [(2.4, "%", "Monthly"), (2.3, "%", "Monthly")],
        MacroIndicatorType.IIP: [(3.8, "%", "Monthly"), (3.9, "%", "Monthly")],
        MacroIndicatorType.FII_FLOW: [(-1500.5, "INR_Cr", "Daily"), (500.0, "INR_Cr", "Daily")],
        MacroIndicatorType.DII_FLOW: [(2000.0, "INR_Cr", "Daily"), (1000.0, "INR_Cr", "Daily")],
        MacroIndicatorType.FX_RESERVES: [(680.5, "USD_Bn", "Weekly"), (682.0, "USD_Bn", "Weekly")],
        MacroIndicatorType.RBI_BULLETIN: [(1.0, "count", "Irregular")],
        MacroIndicatorType.INDIA_US_10Y_SPREAD: [(285.0, "bps", "Daily"), (290.0, "bps", "Daily")],
    }
    
    adapter = TypeAdapter(list[MacroIndicator])
    
    for indicator, values in samples.items():
        records = []
        for i, (val, unit, period) in enumerate(values):
            # Stagger timestamps slightly for multiple samples
            records.append(generate_sample(indicator, val, unit, period, days_ago=(i+1)*5))
            
        # Serialize to JSON-compatible dicts
        payload = [r.model_dump(mode="json") for r in records]
        
        # Validate through TypeAdapter
        try:
            adapter.validate_python(payload)
            logger.info("Successfully validated payload for %s", indicator.value)
        except Exception as e:
            logger.error("Failed validation for %s: %s", indicator.value, e)
            continue
            
        # Save to file
        file_path = OUTPUT_DIR / f"{indicator.value.lower()}_sample.json"
        with open(file_path, "w") as f:
            json.dump(payload, f, indent=2)
            
        logger.info("Saved %d records to %s", len(records), file_path)

if __name__ == "__main__":
    main()
