"""
Day 2 Macro Monitor Demo — generates sample MacroIndicator data and saves to Silver layer.

Run from the project root:
    C:/Users/Anoushka/anaconda3/python.exe scripts/demo_macro_day2.py

This script:
  1. Creates stub records from all 5 clients (all 8 Week 3 indicators)
  2. Saves them via MacroSilverRecorder to data/silver/macro/
  3. Reads back and prints the Parquet files as a table
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src.*' imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone, timedelta

from src.agents.macro.clients.mospi_client import MOSPIClient
from src.agents.macro.clients.rbi_client import RBIClient
from src.agents.macro.clients.nse_fiidii_client import NSEDIIFIIClient
from src.agents.macro.clients.fx_reserves_client import FXReservesClient
from src.agents.macro.clients.bond_spread_client import BondSpreadClient
from src.agents.macro.recorder import MacroSilverRecorder
from src.schemas.macro_data import MacroIndicatorType

import pandas as pd


def main():
    print("=" * 70)
    print("  Macro Monitor — Day 2 Demo: Stub Data Generation")
    print("=" * 70)

    # --- 1. Instantiate all clients ---
    mospi = MOSPIClient()
    rbi = RBIClient()
    nse = NSEDIIFIIClient()
    fx = FXReservesClient()
    bond = BondSpreadClient()

    print("\n✅ All 5 clients instantiated:")
    clients = [
        ("MOSPIClient", mospi),
        ("RBIClient", rbi),
        ("NSEDIIFIIClient", nse),
        ("FXReservesClient", fx),
        ("BondSpreadClient", bond),
    ]
    for name, client in clients:
        indicators = ", ".join(sorted(i.value for i in client.supported_indicators))
        print(f"   {name:22s} → {indicators}")

    # --- 2. Generate stub records for all 8 Week 3 indicators ---
    base_date = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)
    records = []

    # MOSPI: CPI, WPI, IIP (Monthly)
    records.append(mospi._make_stub_record(MacroIndicatorType.CPI, 5.09, base_date))
    records.append(mospi._make_stub_record(MacroIndicatorType.WPI, 2.37, base_date))
    records.append(mospi._make_stub_record(MacroIndicatorType.IIP, 3.80, base_date))

    # NSE: FII_FLOW, DII_FLOW (Daily — simulate 3 days)
    for day_offset in range(3):
        obs_date = base_date + timedelta(days=day_offset)
        records.append(nse._make_stub_record(MacroIndicatorType.FII_FLOW, 1234.56 + day_offset * 100, obs_date))
        records.append(nse._make_stub_record(MacroIndicatorType.DII_FLOW, -890.12 + day_offset * 50, obs_date))

    # FX Reserves (Weekly)
    records.append(fx._make_stub_record(628.5, base_date))

    # RBI Bulletin (event marker, value always = 1.0)
    records.append(rbi._make_stub_record(MacroIndicatorType.RBI_BULLETIN, 99.0, base_date))

    # India-US 10Y Spread (computed)
    records.append(bond._make_stub_record(india_10y_pct=7.15, us_10y_pct=4.25, observation_date=base_date))

    print(f"\n✅ Generated {len(records)} stub MacroIndicator records")
    print(f"   Indicators covered: {sorted(set(r.indicator_name.value for r in records))}")

    # --- 3. Verify provenance tags ---
    print("\n--- Provenance Check (first record) ---")
    sample = records[0]
    print(f"   indicator_name:        {sample.indicator_name.value}")
    print(f"   value:                 {sample.value}")
    print(f"   unit:                  {sample.unit}")
    print(f"   period:                {sample.period}")
    print(f"   source_type:           {sample.source_type.value}")
    print(f"   schema_version:        {sample.schema_version}")
    print(f"   quality_status:        {sample.quality_status.value}")
    print(f"   ingestion_timestamp_utc: {sample.ingestion_timestamp_utc}")
    print(f"   ingestion_timestamp_ist: {sample.ingestion_timestamp_ist}")
    print(f"   region:                {sample.region}")

    # --- 4. Save to Silver layer via MacroSilverRecorder ---
    output_dir = "data/silver/macro"
    recorder = MacroSilverRecorder(base_dir=output_dir)
    recorder.save_indicators(records)
    print(f"\n✅ Saved to Parquet: {output_dir}/")

    # --- 5. Read back and display ---
    print("\n--- Silver Layer Contents ---")
    silver_root = Path(output_dir)
    parquet_files = sorted(silver_root.rglob("*.parquet"))
    print(f"   Parquet files written: {len(parquet_files)}")

    all_dfs = []
    for pf in parquet_files:
        rel = pf.relative_to(silver_root)
        df = pd.read_parquet(pf)
        all_dfs.append(df)
        print(f"   📄 {rel}  ({len(df)} records)")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        display_cols = ["indicator_name", "value", "unit", "period", "timestamp", "source_type", "schema_version", "quality_status"]
        available = [c for c in display_cols if c in combined.columns]
        print(f"\n{'─' * 120}")
        print(combined[available].to_string(index=False))
        print(f"{'─' * 120}")

    # --- 6. RBI Bulletin encoding check ---
    bulletin = [r for r in records if r.indicator_name == MacroIndicatorType.RBI_BULLETIN]
    if bulletin:
        b = bulletin[0]
        status = "✅ PASS" if b.value == 1.0 else "❌ FAIL"
        print(f"\n   RBI Bulletin encoding check: value={b.value}, unit={b.unit}, period={b.period} → {status}")

    # --- 7. Bond Spread computation check ---
    spread = [r for r in records if r.indicator_name == MacroIndicatorType.INDIA_US_10Y_SPREAD]
    if spread:
        s = spread[0]
        expected = (7.15 - 4.25) * 100
        status = "✅ PASS" if abs(s.value - expected) < 0.01 else "❌ FAIL"
        print(f"   Bond Spread check: {s.value} bps (expected {expected} bps) → {status}")

    print(f"\n{'=' * 70}")
    print("  Day 2 demo complete — all stubs, provenance, and recorder working!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
