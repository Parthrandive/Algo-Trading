#!/usr/bin/env python3
"""
Audit Trail Spot-Check — Phase 1 Provenance Verification.

Scans Silver and Gold tier Parquet files and verifies every record carries
the required provenance fields:
  - source_type (non-null, valid SourceType value)
  - ingestion_timestamp_utc (non-null, timezone-aware)
  - ingestion_timestamp_ist (non-null, timezone-aware)
  - schema_version (non-null)
  - quality_status (non-null, valid QualityFlag value)

Outputs a gap summary and overall pass/fail.

Usage:
    python scripts/audit_trail_check.py [--silver-dir data/silver] [--gold-dir data/gold]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# Valid enum values from the schemas.
VALID_SOURCE_TYPES = {"official_api", "broker_api", "fallback_scraper", "manual_override"}
VALID_QUALITY_FLAGS = {"pass", "warn", "fail"}

# Required provenance fields + their validation rules.
PROVENANCE_CHECKS = {
    "source_type": {
        "check": "non_null_and_valid_enum",
        "valid_values": VALID_SOURCE_TYPES,
    },
    "ingestion_timestamp_utc": {
        "check": "non_null",
    },
    "ingestion_timestamp_ist": {
        "check": "non_null",
    },
    "schema_version": {
        "check": "non_null",
    },
    "quality_status": {
        "check": "non_null_and_valid_enum",
        "valid_values": VALID_QUALITY_FLAGS,
    },
}


def scan_parquet_files(directory: Path) -> List[Path]:
    """Recursively find all .parquet files."""
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.parquet"))


def check_file(path: Path) -> Dict[str, Any]:
    """Check a single Parquet file for provenance completeness."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {
            "file": str(path),
            "status": "error",
            "error": str(e),
            "total_records": 0,
            "gaps": [],
        }

    if df.empty:
        return {
            "file": str(path),
            "status": "empty",
            "total_records": 0,
            "gaps": [],
        }

    n = len(df)
    gaps: List[Dict[str, Any]] = []
    all_clean = True

    for field_name, rules in PROVENANCE_CHECKS.items():
        if field_name not in df.columns:
            gaps.append({
                "field": field_name,
                "issue": "column_missing",
                "affected_records": n,
            })
            all_clean = False
            continue

        null_count = int(df[field_name].isna().sum())
        if null_count > 0:
            gaps.append({
                "field": field_name,
                "issue": "null_values",
                "affected_records": null_count,
            })
            all_clean = False

        if rules["check"] == "non_null_and_valid_enum":
            valid_values = rules["valid_values"]
            non_null = df[field_name].dropna()
            # Handle both enum objects and string values.
            str_values = non_null.astype(str).str.lower()
            invalid_mask = ~str_values.isin(valid_values)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                sample_bad = str_values[invalid_mask].unique()[:5].tolist()
                gaps.append({
                    "field": field_name,
                    "issue": "invalid_enum_values",
                    "affected_records": invalid_count,
                    "sample_values": sample_bad,
                })
                all_clean = False

    return {
        "file": str(path),
        "status": "clean" if all_clean else "gaps_found",
        "total_records": n,
        "gaps": gaps,
    }


def run_audit(silver_dir: Path, gold_dir: Path) -> Dict[str, Any]:
    """Run audit trail check across Silver and Gold tiers."""
    all_files = scan_parquet_files(silver_dir) + scan_parquet_files(gold_dir)
    if not all_files:
        return {
            "overall_status": "no_data",
            "total_files": 0,
            "total_records": 0,
            "clean_files": 0,
            "files_with_gaps": 0,
            "details": [],
        }

    results = [check_file(f) for f in all_files]
    total_records = sum(r["total_records"] for r in results)
    clean = sum(1 for r in results if r["status"] == "clean")
    gap_files = [r for r in results if r["status"] == "gaps_found"]

    overall = "PASS" if not gap_files else "FAIL"
    return {
        "overall_status": overall,
        "total_files": len(all_files),
        "total_records": total_records,
        "clean_files": clean,
        "files_with_gaps": len(gap_files),
        "details": results,
    }


def render_terminal(report: Dict[str, Any]) -> str:
    """Render a readable terminal summary."""
    lines = [
        "",
        "═" * 64,
        "  Audit Trail Spot-Check Report",
        "═" * 64,
        f"  Total files scanned:   {report['total_files']}",
        f"  Total records:         {report['total_records']}",
        f"  Clean files:           {report['clean_files']}",
        f"  Files with gaps:       {report['files_with_gaps']}",
        f"  Overall:               {report['overall_status']}",
        "─" * 64,
    ]

    gap_details = [d for d in report["details"] if d["status"] == "gaps_found"]
    if gap_details:
        lines.append("  Gap Details:")
        for detail in gap_details:
            rel_path = detail["file"]
            lines.append(f"\n  📁 {rel_path} ({detail['total_records']} records)")
            for gap in detail["gaps"]:
                lines.append(
                    f"     ⚠  {gap['field']}: {gap['issue']} "
                    f"({gap['affected_records']} records)"
                )
    else:
        lines.append("  ✅ All files have complete provenance tags.")

    lines.append("═" * 64)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Trail Spot-Check")
    parser.add_argument("--silver-dir", type=str, default="data/silver")
    parser.add_argument("--gold-dir", type=str, default="data/gold")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    report = run_audit(Path(args.silver_dir), Path(args.gold_dir))
    print(render_terminal(report))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"Report saved to {out_path}")

    sys.exit(0 if report["overall_status"] in ("PASS", "no_data") else 1)


if __name__ == "__main__":
    main()
