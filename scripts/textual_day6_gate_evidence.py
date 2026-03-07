from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"
PLAN_DIR = REPO_ROOT / "docs" / "plans"

JUNIT_REPORT_PATH = LOG_DIR / "textual_day6_pytest_junit.xml"
PYTEST_STDOUT_PATH = LOG_DIR / "textual_day6_pytest_output.txt"
SUMMARY_PATH = LOG_DIR / "textual_day6_gate_evidence_summary.json"
DEFECT_LOG_PATH = LOG_DIR / "textual_day6_defect_log.json"
REMEDIATION_PATH = LOG_DIR / "textual_day6_remediation_list.md"
MARKDOWN_REPORT_PATH = PLAN_DIR / "week-4-textual-data-agent-day6-gate-evidence.md"
SIDECAR_ARTIFACT_PATH = LOG_DIR / "textual_sidecar_records.json"
PDF_REPORT_PATH = LOG_DIR / "textual_pdf_spot_check_report.json"
COMPLIANCE_LOG_PATH = LOG_DIR / "compliance_rejects.log"

DAY6_TEST_TARGETS = [
    "tests/test_textual_day1.py",
    "tests/test_textual_day2.py",
    "tests/test_textual_day3.py",
    "tests/test_textual_day4.py",
    "tests/test_textual_day5.py",
    "tests/test_textual_day6.py",
    "tests/test_textual_adapters.py",
]


@dataclass(frozen=True)
class PytestSummary:
    tests: int
    failures: int
    errors: int
    skipped: int
    passed: int
    duration_seconds: float
    failing_tests: list[dict[str, str]]


def _ensure_output_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_DIR.mkdir(parents=True, exist_ok=True)


def _run_pytest(junit_path: Path) -> int:
    pytest_temp_dir = LOG_DIR / "pytest_tmp"
    pytest_temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *DAY6_TEST_TARGETS,
        "-q",
        "--disable-warnings",
        "-p",
        "no:cacheprovider",
        f"--basetemp={pytest_temp_dir}",
        f"--junitxml={junit_path}",
    ]
    process_env = dict(os.environ)
    process_env["TMP"] = str(pytest_temp_dir)
    process_env["TEMP"] = str(pytest_temp_dir)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=process_env,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    PYTEST_STDOUT_PATH.write_text(output, encoding="utf-8")
    return int(completed.returncode)


def _parse_junit_report(junit_path: Path) -> PytestSummary | None:
    if not junit_path.exists():
        return None

    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        return PytestSummary(0, 0, 0, 0, 0, 0.0, [])

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_duration = 0.0
    failing_tests: list[dict[str, str]] = []

    for suite in suites:
        total_tests += int(suite.attrib.get("tests", "0"))
        total_failures += int(suite.attrib.get("failures", "0"))
        total_errors += int(suite.attrib.get("errors", "0"))
        total_skipped += int(suite.attrib.get("skipped", "0"))
        total_duration += float(suite.attrib.get("time", "0") or 0.0)

        for case in suite.iter("testcase"):
            classname = case.attrib.get("classname", "").strip()
            name = case.attrib.get("name", "").strip()
            test_id = "::".join(part for part in (classname, name) if part)
            for child in case:
                if child.tag not in {"failure", "error"}:
                    continue
                message = (child.attrib.get("message") or (child.text or "")).strip()
                failing_tests.append(
                    {
                        "test_id": test_id or "unknown_test",
                        "kind": child.tag,
                        "message": message[:400],
                    }
                )

    passed = max(total_tests - total_failures - total_errors - total_skipped, 0)
    return PytestSummary(
        tests=total_tests,
        failures=total_failures,
        errors=total_errors,
        skipped=total_skipped,
        passed=passed,
        duration_seconds=round(total_duration, 3),
        failing_tests=failing_tests,
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _provenance_summary(sidecar_artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not sidecar_artifact:
        return {
            "records_seen": 0,
            "provenance_complete_records": 0,
            "provenance_coverage_ratio": 0.0,
            "fallback_route_count": 0,
            "allow_count": 0,
            "reject_count": 0,
        }

    records = sidecar_artifact.get("records", [])
    if not isinstance(records, list):
        records = []

    complete = 0
    fallback_route_count = 0
    allow_count = 0
    reject_count = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        if all(item.get(key) for key in ("source_id", "source_type", "source_route_detail", "ingestion_timestamp_utc")):
            complete += 1
        if item.get("source_route_detail") == "fallback_scraper":
            fallback_route_count += 1
        if item.get("compliance_status") == "allow":
            allow_count += 1
        if item.get("compliance_status") == "reject":
            reject_count += 1

    total_records = len(records)
    coverage_ratio = round((complete / total_records), 4) if total_records else 0.0
    return {
        "records_seen": total_records,
        "provenance_complete_records": complete,
        "provenance_coverage_ratio": coverage_ratio,
        "fallback_route_count": fallback_route_count,
        "allow_count": allow_count,
        "reject_count": reject_count,
    }


def _pdf_summary(pdf_report: dict[str, Any] | None) -> dict[str, Any]:
    if not pdf_report:
        return {
            "total_documents": 0,
            "warn_count": 0,
            "fail_count": 0,
            "average_quality_score": None,
        }
    return {
        "total_documents": int(pdf_report.get("total_documents", 0)),
        "warn_count": int(pdf_report.get("warn_count", 0)),
        "fail_count": int(pdf_report.get("fail_count", 0)),
        "average_quality_score": pdf_report.get("average_quality_score"),
    }


def _compliance_summary() -> dict[str, int]:
    if not COMPLIANCE_LOG_PATH.exists():
        return {"reject_log_line_count": 0}
    line_count = len([line for line in COMPLIANCE_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()])
    return {"reject_log_line_count": line_count}


def _build_defects(
    *,
    pytest_summary: PytestSummary | None,
    pytest_return_code: int | None,
    provenance: dict[str, Any],
    pdf: dict[str, Any],
) -> list[dict[str, str]]:
    defects: list[dict[str, str]] = []

    if pytest_summary is None:
        defects.append(
            {
                "id": "D6-001",
                "severity": "high",
                "source": "pytest",
                "status": "open",
                "description": "JUnit report missing; unable to verify Day 6 test execution evidence.",
            }
        )
    else:
        for idx, failure in enumerate(pytest_summary.failing_tests, start=1):
            defects.append(
                {
                    "id": f"D6-1{idx:02d}",
                    "severity": "high",
                    "source": "pytest",
                    "status": "open",
                    "description": f"{failure['kind']}: {failure['test_id']} | {failure['message']}",
                }
            )
        if pytest_return_code not in (None, 0) and not pytest_summary.failing_tests:
            defects.append(
                {
                    "id": "D6-190",
                    "severity": "high",
                    "source": "pytest",
                    "status": "open",
                    "description": "Pytest returned non-zero but no failures were parsed from JUnit.",
                }
            )

    if provenance.get("records_seen", 0) > 0 and provenance.get("provenance_coverage_ratio", 0.0) < 1.0:
        defects.append(
            {
                "id": "D6-201",
                "severity": "medium",
                "source": "provenance",
                "status": "open",
                "description": (
                    "Provenance completeness below 100% "
                    f"({provenance.get('provenance_coverage_ratio')})."
                ),
            }
        )

    if int(pdf.get("fail_count", 0)) > 0:
        defects.append(
            {
                "id": "D6-301",
                "severity": "medium",
                "source": "pdf_quality",
                "status": "open",
                "description": f"PDF extraction fail_count={pdf.get('fail_count')} exceeds gate threshold.",
            }
        )

    return defects


def _remediation_action(defect: dict[str, str]) -> str:
    source = defect.get("source")
    if source == "pytest":
        return "Fix failing test or production logic, then rerun Day 6 test targets and refresh gate evidence."
    if source == "provenance":
        return "Backfill missing sidecar provenance fields and add regression checks for source tags."
    if source == "pdf_quality":
        return "Inspect failed PDF samples, improve extractor quality, and retest spot-check thresholds."
    return "Investigate defect root cause and assign an owner with a dated closure plan."


def _write_markdown_report(
    *,
    summary: dict[str, Any],
    defects: list[dict[str, str]],
) -> None:
    pytest_section = summary["pytest"]
    provenance = summary["provenance"]
    pdf = summary["pdf_quality"]

    lines = [
        "## Week 4 Textual Data Agent Day 6 Gate Evidence (March 7, 2026)",
        "",
        "### Gate Summary",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Test targets: `{len(summary['test_targets'])}` files",
        f"- Pytest totals: `{pytest_section['tests']}` tests, `{pytest_section['passed']}` passed, "
        f"`{pytest_section['failures']}` failures, `{pytest_section['errors']}` errors, "
        f"`{pytest_section['skipped']}` skipped",
        f"- Provenance coverage: `{provenance['provenance_complete_records']}` / "
        f"`{provenance['records_seen']}` (`{provenance['provenance_coverage_ratio']}`)",
        f"- Fallback route records observed: `{provenance['fallback_route_count']}`",
        f"- PDF quality: `{pdf['total_documents']}` documents, "
        f"`warn={pdf['warn_count']}`, `fail={pdf['fail_count']}`",
        f"- Compliance reject log lines: `{summary['compliance']['reject_log_line_count']}`",
        "",
        "### Defect Log",
    ]

    if not defects:
        lines.append("- No unresolved defects.")
    else:
        for defect in defects:
            lines.append(
                f"- `{defect['id']}` [{defect['severity']}] ({defect['source']}): "
                f"{defect['description']}"
            )

    lines.extend(["", "### Remediation List"])
    if not defects:
        lines.append("- No remediation actions required for current gate run.")
    else:
        for defect in defects:
            lines.append(f"- `{defect['id']}`: {_remediation_action(defect)}")

    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Day 6 textual gate evidence artifacts.")
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Do not execute pytest; parse existing JUnit report if present.",
    )
    parser.add_argument(
        "--junit-path",
        default=str(JUNIT_REPORT_PATH),
        help="Path to pytest JUnit XML report.",
    )
    args = parser.parse_args()

    _ensure_output_dirs()
    junit_path = Path(args.junit_path)
    pytest_return_code: int | None = None

    if not args.skip_pytest:
        try:
            pytest_return_code = _run_pytest(junit_path)
        except Exception as exc:  # noqa: BLE001
            PYTEST_STDOUT_PATH.write_text(f"pytest execution failed: {exc}\n", encoding="utf-8")
            pytest_return_code = 2

    pytest_summary = _parse_junit_report(junit_path)
    sidecar_artifact = _load_json(SIDECAR_ARTIFACT_PATH)
    pdf_report = _load_json(PDF_REPORT_PATH)

    provenance = _provenance_summary(sidecar_artifact)
    pdf = _pdf_summary(pdf_report)
    compliance = _compliance_summary()

    pytest_section = {
        "tests": pytest_summary.tests if pytest_summary else 0,
        "passed": pytest_summary.passed if pytest_summary else 0,
        "failures": pytest_summary.failures if pytest_summary else 0,
        "errors": pytest_summary.errors if pytest_summary else 0,
        "skipped": pytest_summary.skipped if pytest_summary else 0,
        "duration_seconds": pytest_summary.duration_seconds if pytest_summary else 0.0,
        "return_code": pytest_return_code,
    }
    defects = _build_defects(
        pytest_summary=pytest_summary,
        pytest_return_code=pytest_return_code,
        provenance=provenance,
        pdf=pdf,
    )

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "gate_name": "week4_textual_day6",
        "test_targets": DAY6_TEST_TARGETS,
        "pytest": pytest_section,
        "provenance": provenance,
        "pdf_quality": pdf,
        "compliance": compliance,
        "unresolved_defects_count": len(defects),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    DEFECT_LOG_PATH.write_text(json.dumps(defects, indent=2), encoding="utf-8")

    remediation_lines = ["# Day 6 Remediation List", ""]
    if not defects:
        remediation_lines.append("- No remediation actions required.")
    else:
        for defect in defects:
            remediation_lines.append(f"- `{defect['id']}`: {_remediation_action(defect)}")
    REMEDIATION_PATH.write_text("\n".join(remediation_lines) + "\n", encoding="utf-8")

    _write_markdown_report(summary=summary, defects=defects)

    print(f"Summary: {SUMMARY_PATH}")
    print(f"Defect log: {DEFECT_LOG_PATH}")
    print(f"Remediation list: {REMEDIATION_PATH}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
