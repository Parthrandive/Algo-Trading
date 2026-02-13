# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `requirements.lock` for pinned reproducible dependency resolution from the active `venv`.
- `architecture/architecture_decisions_v1.md` as the locked Day 2 architecture decision record.
- `architecture/module_boundaries.md` with explicit module I/O boundaries and failure behavior.
- `governance/data_contracts_v1.md` documenting frozen Day 3 contracts and mandatory provenance fields.
- `governance/schema_compatibility_rules.md` defining backward/forward compatibility and version bump rules.

### Changed
- Canonical provenance fields in schemas now use `source_type`, `ingestion_timestamp_utc`, `ingestion_timestamp_ist`, and `quality_status` with backward-compatible input aliases.
- Added/updated schema tests to validate provenance aliases, UTC+IST timestamps, and stricter contract behavior.

## [0.1.0] - 2026-02-09
### Added
- Initial project structure (`src/`, `tests/`, `configs/`, `scripts/`, `governance/`).
- `requirements.txt` with Day 1 dependencies (`pandas`, `numpy`, `nsepython`, `pydantic`, etc.).
- `governance/source_inventory.md` defining data sources and integrity ranks.
- `CHANGELOG.md` for document control.
