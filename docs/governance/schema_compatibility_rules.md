# Schema Compatibility Rules v1
Date: 2026-02-11
Status: Active

## 1. Versioning Policy
- Use semantic versioning in `schema_version` (`MAJOR.MINOR`).
- Registry keys use `{Entity}_v{MAJOR.MINOR}`.
- Only one active contract per key is allowed.

## 2. Backward-Compatible Changes (MINOR bump)
- Add optional fields with defaults.
- Add new enum members when downstream systems tolerate unknown values.
- Add input aliases for legacy field names.
- Tighten validation only if existing valid payloads continue to pass.

## 3. Breaking Changes (MAJOR bump)
- Rename/remove fields without compatibility alias.
- Change field type/semantics (for example, `str` to `int`).
- Make optional fields required.
- Tighten validation such that previously valid data becomes invalid.

## 4. Deprecation Process
- Mark legacy fields as compatibility aliases for one release cycle minimum.
- Record deprecations in `CHANGELOG.md`.
- Remove aliases only after all publishers migrate and CI checks pass.

## 5. Validation and Promotion Gates
- New schema version must include tests for valid, invalid, and migration payloads.
- Registry update must be atomic (register new key before producer cutover).
- No production write path may emit unregistered schema versions.
