# SKILL: Green PRs — Merge Conflict Resolution & CI Recovery
**Project:** Multi-Agent AI Trading System — Indian Market  
**Applies To:** All branches, pull requests, and CI pipelines  
**Version:** aligned with Plan v1.3.7

---

## PURPOSE
This skill provides a deterministic, step-by-step workflow for resolving Git merge conflicts and recovering failing CI checks on pull requests. The goal is to get every PR to a green ✅ state before merge.

---

## 1. TRIAGE — Identify the Conflict Source

### 1.1 Fetch & Inspect
```bash
git fetch origin
git status                     # confirm current branch
git merge origin/main          # (or the PR target branch)
```

### 1.2 List Conflicted Files
```bash
git diff --name-only --diff-filter=U
```

### 1.3 Categorise Each Conflict
| Category | Example | Resolution Strategy |
|----------|---------|---------------------|
| **Import order** | `from sqlalchemy import A, B` vs `from sqlalchemy import B, A` | Keep alphabetical; prefer HEAD ordering |
| **Rename / move** | `strategic_recorder.py` → `phase3_recorder.py` | Keep the **newer** name; update all import sites |
| **Schema divergence** | Different fields added to same model | Merge both field sets; verify DB migration compatibility |
| **Config defaults** | Different default values for same field | Prefer the value from the **feature branch** unless main has a bug-fix |
| **Deleted vs modified** | File deleted on one side, modified on other | If deletion was intentional (replaced by new file), delete and update imports |

---

## 2. RESOLUTION — Fix Each File

### 2.1 Resolution Rules (Non-Negotiable)
- **Never blindly accept `--ours` or `--theirs`** on production code; always inspect the diff
- **Contract version** must remain `strat_exec_v1` across all resolved files
- **Audit trail fields** must never be dropped during conflict resolution
- **Risk Overseer rules** are never overridden by merge resolution
- If a file was **renamed** on one branch and **modified** on another, keep the rename and apply the modifications to the new filename

### 2.2 Step-by-Step Per File
1. Open the conflicted file  
2. Search for `<<<<<<<`, `=======`, `>>>>>>>` markers  
3. For each conflict block:
   - Read both sides carefully
   - Determine which change is **additive** vs **destructive**
   - Combine additive changes; discard only truly redundant code
4. Remove all conflict markers  
5. Run the file through the linter / formatter  
6. Stage the file: `git add <file>`

### 2.3 Recorder & Schema Conflicts (Common Pattern)
When `phase3_recorder.py` and `strategic_recorder.py` conflict:
- Prefer `phase3_recorder.py` (canonical name post-Phase 3)
- Ensure `_ensure_dict()` / `_coerce_mapping()` helper exists for Pydantic ↔ dict interop
- Delete the superseded file: `git rm src/db/strategic_recorder.py`
- Update all import sites (`scripts/`, `tests/`, `src/`) to reference the surviving module

---

## 3. VERIFICATION — Prove the Fix

### 3.1 Local Test Gate (Mandatory Before Push)
```bash
# Run the foundation test suite
pytest tests/test_strategic_foundation.py -v

# Run the full Phase 3 week-1 tests if they exist
pytest tests/agents/strategic/ -v

# Run any module-specific tests for conflicted files
pytest tests/ -k "recorder or observation or strategic" -v
```

### 3.2 Import Smoke Check
```bash
python -c "from src.db.phase3_recorder import Phase3Recorder; print('OK')"
python -c "from src.agents.strategic import ObservationAssembler; print('OK')"
python -c "from src.agents.strategic.policies import SACPolicyFoundation; print('OK')"
```

### 3.3 Commit & Push
```bash
git commit -m "fix: resolve merge conflicts with main, keep Phase3Recorder"
git push origin <branch-name>
```

### 3.4 CI Monitoring
After push, monitor the PR checks:
- Wait for CI build to complete (typically ~9 min)
- If CI still fails, read the CI log to identify the **specific failure**
- Common post-merge CI failures:

| CI Failure | Root Cause | Fix |
|------------|-----------|-----|
| `ImportError: cannot import name 'X'` | Missing schema/model after merge | Add the missing class to schemas.py or config.py |
| `AttributeError: object has no attribute 'Y'` | Config dataclass missing a field | Add the field with a sensible default |
| `TypeError: object is not subscriptable` | Pydantic model passed where dict expected | Add `_ensure_dict()` conversion in the recorder |
| `AssertionError` in split tests | WalkForwardConfig date ranges don't cover test data | Update default timestamps to span the test fixture dates |

---

## 4. COMMON CONFLICT PATTERNS IN THIS PROJECT

### 4.1 Recorder Naming
- `phase3_recorder.py` (feature1) vs `strategic_recorder.py` (main/feature3)
- **Resolution:** Keep `phase3_recorder.py`, alias `StrategicRecorder = Phase3Recorder` if backward compat needed

### 4.2 DB Model Imports
- Both branches add new ORM models to `src/db/models.py`
- **Resolution:** Union of all imports; no model should be dropped

### 4.3 Schema Field Additions
- Both branches add fields to `StrategicObservation`, `PolicyFoundationConfig`, etc.
- **Resolution:** Union of all fields; run tests to verify no type conflicts

### 4.4 Query Function Signatures
- `src/db/queries.py` imports may differ in order or content
- **Resolution:** Keep alphabetical import order; union of all query functions

---

## 5. ESCALATION

If after two resolution attempts the CI still fails:
1. Run `pytest --tb=long` to get full tracebacks
2. Check if the failure is in **test infrastructure** vs **production code**
3. If the DB schema has diverged irrecoverably, consider a migration script
4. Never force-push to a shared branch without team agreement

---

## GREEN PR CHECKLIST
- [ ] All `<<<<<<<` / `=======` / `>>>>>>>` markers removed from codebase
- [ ] `contract_version` = `strat_exec_v1` in all resolved files
- [ ] All audit trail fields preserved (never dropped)
- [ ] Import smoke check passes locally
- [ ] `pytest tests/test_strategic_foundation.py` passes (6/6)
- [ ] Changes pushed; CI checks are green ✅
- [ ] No `strategic_recorder.py` / `phase3_recorder.py` dual-existence
