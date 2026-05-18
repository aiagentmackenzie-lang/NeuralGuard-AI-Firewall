# NeuralGuard Code Quality Audit — Session Handoff

> **PASS THIS FILE AS THE FIRST MESSAGE TO THE NEXT AGENT SESSION.**
> The prompt below gives full context. After reading it, continue from "Remaining Work" section.

---

## 🎯 COPY-PASTE PROMPT FOR NEXT SESSION

```
You are continuing a code quality audit on NeuralGuard (LLM Guard / AI Application Firewall).

Project: /Users/main/Security Apps/NeuralGuard-AI-Firewall
Repo: https://github.com/aiagentmackenzie-lang/NeuralGuard-AI-Firewall
Branch: main (clean, all changes uncommitted — see git status)
Venv: source .venv/bin/activate (Python 3.11.15)
Tests: 423 passing, ruff lint clean (as of session start)
Bug catalog: BUG_CATALOG.md in project root

WHAT WAS DONE:
- Read all 25 source files systematically
- Found 30 bugs (4 Critical, 8 High, 10 Medium, 8 Low)
- Fixed all 4 Critical bugs (C-01 through C-04) — edits made but NOT yet tested or committed

CRITICAL FIXES ALREADY APPLIED (uncommitted):
1. C-01: db/models.py — datetime.utcnow → datetime.now(UTC) (naive → timezone-aware)
2. C-02: middleware/ratelimit.py — burst logic rewritten (was double-blocking at limit, now allows limit+burst)
3. C-03: logging/audit.py — JSONL cleanup every 100 writes instead of every write + write counter added
4. C-04: logging/audit.py — asyncio.ensure_future → loop.create_task() with RuntimeError fallback to JSONL

REMAINING WORK (in order):
1. Run tests to verify Critical fixes don't break anything: `cd "/Users/main/Security Apps/NeuralGuard-AI-Firewall" && source .venv/bin/activate && python -m pytest --tb=short -q`
2. Fix High bugs (H-01 through H-08) — see BUG_CATALOG.md for details
3. Fix Medium bugs (M-01 through M-10) — see BUG_CATALOG.md for details
4. Fix Low bugs (L-01 through L-08) — see BUG_CATALOG.md for details
5. Add tests for each bug fix where practical
6. Update README.md for accuracy issues (R-05 canary stub, R-06 OWASP coverage note, R-08 internal doc 404)
7. Run final test suite + ruff lint
8. Git commit all changes with clear messages

KEY FILES:
- Bug catalog: BUG_CATALOG.md (full details on every bug)
- Source: src/neuralguard/ (25 Python files)
- Tests: tests/ (unit/, integration/, redteam/)
- Config: pyproject.toml
- Entry: src/neuralguard/main.py
- Pipeline: src/neuralguard/scanners/pipeline.py

IMPORTANT RULES:
- Show commands before running, show outputs
- Never rm -rf system directories
- Commit after each bug batch (Critical done, then High, then Medium, then Low)
- Keep BUG_CATALOG.md updated with fix status as you go
- Run `python -m ruff check src/ tests/` after changes
- Run full test suite after each batch
```

---

## Audit Progress

### Baseline (Start of Session)
- **Tests:** 423 passing, 0 failing
- **Lint:** ruff clean
- **Git:** Clean on `main` branch
- **Coverage:** 90.34%

### Bugs Found: 30 Total

| Severity | Count | Fixed | Remaining |
|----------|-------|-------|-----------|
| 🔴 Critical | 4 | 4 | 0 |
| 🟡 High | 8 | 0 | 8 |
| 🟠 Medium | 10 | 0 | 10 |
| 🟢 Low | 8 | 0 | 8 |

### Critical Fixes Applied (Uncommitted)

| ID | File | Fix |
|----|------|-----|
| C-01 | `db/models.py` | `datetime.utcnow` → `datetime.now(UTC)` with lambda |
| C-02 | `middleware/ratelimit.py` | Burst logic: removed double-block at `limit`, now allows up to `limit + burst` |
| C-03 | `logging/audit.py` | `_cleanup_retention()` called every 100 writes instead of every write |
| C-04 | `logging/audit.py` | `asyncio.ensure_future()` → `loop.create_task()` with `RuntimeError` fallback to JSONL |

### High Bugs (Next Batch)

| ID | File | Summary |
|----|------|---------|
| H-01 | `logging/audit.py` | `_cleanup_retention()` now fixed (every 100 writes) ✅ |
| H-02 | `scanners/pattern.py` | `_severity_to_verdict` maps LOW→SANITIZE, should be LOW→ALLOW |
| H-03 | `semantic/judge.py` | Latency = 0 when Ollama doesn't return timing (uses `time.perf_counter() - time.perf_counter()`) |
| H-04 | `main.py` | CORS `allow_origins=[]` in production blocks all browser clients |
| H-05 | `api/routes.py` | `evaluate()` returns `JSONResponse` but signature says `EvaluateResponse` — misleading OpenAPI |
| H-06 | `middleware/ratelimit.py` | SlidingWindowCounter is per-process, doesn't work with multi-worker |
| H-07 | `scanners/pattern.py` | Global `(?i)` flag redundant with inline `(?i)` in patterns |
| H-08 | `models/schemas.py` | `ScanOutputRequest` doesn't validate empty `output` |

### Medium Bugs

| ID | File | Summary |
|----|------|---------|
| M-01 | `logging/audit.py` | Retention uses file mtime instead of filename date |
| M-02 | `logging/audit.py` | PII tokenization threshold (10 chars) misses short PII |
| M-03 | `semantic/hybrid.py` | Hybrid composite finding uses `SELF_ATTACK` category — misleading |
| M-04 | `actions/__init__.py` | `ActionDispatcher` has no ALLOW handler — response body differs from schema |
| M-05 | `actions/escalate.py` | Uses sync `httpx.Client` in async context (blocks event loop) |
| M-06 | `middleware/ratelimit.py` | `_counters` dict grows unbounded for inactive tenants |
| M-07 | `models/schemas.py` | `EvaluateRequest.scanners` allows duplicate layers |
| M-08 | `cli.py` | Env var precedence vs .env — works correctly, just documented |
| M-09 | `api/routes.py` | `scan_output` endpoint `max()` on empty findings would crash |
| M-10 | `scanners/pattern.py` | `output_only` mode only checks EXF, misses EXT and ENC |

### Low Bugs

| ID | File | Summary |
|----|------|---------|
| L-01 | `__init__.py` + `pyproject.toml` | Version `0.1.0` duplicated — use `importlib.metadata` |
| L-02 | `docker-compose.yml` | Hardcoded `changeme` password — add warning |
| L-03 | `Dockerfile` | `uv sync` may fail without `uv.lock` — add fallback |
| L-04 | `pyproject.toml` | `fail_under = 90` — need tests for new code |
| L-05 | `pattern_i18n.py` | `ThreatCategory` import at bottom — move to top |
| L-06 | `.gitignore` | `sbom.json` / `sbom.xml` tracked but gitignored |
| L-07 | `db/models.py` | Same as C-01 (naive datetime) |
| L-08 | `README.md` | Links to gitignored internal docs (404 on GitHub) |

### README Inconsistencies

| ID | Issue |
|----|-------|
| R-05 | Canary detection is hardcoded `False` — README implies it works |
| R-06 | OWASP ASI04/ASI10 listed in `/v1/info` but no dedicated detection rules |
| R-08 | `SRD-001` link 404s on GitHub (gitignored) |

---

## Commit Plan

```
1. fix(critical): C-01 through C-04 — datetime, ratelimit, audit, asyncio
2. fix(high): H-02 through H-08 — verdict mapping, judge latency, CORS, OpenAPI, ratelimit docs, pattern flags, validation
3. fix(medium): M-01 through M-10 — audit retention, PII, hybrid category, AllowAction, async httpx, tenant cleanup, duplicate scanners, confidence default, output-only scope
4. fix(low): L-01 through L-08 — version, docker, sbom, imports
5. docs: README updates (canary, OWASP, doc links)
6. test: new tests for bug fixes
```

---

## Commands Reference

```bash
# Activate venv
cd "/Users/main/Security Apps/NeuralGuard-AI-Firewall" && source .venv/bin/activate

# Run tests
python -m pytest --tb=short -q

# Run lint
python -m ruff check src/ tests/

# Run coverage
python -m pytest --cov=neuralguard --cov-report=term-missing --cov-fail-under=90

# Git commit
git add -A && git commit -m "fix(critical): ..."

# Check git status
git status --short
```