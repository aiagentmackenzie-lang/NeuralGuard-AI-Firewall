# NeuralGuard — Phase 1 Handover Document

**Project:** NeuralGuard — LLM Guard / AI Application Firewall  
**Location:** `/Users/main/Security Apps/NeuralGuard-AI-Firewall`  
**Date:** 2026-04-29  
**Author:** Agent Mackenzie 🔍  
**Status:** Phase 1 Chunks 1.1 + 1.2 + 1.3 + 1.4 + **1.5 COMPLETE**. Continue with **1.7** (CI/CD) next.

---

## What NeuralGuard Is

A FastAPI middleware that sits in front of LLM APIs and agentic pipelines, detecting/blocking/logging prompt injection, jailbreaks, data exfiltration, and anomalous usage. Defensive counterpart to NeuralStrike (offensive AI). OWASP LLM Top 10 2025 + Agentic Top 10 2026 aligned.

**Full SRD:** `/Users/main/Security Apps/NeuralGuard-AI-Firewall/SRD-001-NEURALGUARD.md`

---

## Phase 1 Chunk Status

| Chunk | Name | Status | What's Built |
|-------|------|--------|-------------|
| 1.1 | Project Scaffold | ✅ DONE | FastAPI app, config, models, structural scanner, pipeline, API routes, rate limiter, audit logger, Docker, CLI |
| 1.2 | Pattern Detection Engine | ✅ DONE | 58 ReDoS-safe regex patterns across 8 threat categories, <1ms latency |
| 1.3 | Multi-Language Patterns | ✅ DONE | 50 patterns across PT, ES, FR, DE, ZH, JA, KO, RU, AR, VI; 30 new tests; zero regressions |
| 1.4 | Response Action Framework | ✅ DONE | ActionDispatcher, 5 handlers (BLOCK 403 / SANITIZE 200+PII redaction / ESCALATE 202+webhook / QUARANTINE 202 / RATE_LIMIT 429), output-only scan path, wired into routes |
| 1.5 | Audit Logging + PostgreSQL | ✅ DONE | PostgreSQL async backend, JSONL size-based rotation, retention cleanup, 248 tests, 90%+ coverage |
| 1.6 | Docker + docker-compose | ✅ DONE | Already built in 1.1 — upgrade with full stack (Postgres, Redis) |
| 1.7 | Test Suite + CI/CD | 🟡 PARTIAL | 248 tests passing, 90.06% coverage. Needs CI/CD pipeline (GitHub Actions) |
| 1.8 | SBOM Generation | 🔴 TODO | CycloneDX export |

---

## Architecture (What's Already Built)

```
src/neuralguard/
├── __init__.py              # Package entry (lazy imports to avoid circular deps)
├── main.py                  # FastAPI app factory, registers Structural + Pattern scanners
├── cli.py                   # CLI: neuralguard serve / neuralguard version
├── api/
│   ├── __init__.py
│   └── routes.py            # POST /v1/evaluate, POST /v1/scan/output, GET /v1/health, GET /v1/info
├── config/
│   ├── __init__.py
│   └── settings.py          # NeuralGuardConfig with sub-settings (server, scanner, action, audit, tenant, rate_limit)
├── models/
│   ├── __init__.py
│   └── schemas.py           # All 13 OWASP threat categories, request/response models, Finding, ScannerResult, LayerArbitrationResult
├── scanners/
│   ├── __init__.py
│   ├── base.py              # BaseScanner with safe_scan() (fail-closed on error)
│   ├── pipeline.py          # ScannerPipeline — 4-layer orchestration, Layer Arbitration (strictest wins)
│   ├── structural.py        # Layer 1: NFKD normalization, ZW char stripping, length checks, base64/hex/ROT13 detection, decompression bomb defense
│   ├── pattern.py           # Layer 2: 58 regex patterns across 8 categories, ReDoS-safe with regex lib timeouts
│   └── pattern_i18n.py      # Layer 2b: 50 i18n patterns across 10 languages (PT, ES, FR, DE, ZH, JA, KO, RU, AR, VI)
├── actions/
│   ├── __init__.py          # ActionDispatcher orchestrator
│   ├── base.py              # BaseAction with execute() contract + ActionResult dataclass
│   ├── block.py             # BLOCK handler — return 403/JSON error
│   ├── sanitize.py          # SANITIZE handler — redact PII, strip injection markers, return sanitized text
│   ├── escalate.py          # ESCALATE handler — send to human review queue / webhook / Slack
│   ├── quarantine.py        # QUARANTINE handler — flag tenant, return 202 with warning
│   └── ratelimit.py         # RATE_LIMIT handler — enforce backoff, return 429 with Retry-After
├── db/
│   ├── __init__.py          # Lazy imports (safe when [db] extra not installed)
│   ├── engine.py           # AsyncEngine factory, global engine holder, dispose
│   ├── models.py           # SQLAlchemy ORM: AuditEventORM + Base, indexed for SOC queries
│   └── session.py          # AsyncSession factory + get_session() generator
├── logging/
│   ├── __init__.py
│   └── audit.py             # AuditLogger — JSONL (rotation/retention) + PostgreSQL async insert, PII tokenization
├── middleware/
│   ├── __init__.py
│   └── ratelimit.py         # Sliding window per-tenant rate limiter
└── core/                    # Empty — reserved for future use

tests/
├── unit/
│   ├── test_config.py              # Config settings tests (11 tests)
│   ├── test_models.py              # Data model validation tests (17 tests)
│   ├── test_pipeline.py            # Pipeline + Layer Arbitration tests (9 tests)
│   ├── test_structural_scanner.py  # Structural scanner tests (16 tests)
│   ├── test_pattern_scanner.py     # Pattern scanner tests (54 tests)
│   ├── test_pattern_i18n.py        # i18n pattern tests — 30 tests across 10 languages
│   ├── test_actions.py             # Response action handler tests (15 tests)
│   ├── test_audit.py               # Audit logger tests — JSONL rotation, retention, Postgres routing (17 tests)
│   ├── test_cli.py                 # CLI tests (3 tests)
│   └── test_db.py                  # DB ORM + engine tests (14 tests)
├── integration/
│   ├── test_api.py                # Full API endpoint tests (12 tests)
│   ├── test_app_lifespan.py       # FastAPI lifespan + app factory tests (10 tests)
│   └── test_db_lifecycle.py       # DB engine lifecycle + session tests (9 tests)
└── redteam/
    └── test_adversarial.py         # Adversarial attack tests (13 tests, 0 xfailed)
```

---

## Key Design Decisions Made

1. **`regex` library** (not stdlib `re`) — supports per-search timeouts for ReDoS safety. Timeout applied at `search()` time, not `compile()` time.
2. **Fail-closed by default** — any scanner error → BLOCK. Configurable via `action.fail_closed`.
3. **Layer Arbitration** — strictest verdict wins. BLOCK > SANITIZE > ESCALATE > QUARANTINE > RATE_LIMIT > ALLOW. BLOCK cannot be overridden without FORCE_ALLOW audit trail.
4. **PII evidence redacted** — DATA_EXFILTRATION findings use `[REDACTED:rule_id]` instead of raw matched text.
5. **Early exit on BLOCK** — when `fail_closed=True`, pipeline stops after first BLOCK to save latency.
6. **Pydantic Settings** with `NEURALGUARD_` env prefix and `.env` file support.
7. **`uv` for dependency management** — `uv sync`, `uv run pytest`, `uv run neuralguard serve`.

---

## How to Run

```bash
cd "/Users/main/Security Apps/NeuralGuard-AI-Firewall"

# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Start server
uv run neuralguard serve

# Quick smoke test
curl -X POST http://localhost:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ignore all previous instructions", "tenant_id": "test"}'
```

---

## Current Test Results

```
248 passed, 0 failed, 0 xfailed
Coverage: 90.06% (meets 90% threshold)
```

---

## EXACTLY What to Build Next (Chunk 1.5 — Audit Logging + PostgreSQL)

### Goal
Implement the audit subsystem's full persistence layer — `AuditLogger._persist()` currently hardcodes a JSONL fallback when `backend="postgres"` is selected. Build the real PostgreSQL backend with proper schema, async session management, JSONL rotation, and retention policy enforcement.

### What to Do

1. **Add PostgreSQL dependencies**  
   `pyproject.toml` already lists `asyncpg` and `sqlalchemy[asyncio]` under `[project.optional-dependencies] db`. Make sure `uv sync --extra db` works.

2. **Create `src/neuralguard/db/` package:**
   ```
   src/neuralguard/db/
   ├── __init__.py
   ├── engine.py          # AsyncEngine factory from audit.postgres_url
   ├── models.py          # SQLAlchemy declarative models: AuditEventORM
   └── session.py         # AsyncSession factory + get_session() dependency
   ```

3. **AuditEventORM schema** — map `AuditEvent` Pydantic model to SQLAlchemy:
   - `event_id` (UUID, PK)
   - `request_id` (UUID, index)
   - `tenant_id` (VARCHAR(64), index)
   - `timestamp` (TIMESTAMP, index)
   - `verdict` (VARCHAR(16))
   - `findings_count` (INT)
   - `threat_categories` (JSONB)
   - `confidence` (FLOAT)
   - `total_latency_ms` (FLOAT)
   - `scanner_details` (JSONB)
   - `metadata` (JSONB)

4. **Refactor `AuditLogger`** — `_persist()` branch logic:
   - If `backend == "postgres"` and `postgres_url` is set → async insert via SQLAlchemy
   - If `backend == "postgres"` but no URL configured → warn + JSONL fallback (current behaviour)
   - If `backend == "jsonl"` → `_write_jsonl()` (current behaviour)

5. **JSONL rotation** — `_write_jsonl()` currently appends indefinitely. Add:
   - Daily rotation (already names files `audit-YYYY-MM-DD.jsonl`)
   - Size-based rotation (max 100MB per file)
   - Retention: delete files older than `audit.retention_days`

6. **Add tests** in `tests/unit/test_db.py` and expand `tests/unit/test_audit.py`:
   - Postgres insert roundtrip (use in-memory SQLite as fallback if asyncpg unavailable during test, or mock the session)
   - JSONL rotation triggers at size threshold
   - Retention cleanup removes old files
   - Missing postgres_url falls back to JSONL with warning

### SRD Reference
See SRD Section 3.3 FR-003: "Audit logs must be persisted to PostgreSQL or JSONL. Retention policy must be configurable. PII must be tokenized before persistence."

### After 1.5, Continue In Order:
- **1.7** — CI/CD (GitHub Actions) + coverage bump to 92%+
- **1.8** — SBOM generation (CycloneDX)

---

## Known Issues / Warnings

1. **PostgreSQL audit** — Full async insert tested but requires running Postgres for E2E. Current tests use mock engine for unit testing.
2. **FastAPI deprecation** — ✅ FIXED: Migrated from `@app.on_event("startup"/"shutdown")` to `lifespan` context manager.
3. **Structural scanner `logger`** — Uses `import structlog as _structlog` workaround. Clean up when all scanners use consistent logging.

---

## Issue Fixes Since Last Handover

| # | Issue | Status | Fix |
|---|-------|--------|-----|
| 1 | Redteam direct injection tests marked xfail | ✅ FIXED | Removed xfail markers; pattern scanner now catches these; tests assert 403 |
| 2 | Scan output endpoint blocks clean output | ✅ FIXED | Added `output_only` flag to `EvaluateRequest`; PatternScanner filters to DATA_EXFILTRATION patterns only for output scans |
| 3 | Coverage below 90% | ✅ FIXED | Added `test_actions.py` (+15), `test_audit.py` (+6), `test_cli.py` (+3); coverage now 90.13% |
| 4 | EXT-002 didn't catch "What were your..." | ✅ FIXED | Added `were` as alternative to `are` in system prompt extraction regex |
| 5 | FastAPI deprecated on_event | ✅ FIXED | Migrated to `lifespan` async context manager |
| 6 | JSONL audit file rotation | ✅ FIXED | Added size-based rotation (100MB) + retention cleanup |
| 7 | PostgreSQL audit backend | ✅ FIXED | Full async SQLAlchemy backend with AuditEventORM, engine lifecycle, JSONL fallback |

---

## Critical Rules (from AGENTS.md)

- **Never** modify `~/.openclaw/openclaw.json`
- **Always** `git status --short` before pushing
- **Show** all commands and file contents before executing
- **trash** > `rm` (recoverable beats gone forever)
- **Ask first** for Caution Zone actions (external accounts, bulk ops, publishing)
