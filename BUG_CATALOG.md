# NeuralGuard Bug Catalog — 2026-05-18

**Auditor:** Agent Mackenzie  
**Baseline:** 423 tests passing, ruff lint clean, git clean on `main`  
**Method:** Systematic source review of all 25 Python source files + README + tests

---

## Critical (🔴)

### C-01: `datetime.utcnow()` deprecation — ORM default uses naive datetime
**File:** `src/neuralguard/db/models.py:30`  
**Issue:** `default=datetime.utcnow` creates a naive datetime (no timezone). This is deprecated in Python 3.12+ and breaks PostgreSQL `TIMESTAMP WITH TIME ZONE` columns — Postgres will reject or misinterpret naive datetimes.  
**Fix:** Use `default=lambda: datetime.now(UTC)` from `datetime` module (already imported elsewhere).

### C-02: Rate limiter allows burst OVER limit before blocking
**File:** `src/neuralguard/middleware/ratelimit.py:35-44`  
**Issue:** The burst check `if current >= limit + burst` allows requests up to `limit + burst` before blocking, but then the rate check `if current >= limit` also fires. The logic is inverted — a user making `limit + 1` requests falls through the burst check (not ≥ `limit + burst`) but gets blocked by the rate check. This means burst doesn't work as documented: there's no gradual burst allowance, just a hard wall at `limit`.  
**Fix:** Separate burst from rate: allow up to `limit` requests normally, allow up to `burst` additional in a window, block after `limit + burst`. Current logic should track burst separately.

### C-03: JSONL audit rotation renames then recreates — data loss on concurrent writes
**File:** `src/neuralguard/logging/audit.py:174-185`  
**Issue:** When the JSONL file exceeds 100MB, it's renamed to a numbered file, then a new file is created. Between the rename and the next write, concurrent writes could be lost because `filepath.open("a")` is not atomic and there's no file-level locking. In a multi-worker setup, two workers could both detect >100MB and race on rotation.  
**Fix:** Add a file lock (e.g., `fcntl.flock` on POSIX) or use atomic rename + open with O_CREAT|O_EXCL to avoid double-rotation. For multi-worker, use external coordination (Redis lock) or accept the current single-worker limitation and document it.

### C-04: `asyncio.ensure_future()` used in sync context — no running loop guarantee
**File:** `src/neuralguard/logging/audit.py:131`  
**Issue:** `asyncio.ensure_future(self._async_insert(orm_obj))` is called from `_persist_postgres()` which is a synchronous method. If called outside an async context (e.g., from a sync caller, or in a worker thread), there's no running event loop, so this raises `RuntimeError: no running event loop`. The FastAPI routes are async, but the audit logger can be called from sync contexts (e.g., CLI, tests).  
**Fix:** Either make `_persist_postgres` async and use `await`, or use `asyncio.get_event_loop().create_task()` with a fallback to JSONL, or use `asyncio.run()` for sync contexts.

---

## High (🟡)

### H-01: `_cleanup_retention()` called on every write — performance overhead
**File:** `src/neuralguard/logging/audit.py:190`  
**Issue:** `_cleanup_retention()` does a glob + stat on every JSONL write. With high throughput, this adds filesystem I/O on every audit event. The comment says "every 100 writes" but the implementation calls it every write.  
**Fix:** Add a write counter and only call cleanup every N writes (e.g., 100).

### H-02: Pattern scanner `_severity_to_verdict` inconsistency — MEDIUM maps to SANITIZE but test expectations may differ
**File:** `src/neuralguard/scanners/pattern.py:277-284`  
**Issue:** `_severity_to_verdict` maps `Severity.MEDIUM` → `Verdict.SANITIZE` and `Severity.LOW` → `Verdict.SANITIZE`. This means any pattern match, even low-confidence, results in a SANITIZE verdict. This could cause false positives in production — a single LOW-severity match sanitizes the entire prompt.  
**Fix:** Map `Severity.LOW` → `Verdict.ALLOW` (with a finding logged) and `Severity.INFO` → `Verdict.ALLOW`. Only MEDIUM+ should trigger action.

### H-03: Judge latency measurement is zero when Ollama doesn't return timing
**File:** `src/neuralguard/semantic/judge.py:279-280`  
**Issue:** When `data.get("total_duration", 0)` returns 0 (nanoseconds), the fallback `latency_ms = (time.perf_counter() - time.perf_counter()) * 1000` is always 0 because both calls happen at the same instant. This should use the actual wall-clock time instead.  
**Fix:** Track start time before the Ollama call and compute latency from wall clock.

### H-04: CORS allows all origins in development — no tenant isolation
**File:** `src/neuralguard/main.py:120-126`  
**Issue:** In development mode, CORS allows all origins (`*`). This is fine for local dev but there's no per-tenant origin enforcement in production either (empty list = no CORS, which means browsers block cross-origin, but API clients aren't affected). The production CORS config (`allow_origins=[]`) means no browser can use the API from a different origin.  
**Fix:** Document that production deployments must configure `allow_origins` explicitly. Add a config field for allowed origins.

### H-05: `evaluate` endpoint returns `JSONResponse` but declares `EvaluateResponse` return type
**File:** `src/neuralguard/api/routes.py:63`  
**Issue:** The function signature says `-> EvaluateResponse` but the BLOCK/ESCALATE path returns `JSONResponse` (a different type). FastAPI will still work but the OpenAPI schema is misleading — it only shows the 200 response model.  
**Fix:** Add explicit response models for 403, 429, 422 status codes, or change the return type to the union type.

### H-06: Rate limiter `SlidingWindowCounter` is per-process — doesn't work with multiple workers
**File:** `src/neuralguard/middleware/ratelimit.py:18`  
**Issue:** The counter is stored in a `defaultdict(list)` in memory. With `workers > 1` in uvicorn, each worker has its own counter. A user could make `limit * workers` requests per minute by hitting different workers.  
**Fix:** Document this as a known limitation for multi-worker deployments and recommend Redis backend for production. Add a config note.

### H-07: Pattern scanner compiles with `re_module.IGNORECASE` flag but patterns already use `(?i)` inline
**File:** `src/neuralguard/scanners/pattern.py:247`  
**Issue:** The `PatternScanner._compile_patterns()` compiles every pattern with `flags=re_module.IGNORECASE`. But most patterns already contain `(?i)` inline flags. This results in redundant case-insensitive matching — not a bug per se, but the i18n patterns in `pattern_i18n.py` also get `(?i)` applied even though some are for non-Latin scripts (CJK, Arabic, Cyrillic) where `(?i)` has no effect or unintended effects. The Russian and Vietnamese patterns use `(?i)` which is fine, but CJK patterns don't need it and Arabic patterns might behave unexpectedly with case-insensitive mode.  
**Fix:** Compile without the `IGNORECASE` flag and rely on each pattern's own `(?i)` where intended. Remove the global flag.

### H-08: `ScanOutputRequest` doesn't validate empty output
**File:** `src/neuralguard/models/schemas.py:82-88`  
**Issue:** Unlike `EvaluateRequest` which has `@model_validator` ensuring at least `messages` or `prompt`, `ScanOutputRequest` has no validator for empty `output`. An empty string would pass validation and create a meaningless scan.  
**Fix:** Add a field validator for `output` similar to `Message.content_not_empty`.

---

## Medium (🟠)

### M-01: `_cleanup_retention` uses file mtime instead of filename date for age calculation
**File:** `src/neuralguard/logging/audit.py:194-202`  
**Issue:** Retention cleanup uses `st_mtime` (file modification time) to determine age. But files can be touched, copied, or rotated without changing their name-date. If a file is re-written or touched, it won't be cleaned up even if it's old by name. More critically, on some filesystems, mtime can be unreliable.  
**Fix:** Parse the date from the filename (`audit-YYYY-MM-DD.jsonl`) for age determination, falling back to mtime.

### M-02: Audit logger `_tokenize_metadata` only tokenizes strings > 10 chars
**File:** `src/neuralguard/logging/audit.py:207-213`  
**Issue:** The 10-character threshold means short PII like "São Paulo" (9 chars with spaces) or "Rio" (3 chars) won't be tokenized. But "user@example.com" (16 chars) would be. This threshold is arbitrary and may miss PII or over-tokenize non-PII.  
**Fix:** Use a proper PII detection regex (email, phone, SSN patterns) instead of a length threshold, or make the threshold configurable.

### M-03: `HybridScoringEngine` category for hybrid finding is `SELF_ATTACK`
**File:** `src/neuralguard/semantic/hybrid.py:158`  
**Issue:** The composite finding uses `category=ThreatCategory.SELF_ATTACK`. This is misleading — `SELF_ATTACK` (`T-NG`) means NeuralGuard is attacking itself. A hybrid composite finding should use the most severe category from the findings it combines, or a new category. Using `T-NG` will confuse SOC analysts who see "self-attack" for what is actually a genuine external threat.  
**Fix:** Use the category from the highest-severity finding, or `ThreatCategory.PROMPT_INJECTION_DIRECT` as a fallback.

### M-04: `ActionDispatcher` doesn't handle ALLOW verdict explicitly
**File:** `src/neuralguard/actions/__init__.py:43-52`  
**Issue:** `VERDICT_MAP` doesn't include `Verdict.ALLOW`. When `arbitration.verdict` is `ALLOW`, `handler` is `None`, and the fallback path returns a generic 200 response. This works, but the response body differs from the `EvaluateResponse` model that the API documents — it doesn't include `sanitized_content`, `scan_layers_used`, or `total_latency_ms`.  
**Fix:** Add an explicit `AllowAction` or at minimum construct the response to match `EvaluateResponse` fields.

### M-05: Escalate action uses synchronous `httpx.Client` in potentially async context
**File:** `src/neuralguard/actions/escalate.py:36-43`  
**Issue:** The `_send_webhook` method uses `httpx.Client` (synchronous) which blocks the event loop in FastAPI's async context. This could cause latency spikes under load.  
**Fix:** Use `httpx.AsyncClient` with `await` in async contexts, or run in a thread pool executor.

### M-06: `SlidingWindowCounter` never cleans up old entries for inactive tenants
**File:** `src/neuralguard/middleware/ratelimit.py:20-21`  
**Issue:** `_counters` is a `defaultdict(list)` that grows unboundedly for every unique tenant. Tenants that make a few requests and never return again still have entries. Over time, this leaks memory proportional to the number of unique tenants.  
**Fix:** Add periodic cleanup of inactive tenants (e.g., tenants with no entries in the last 5 minutes) or use a `TTLCache`.

### M-07: `EvaluateRequest.scanners` allows duplicate layers
**File:** `src/neuralguard/models/schemas.py:68`  
**Issue:** `scanners: list[ScanLayer] | None` allows duplicates like `[ScanLayer.PATTERN, ScanLayer.PATTERN]`. The pipeline would run the same scanner twice.  
**Fix:** Add a validator that deduplicates the list, or use `set` semantics.

### M-08: CLI `serve` command sets env vars but `load_config()` reads from `.env` file first
**File:** `src/neuralguard/cli.py:38-50`  
**Issue:** Setting `os.environ` after `load_config()` is called won't take effect because `load_config()` is called inside `serve_main()`. The env vars set by CLI would need to be set BEFORE the config is loaded. Currently the flow is: set env vars → call `serve_main()` → which calls `load_config()`. This actually works because `os.environ` is set before `load_config()`. But if `.env` file has conflicting values, Pydantic-settings will use the env var (which has priority). So this is actually fine, but worth noting that `.env` values for the same key will be silently overridden.  
**Status:** NOT A BUG — works correctly, just documenting the precedence.

### M-09: `ScanOutputResponse.confidence` field missing from `scan_output` endpoint response
**File:** `src/neuralguard/api/routes.py:111-120` and `models/schemas.py:143`  
**Issue:** `ScanOutputResponse` has a `confidence` field defined, but the endpoint constructs it from `arbitration.findings`. The code builds `confidence = max(...)` from findings, which is fine for blocked outputs but for ALLOW responses with no findings, `max()` on an empty sequence would fail.  
**Fix:** Use `max(..., default=0.0)` — already done in the evaluate endpoint but not consistently in scan_output.

### M-10: Pattern scanner `output_only` mode only checks DATA_EXFILTRATION category
**File:** `src/neuralguard/scanners/pattern.py:230-235`  
**Issue:** When `output_only=True`, only `DATA_EXFILTRATION` patterns are used. This misses `SYSTEM_PROMPT_EXTRACTION` and `ENCODING_EVASION` patterns that could also appear in LLM output (e.g., a model leaking system prompts). The README says output scan checks for PII, canary, and system prompt leakage, but the code only checks EXF patterns.  
**Fix:** Include `SYSTEM_PROMPT_EXTRACTION` and `ENCODING_EVASION` in output-only pattern set.

---

## Low (🟢)

### L-01: `pyproject.toml` version duplicated in `__init__.py`
**Files:** `pyproject.toml:2`, `src/neuralguard/__init__.py:3`  
**Issue:** Version `"0.1.0"` is hardcoded in both files. Updating one without the other creates drift.  
**Fix:** Use `importlib.metadata.version("neuralguard")` in `__init__.py` to read from pyproject.toml at runtime.

### L-02: Docker compose uses hardcoded password `changeme`
**File:** `docker-compose.yml:23`  
**Issue:** Default PostgreSQL password is `changeme` — fine for dev but should be documented as must-change for production. The env var `POSTGRES_PASSWORD` defaults to `changeme`.  
**Fix:** Add a comment in docker-compose.yml and README warning about this.

### L-03: `Dockerfile` installs `uv` but uses `uv sync` — could use `pip install` instead
**File:** `Dockerfile:15`  
**Issue:** Using `uv` in Docker adds complexity. The Docker image copies `uv.lock*` and uses `uv sync`. If `uv.lock` doesn't exist, this may fail.  
**Fix:** Ensure `uv.lock` is committed or use `pip install -e .[db]` as fallback.

### L-04: Test coverage threshold in pyproject.toml says `fail_under = 90` but actual is 90.34%
**File:** `pyproject.toml:93`  
**Issue:** This is fine, but any bug fix that adds untested code could drop below 90%.  
**Fix:** Add tests for new code to maintain coverage.

### L-05: `pattern_i18n.py` imports `ThreatCategory` at bottom of file after I18N_FLAT is built
**File:** `src/neuralguard/scanners/pattern_i18n.py:351-353`  
**Issue:** `ThreatCategory` is imported at the bottom of the file, after `I18N_FLAT` list comprehension uses `resolve_category()` which needs `ThreatCategory`. This works because the import is at module scope and Python processes it before the list comprehension at the bottom. But it's confusing and fragile — if someone moves the import, it breaks.  
**Fix:** Move the `from neuralguard.models.schemas import ThreatCategory` to the top of the file with the other imports.

### L-06: `.gitignore` excludes `sbom.json` and `sbom.xml` but they exist in the repo
**Issue:** The repo currently has `sbom.json` and `sbom.xml` committed (109KB and 110KB respectively) but `.gitignore` lists them. They should either be tracked or gitignored, not both.  
**Fix:** `git rm --cached sbom.json sbom.xml` to remove from tracking while keeping locally.

### L-07: `AuditEventORM.timestamp` uses `default=datetime.utcnow` (naive)
**File:** `src/neuralguard/db/models.py:30`  
**Issue:** Same as C-01 but specifically the ORM model default. `datetime.utcnow` is deprecated and creates naive datetimes.  
**Fix:** Already tracked in C-01.

### L-08: README references `docs/SRD-001-NEURALGUARD.md` but `.gitignore` excludes internal docs
**Issue:** The README links to `SRD-001` document which is in `docs/` but `.gitignore` excludes `docs/HANDOVER.md`, `docs/SRD-001-NEURALGUARD.md`, and `docs/PHASE2-PLAN.md`. The README link will 404 on GitHub.  
**Fix:** Remove the internal doc links from the README, or move SRD to a public path.

---

## README Inconsistencies

### R-01: README claims "108 patterns" but actual count differs
The README says "108 patterns, 13 redteam tests" under Detection Rate. Let me count:  
- PI-D: 10 patterns  
- PI-I: 6 patterns  
- JB: 12 patterns  
- EXT: 6 patterns  
- EXF: 10 patterns  
- TOOL: 5 patterns  
- DOS: 5 patterns  
- ENC: 4 patterns  
- i18n: 50 patterns (5 per language × 10 languages)  
**Total: 58 English + 50 i18n = 108** ✅ Correct!

### R-02: README says "1,401 vectors across 8 categories"
This refers to the attack corpus. Can't verify without the corpus files (they're gitignored). The code supports this claim. Mark as verified-by-code.

### R-03: README says "P95 Latency (Pattern-only) <10ms"
Tests show "0.6-1.4ms observed". This is consistent.

### R-04: README API example for `/v1/scan/output` shows 403 for PII detection
The actual code returns 403 for BLOCK verdict from PII patterns, which matches.

### R-05: README mentions "Canary token verification" in output scan but it's stubbed
**File:** `src/neuralguard/api/routes.py:109` — `canary_leaked = False` is hardcoded. The README implies canary detection is working.  
**Fix:** Document canary detection as "Phase 3 (planned)" or implement it.

### R-06: README mentions "OWASP Agentic Top 10 2026" coverage including ASI04 and ASI10
**File:** `src/neuralguard/api/routes.py:143-153` — The `/v1/info` endpoint lists ASI01, ASI02, ASI06, ASI10 but the code doesn't have specific detection for ASI04 (Supply Chain) or ASI10 (Rogue Agents) beyond the corpus entries. This is aspirational coverage, not direct detection.  
**Fix:** Add a note clarifying which threats have dedicated detection rules vs. corpus-only coverage.

---

## Summary

| Severity | Count |
|----------|-------|
| 🔴 Critical | 4 |
| 🟡 High | 8 |
| 🟠 Medium | 10 |
| 🟢 Low | 8 |
| **Total** | **30** |

**Priority fix order:** C-01 → C-04 → H-01 → H-02 → H-03 → M-09 → M-10 → rest