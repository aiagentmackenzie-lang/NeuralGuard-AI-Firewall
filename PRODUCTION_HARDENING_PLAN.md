# NeuralGuard Production Hardening Plan

**Created:** 2026-06-19 by Agent Mackenzie 🔍 (Lead Security Engineer review)
**Branch:** `production-hardening`
**Baseline:** 434 tests passing, ruff clean, mypy 36 errors (non-blocking in CI), coverage 90.08%

This plan turns the alpha into a production-grade security control. Each phase
ends with a commit and a green test+ruff+mypy gate.

---

## Phase A — Ship blockers (P0)

**Goal:** the service cannot be deployed insecurely by accident.

| ID | Item | Files |
|----|------|-------|
| A-1 | API-key auth middleware + `AuthSettings`; tenant derived from key; rate-limit keyed on auth principal | `config/settings.py`, `middleware/auth.py` (new), `middleware/ratelimit.py`, `main.py` |
| A-2 | Production fail-fast: lifespan refuses to start in `production` without auth enabled + ≥1 key | `main.py` |
| A-3 | Request body-size middleware (config `server.max_request_body_bytes`, default 1 MiB); 413 early | `config/settings.py`, `middleware/bodysize.py` (new), `main.py` |
| A-4 | Bounded decompression bomb defense (`decompressobj` + hard byte cap, not full materialize) | `scanners/structural.py` |
| A-5 | Capped base64 decode (match length cap before `b64decode`) | `scanners/structural.py` |
| A-6 | TLS story: production refuses to serve over plain HTTP unless `server.allow_insecure_http=true` is explicitly set; document reverse-proxy | `config/settings.py`, `main.py`, `cli.py` |
| A-7 | docker-compose: drop Postgres `ports:` mapping, add resource limits, no `changeme` default | `docker-compose.yml` |
| A-8 | CORS: default empty allowlist, `allow_credentials=False`, no wildcard+credentials combo | `config/settings.py`, `main.py` |
| A-9 | `.env.example` committed documenting every var incl. secrets | `.env.example` (new) |
| A-10 | Tests: auth reject/accept, body-size 413, bounded bomb, base64 cap, prod fail-fast | `tests/` |

---

## Phase B — Security-product integrity (P1)

| ID | Item | Files |
|----|------|-------|
| B-1 | Make mypy blocking in CI; fix all 36 type errors | `src/**`, `.github/workflows/ci.yml` |
| B-2 | `/metrics` Prometheus endpoint (verdicts, scanner latencies, judge timeouts, circuit breaker, audit failures) | `api/routes.py`, new `metrics` module |
| B-3 | JSON logs in production (`JSONRenderer`), `ConsoleRenderer` only in dev | `main.py` |
| B-4 | Global exception handler: request_id correlation, sanitized 500 body, audit-on-error | `main.py`, `api/routes.py` |
| B-5 | Gate `/v1/info` behind auth; `/v1/health` returns minimal data unless authenticated | `api/routes.py`, `middleware/auth.py` |
| B-6 | Fix overstated OWASP coverage (ASI04/ASI10 marked "corpus-assisted, no dedicated rule") | `api/routes.py`, `README.md` |
| B-7 | Judge URL configurable (`judge_ollama_url`); loopback/private validation in production | `config/settings.py`, `semantic/judge.py` |
| B-8 | Bound audit DB writes (semaphore + drop/log on overflow); track fire-and-forget tasks | `logging/audit.py` |
| B-9 | Raise coverage floor to 92% (cover lifespan, i18n resolver, error paths) | `tests/`, `pyproject.toml` |
| B-10 | Properly fix H-05: correct return types on evaluate/scan_output | `api/routes.py` |

---

## Phase C — Supply chain & testing (P2)

| ID | Item | Files |
|----|------|-------|
| C-1 | CI matrix adds `--extra semantic` job (skip gracefully if model absent) | `.github/workflows/ci.yml` |
| C-2 | Pin GitHub Actions to SHAs; add `dependabot.yml` | `.github/workflows/ci.yml`, `.github/dependabot.yml` (new) |
| C-3 | Property-based tests (hypothesis): normalization idempotent, ZW stripped, monotonic verdict, ReDoS timeout | `tests/property/` (new), `pyproject.toml` |
| C-4 | Negative security tests: unauth rejected, spoofed tenant ignored, oversized 413, bomb memory-bounded | `tests/security/` (new) |
| C-5 | `git rm --cached sbom.json sbom.xml` (tracked+ignored conflict) | repo root |
| C-6 | Canary detection: remove hardcoded `False`, document as Phase-3 stub explicitly | `api/routes.py`, `README.md` |
| C-7 | README accuracy pass: status, canary, OWASP notes, TLS/auth deploy guide | `README.md` |

---

## Verification gate (run after every phase)

```bash
source .venv/bin/activate
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
python -m mypy src/neuralguard/        # must be clean from Phase B onward
python -m pytest --cov=neuralguard --cov-report=term --cov-fail-under=90
git status --short
```

Commit message convention: `feat(phase-a): ...`, `fix(phase-b): ...`, etc.