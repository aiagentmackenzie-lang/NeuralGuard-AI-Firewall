# NeuralGuard Production Hardening Plan

> **Living roadmap.** This file tracks what is done and what remains to make
> NeuralGuard production-ready. Update it as work lands. The `routes.py`
> canary comment references this file.
>
> **Last updated:** 2026-06-19
> **Baseline at last update:** `main` @ db7d18d — 484 tests, ruff + mypy
> clean, 90.19% coverage, framework CVEs patched.

---

## What "production-ready" means here

NeuralGuard is a security control in front of LLM APIs. "Production-ready"
means: it can be deployed in front of real traffic with **confidentiality,
integrity, availability, and observability** guarantees, and an operator can
prove it is working. Concretely:

1. **Confidentiality** — no prompt leaves the trust boundary unencrypted or
   unauthenticated; the judge (if enabled) stays local.
2. **Integrity** — the firewall cannot be bypassed by spoofing identity,
   memory-bombing, or ReDoS; verdicts are tamper-evident in audit.
3. **Availability** — a malicious or runaway request cannot exhaust worker
   memory/CPU; rate limits hold across workers.
4. **Observability** — an operator/SOC can see verdicts, latency, failures,
   and bypass attempts in real time and reconstruct any decision.
5. **Operability** — deploy/rollback, secret rotation, backup/restore, and
   incident response are documented and rehearsed.

---

## ✅ Done — current production posture

### Deterministic core (Phase 1)
- 108-pattern scanner (8 EN categories + 50 i18n patterns), ReDoS-protected
  via the `regex` module's per-search `timeout`.
- Structural scanner: NFKD normalization, zero-width stripping, length cap,
  encoding-evasion (base64/hex/ROT13) detection, role-injection patterns.
- Fail-closed arbitration (strictest-verdict-wins; BLOCK on scanner error).

### Semantic + judge (Phase 2)
- ONNX-embedded similarity vs 1,401-vector attack corpus.
- Hybrid scoring (pattern + semantic fusion); judge gated to the ambiguous
  0.30–0.70 composite zone; circuit breaker on judge timeouts.

### Production hardening (Phase 0 — this work)
- **AuthN/AuthZ:** API-key middleware; tenant identity derived from the key
  (not a spoofable header); tenant binding enforced (key↔tenant mismatch → 403).
- **Fail-fast:** production lifespan refuses to start without auth + ≥1 key,
  and warns on plain HTTP unless `allow_insecure_http` is explicitly set.
- **Resource defense:** body-size middleware (413 before JSON parse);
  bounded decompression (incremental `decompressobj` + 8 MiB cap); capped
  base64 decode.
- **Observability:** `metrics.py` + `GET /v1/metrics` (Prometheus: verdicts,
  scanner/pipeline latency, judge calls/timeouts, circuit breaker, audit
  failures, auth/body/rate-limit rejections); JSON logs in production;
  correlated 500s with `correlation_id`.
- **Info hygiene:** `/v1/info` and `/v1/metrics` auth-gated; OWASP coverage
  split into `dedicated_rules` vs `corpus_assisted_only` (no over-claiming).
- **Configurable judge URL** (was hardcoded); loopback default.
- **Bounded audit writes:** in-flight task cap with JSONL overflow fallback.
- **Deploy:** non-root Docker user, container resource limits, Postgres not
  exposed to host, no `changeme` default, `.env.example`.
- **CI:** ruff + mypy (blocking) + coverage gate (90%) + semantic-smoke job +
  SBOM + pip-audit; GitHub Actions pinned to SHAs; dependabot for actions + pip.
- **Testing:** 484 tests incl. hypothesis property tests and negative
  security-control tests (unauth rejected, tenant isolation, spoof-bypass,
  413, memory-bounded bomb, ReDoS-no-hang, production fail-fast).
- **CVE hygiene:** starlette/aiohttp/idna/urllib3 bumped to patched versions.

---

## 🔴 Remaining — to reach full production-readiness

Prioritized. P0 = blocks a real deploy; P1 = should land before scale;
P2 = hardening/enterprise.

### P0 — must do before exposing to real traffic

- **[P0-1] Real boot smoke test.** The full suite passes via ASGI/pytest; we
  have not yet started `uvicorn` against a production config and exercised
  `/v1/evaluate`, `/v1/scan/output`, `/v1/metrics` end-to-end over HTTP with
  auth. Do this once and record the runbook.
- **[P0-2] TLS termination runbook.** Document the nginx/Caddy/Traefik front
  config (or uvicorn `--ssl-keyfile/--ssl-certfile`) and verify it. The
  fail-fast warns but does not enforce TLS.
- **[P0-3] Secret rotation runbook.** Document rotating `NEURALGUARD_AUTH_API_KEYS`
  and `POSTGRES_PASSWORD` without downtime (dual-key window + DB password
  rotation). Currently only "set them in `.env`" is documented.

### P1 — before scale / multi-tenant production

- **[P1-1] Redis-backed rate limiter.** `SlidingWindowCounter` is per-process;
  with `workers > 1` a tenant gets `limit × workers` RPS. The `[redis]` extra
  exists but no Redis limiter is implemented. Implement `RedisRateLimiter`
  behind a config flag (`ratelimit.backend = memory|redis`) and require it
  when `server.workers > 1` (warn/fail otherwise).
- **[P1-2] Per-tenant config + rate limits.** `TenantSettings` exists but is
  not wired. Tenants all share the global RPM/burst. Wire tenant config files
  (`tenants/<id>.yaml`) → per-tenant RPM/burst/scanner overrides.
- **[P1-3] Readiness probe.** `/v1/health` is liveness only. Add
  `/v1/health?ready=1` (or `/v1/ready`) that checks DB connectivity (if
  postgres backend), semantic model load state, and judge reachability — so
  orchestrators don't route traffic to a half-started worker.
- **[P1-4] Audit tamper-evidence.** JSONL/Postgres audit is append-only by
  convention, not by enforcement. For SOC2/EU AI Act evidence, add hash
  chaining (each event's `prev_hash`) or a WORM sink. At minimum, document
  the current integrity guarantee honestly.
- **[P1-5] Load/perf gate.** The P95 latency claims (<10ms pattern, <50ms
  semantic, <5s judge) are unit-test observations, not load-tested. Add a
  k6/locust harness in CI (nightly) that holds P95 under target at N RPS.
- **[P1-6] Backup/restore for the Postgres audit DB.** Not documented. Add a
  pg_dump/restore runbook + a scheduled backup in docker-compose (or document
  the operator's responsibility).

### P2 — hardening & enterprise

- **[P2-1] Canary token verification.** Currently stubbed
  (`canary_leaked=false`). Implement canary injection/detection so output
  scans can detect system-prompt exfiltration. (Tracked as Phase 3.)
- **[P2-2] Phase 3 — Agent Guardian.** Multi-turn detection, prompt-template
  analysis, memory-poisoning detection (ASI06 has corpus only, no dedicated
  rule). Not started.
- **[P2-3] Dedicated ASI04 / ASI10 detection.** Supply Chain and Rogue Agents
  are corpus-assisted only. Add dedicated rules or document the residual risk
  per deployment.
- **[P2-4] Stronger auth: JWT/OAuth2 + key rotation API.** Today = static API
  keys. For enterprise: short-lived JWT/OIDC, a key-management integration
  (Vault/SOPS), and a rotation endpoint.
- **[P2-5] SBOM signing / SLSA / image signing (cosign).** SBOM is generated
  but not attested; image not signed. Add provenance attestation in CI.
- **[P2-6] Kubernetes deployment artifacts.** Only `docker-compose` exists.
  Add a Helm chart / K8s manifests + HPA on the metrics.
- **[P2-7] SIEM/alert routing.** Escalation webhook exists but no alert
  routing to a SIEM (Splunk/ELK/Sentinel) or structured alerting on
  sustained BLOCK spikes.
- **[P2-8] Dead-code cleanup.** The global `@app.exception_handler(Exception)`
  is unreachable through normal flows (routes handle their own exceptions;
  Starlette `BaseHTTPMiddleware` re-raises past it). Either remove it or
  convert the custom middleware to pure ASGI so it genuinely backstops.
- **[P2-9] Coverage headroom.** 90.19% is a thin margin over the 90% gate;
  `main()` uvicorn entry and a few middleware branches are uncovered. Raise
  the effective floor to 92% so one untested branch can't break CI.

---

## Verification checklist (re-run on every change to this file)

```bash
cd "/Users/main/Security Apps/NeuralGuard-AI-Firewall"
source .venv/bin/activate
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
python -m mypy src/neuralguard/
python -m pytest --cov=neuralguard --cov-fail-under=90 -q
uv run pip-audit   # expect only the pip-installer line (non-runtime)
```

Gate must be green before any P0/P1 item is marked done.

---

## Definition of Done for this plan

NeuralGuard is "production-ready" when **all P0 and P1 items above are
closed**, the verification checklist is green on `main`, and a real
`uvicorn` boot smoke test against a production config has been recorded in a
runbook. P2 items are post-production hardening and do not block the
"production-ready" declaration.