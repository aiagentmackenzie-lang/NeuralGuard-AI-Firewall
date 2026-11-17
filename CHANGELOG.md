# Changelog

All notable changes to NeuralGuard. Version numbers follow the tag on `main`;
every entry below is merged, CI-gated and documented in the README or a runbook.
Things that are *not* shipped are listed under "Known limits" rather than
implied by silence.

The project had no changelog before this file, and the previous tag (`v0.2.1`)
predates the research-frontier controls, the MCP gateway and the fleet campaign
— so `0.3.0` carries everything since, not just the work of the last few weeks.

## [Unreleased] — since v0.3.0 (2026-11-17)

### Added — multi-replica HITL (NG-15 shared store)

The approval broker was per worker and in memory, which is honest for one
replica and wrong for two: an escalation raised on worker A is invisible to an
approver whose request lands on worker B, so the human cannot decide, the wait
times out, and the call is refused — a control that only works on one worker.

- New `neuralguard.mcp.hitl_redis.RedisHitlStore`, selected with
  `NEURALGUARD_MCP_HITL_STORE=redis` (+ `NEURALGUARD_MCP_HITL_REDIS_URL`). The
  store keeps every NG-15 property, and three of them had to be implemented
  rather than inherited from a dict and a lock: **single use** is a Lua
  compare-and-set that deletes the record (read-then-write would let two workers
  both consume one approval); **decisions are single-shot** (a denial can never be
  overwritten into an approval); **expiry is Redis TTL**, so an expired escalation
  cannot be listed, approved late or consumed. Capacity is trimmed inside the
  same script, so two workers cannot both squeeze past the cap.
- **Fail closed on store failure:** every Redis error surfaces as
  `HitlStoreError`; `wait` returns `None` (refusal) on timeout, expiry or store
  failure. No path returns an approval from an error.
- **A per-process store is refused in multi-worker production**, same rule and
  same place as the rate-limiter refusal: `production` + `workers > 1` +
  `hitl_store=memory` refuses to start, because the control would silently
  degrade to a timeout. Single-worker production keeps the in-memory store.
- `pending_for_approval()` and `get()` are now `async` on both stores, so the
  approver API does not care which one a deployment runs. 19 contracts
  (`tests/unit/test_hitl_redis.py`) run on `fakeredis[lua]`, so the atomicity
  tests exercise real Lua — including the race: two workers consuming one
  approval produce exactly one success.

**Measured, two real processes + real Redis (2026-11-17):** raised on process
:8001, the escalation was visible from :8002, approved there (`200`), and the
held request on :8001 then completed **`200`** with the upstream receiving the
actual `tools/call`. The record was gone afterwards (consumed) and re-approving
the same id returned `404`. Negative control with `hitl_store=memory`: the other
process saw **0** escalations and the held request timed out
(`403 mcp_escalation_timeout`) — the exact failure a two-replica deployment would
have shipped. Bonus, by design: with no baselined catalog the approved call was
still refused by NG-7 (`MCP-RUGPULL-002`), which is "approval is not authority"
demonstrated live.

### Fixed — two deployment-path defects found by re-drilling the published image

Neither was visible in any file in the repo; both needed a clean cluster and the
artifact a stranger pulls. Full evidence: `deploy/kubernetes/README.md`
(2026-11-17 drill).

- **A late database disabled the durable audit path for the pod's lifetime.**
  The Postgres schema was created once in the lifespan, in a `try/except` that
  logged and gave up. When the app pods started before the `postgres` Service had
  endpoints (`[Errno -2] Name or service not known`), the engine was never usable
  for schema creation, so every later audit INSERT failed with
  `relation "audit_events" does not exist` while the service kept accepting
  traffic; only a `rollout restart` recovered it. Initialisation now retries with
  backoff (`neuralguard.db.init.initialise_postgres`, 1/2/4/8/8 s, budget set by
  the new `NEURALGUARD_AUDIT_INIT_ATTEMPTS`, default 6) and returns a result
  instead of swallowing the failure. Contract: `tests/unit/test_audit_db_init.py`.
- **`/v1/ready` called that broken audit store `ok`.** The probe was `SELECT 1`
  (connectivity), so it answered `audit_db: "ok"` and aggregate `"healthy"` with
  no `audit_events` table. It now checks the schema with
  `to_regclass('audit_events')` and reports `fail` when a configured backend
  cannot persist; the worker keeps serving (JSONL still records, the documented
  fallback) but the aggregate status drops to `degraded`.

### Fixed — the scheduled Bench gate can no longer be killed by its own runtime

The 2026-11-17 scheduled run went red for the first time in six: the A2 step ran
**39 m 16 s** and was killed by `timeout-minutes: 40`. The step's cost is
`5 goals x (jb_iterations - 1)` = 10 local 7B generations with **no per-call
timeout and no overall budget** (NeuralStrike's `LLMManager.call_local` passes
none to the Ollama client), so its runtime was a function of the runner's CPU.
A job *timeout* also cancels the run rather than failing the job, which is why
the job's `continue-on-error: true` never applied.

- New `benchmarks/ng_vs_ns/budget.py`: a per-call timeout (one hung generation
  costs one attack, not the benchmark) and an overall generation budget
  (deterministic truncation), with results carrying their originating item so a
  dropped call cannot shift later payloads onto the wrong goal.
- The payload records `generation` (planned, produced, elapsed, budget,
  timeouts, `complete`), `_format_results` prints an **INCOMPLETE** banner, and
  the CLI exits **2** on a truncated set — `continue-on-error` keeps the
  scheduled run green while the job tile still says "look at this".
- The workflow passes `--generation-budget-seconds 540` with
  `--per-call-timeout-seconds 180` and `timeout-minutes: 20`; a contract asserts
  budget + 300 s slack fits inside the job timeout, so this configuration cannot
  be recreated by accident.
- Local runs are unchanged: no budget by default, so the benchmark is not
  truncated by design.

**Measured, on `main` after the fix** (`workflow_dispatch` run `44166136037`,
head `0b180f9`): the A2 step ran **8 m 33 s** (was 39 m 16 s), job success, and
the artifact reports `generation: {planned: 10, produced: 10, elapsed_seconds:
501.051, budget_seconds: 540, timed_out_calls: 0, complete: true}` with
`source_commit` and the corpus digest stamped — so the provenance added the day
before is now exercised by CI rather than only locally. The budget did not
truncate; on a runner more than ~8 % slower it will, and then the artifact says
so and the job exits 2 instead of cancelling the run.

### Changed — evidence hygiene

- **Benchmark provenance is now stamped by the harness.** `a2_results.json`
  carried models, iterations and configs but no timestamp and no corpus digest,
  so the tracked artifact could not prove which corpus built it. `live_harness.py`
  now writes `generated_at` (UTC), `source_commit` and a `corpus` block (path,
  SHA-256 and record count, hashed from the corpus bytes at run time). The
  historical file was back-stamped honestly: `generated_at`/`source_commit` are
  `null` with a `provenance_unrecorded` sentence, and the corpus digests are
  recorded because both corpora are unchanged since `676c10b` (2026-06-27).
- **`.gitignore` no longer contradicts the tracked evidence.**
  `benchmarks/ng_vs_ns/results/` was ignored wholesale while four files in it were
  tracked, including the README-cited `a2_results.json`. The rule is now an
  explicit ignore/enumerate split with `!` negations, and a test fails if any
  tracked record is ignored (`git check-ignore --no-index`).
- 1,499 tests / 12 skipped locally, coverage 91.18 % (floor 90), ruff + format +
  mypy strict clean.

### Known limits — unchanged by this entry

- The **published v0.3.0 image does not contain these fixes**; they land in the
  tree and are proven in-cluster against a locally built image. A v0.3.1 publish
  is an owner decision.
- Multi-replica runtime key rotation, prometheus-adapter metrics and ingress/TLS
  remain undrilled; SSE streaming is still refused `422`.
- The shared HITL store closes the *broker* half of the multi-replica gap. A
  two-replica live drill (escalation raised on pod A, approved via the Service
  landing on pod B) is the evidence that would close it, and it has not been run
  yet at the time of writing.
- `NEURALGUARD_SERVER_WORKERS > 1` is the only multi-worker signal the app can
  see: **a multi-pod deployment is invisible to it**, so the manifests/runbook
  carry the instruction to set `hitl_store=redis` and the app cannot enforce it.

## [0.3.0] — 2026-11-16

### Added — the published, signed image

- **`ghcr.io/aiagentmackenzie-lang/neuralguard` is now a real, pullable, signed
  artifact.** Until this release the Kubernetes manifest pinned
  `ghcr.io/aiagentmackenzie-lang/neuralguard:0.2.1`, a package that never
  existed, and no workflow built one; the kind drill had validated the manifests
  by loading a locally built image, which is not the deployment path.
- New `Image` workflow: `linux/amd64` + `linux/arm64`, keyless cosign signature
  on the **digest**, a cosign-verifiable CycloneDX SBOM attestation, and buildx
  `provenance: mode=max` + SPDX SBOM attached to the image index. The job
  verifies its own signature and SBOM attestation identity-scoped, and checks
  the index-embedded attestations structurally — the referrers API does not
  list them, so `cosign verify-attestation` cannot see them, and the first
  publish attempt went red for claiming otherwise.
- The v0.3.0 digest is pinned in the Kubernetes manifest:
  `ghcr.io/aiagentmackenzie-lang/neuralguard@sha256:8984cfdf…`. The 2026-11-12
  cluster drill was run against a locally built image, not this digest.
- A pre-publish `smoke` job boots the built image with `--network none`, health
  checks it from inside, and fails if the log shows runtime package resolution.

### Added — MCP gateway authorization arc (NG-7 → NG-15)

The gateway no longer only says *no*. It decides:

- **Whether** — NG-8 pre-body-parse per-tool Intent Gate on `Mcp-Method` /
  `Mcp-Name`, plus smuggling defence; NG-7 Ed25519-signed tool-catalog baselines
  that refuse a rug pull (strict mode withholds a drifted catalog).
- **How** — NG-10: eight kinds of argument and resource constraint, evaluated on
  the parsed arguments before forwarding.
- **Who** — NG-11: workload identity taken from the credential
  (`key|tenant|agent`), with narrowing-only per-agent policy; the route passes
  `request.state.auth_agent`, never a header.
- **With which upstream credential** — NG-12 per-tool and NG-14 per-`(agent,
  tool)` credential mapping. Keys normalise at config load; resolution is
  most-specific-first; and once any `agent|tool` entry exists for a tool, a
  caller without a matching entry is refused rather than inheriting the broad
  default.
- **Whether a human agrees** — NG-15: `ESCALATE` now has an action behind it. A
  bounded, TTL'd, **single-use** approval broker; the approval is bound to a
  SHA-256 of the canonical JSON-RPC body; decisions require a tenant-level
  credential and the tenant must be a named approver (an agent cannot approve
  its own escalation); and **approval is not authority** — an approved call
  still faces the argument gate, the catalog baseliner, the provenance gate and
  the credential resolver.
- **Provenance** — NG-9 egress taint binding (off / warn / block).

### Added — detection

- **NG-13 context-aware cross-script homoglyph fold.** The 16-operator mutation
  gate measured the Cyrillic/Greek/accent class at **96.30 % Unsafe-ASR**.
  Measured after: **0.00 %** for all three operators, aggregate **56.94 % →
  38.89 %**, Safe-ASR unchanged at 12.50 %, zero benign BLOCKs. Native Russian
  and Greek text is byte-identical (tested).
- NG-5 16-operator mutation gate that reports Unsafe-ASR **and** Safe-ASR, so
  robustness cannot be bought with false positives.
- NG-6 published guarded-FPR SLO — 50 hard negatives at boot and per-PR,
  **0.00 % across 95 probes**, with a per-tenant sensitivity dial.
- NG-1 reasoning/intermediate-token output scan, NG-3 decode-then-activate
  detection, NG-4 overflow-resistant windowed aggregation.

### Added — audit and operations

- **Cross-worker global ordering for the Postgres audit backend**: DB sequence
  plus a global hash link, advisory-lock serialised; `audit-verify --pg-url`
  reports `(global-order)`. JSONL keeps its per-worker chains.
- Kubernetes manifests **cluster-drilled on kind** (2026-11-12): apply, 2/2
  Ready, dependency-aware probes, HPA **2 → 4 → 2**, signed Postgres audit rows.
  Seven drill-found defects fixed — including probes being rate-limited, which
  marked healthy pods `NotReady`.
- Fleet deployment: NeuralGuard + SecurityScarletAI + NeuralStrike in one compose
  project, the ScarletAI producer contract (companion events, opt-in batching),
  and a live-fire receipt from 2026-09-20.

### Fixed

- **Postgres audit insert never worked.** Caught by live-fire, not by the suite:
  every "audit persisted" claim was untested against a real database until the
  coverage job gained a `postgres:17` service.
- **A production false positive in the Portuguese patterns** — bare `você é`
  blocked ordinary Portuguese sentences.
- **The semantic-corpus claim**, and paraphrase passes silently lost during
  augmentation; the corpus now rebuilds reproducibly from tracked sources
  (6,503 vectors) and a docs-drift test fails the build when a README figure
  diverges from the code.
- **HITL edge cases found in a post-merge audit of my own work**: approver
  tenants could list and decide *other* tenants' escalations, and the
  HITL-disabled path had silently renamed the shipped audit outcomes.
- The perf gate measured a shared-runner p95 and gated the SLO on it.

### Changed

- **The container entrypoint no longer uses `uv run`.** `uv run` is a
  resolver-aware launcher: the image is synced `--no-dev`, uv's default groups
  include `dev`, so every container start tried to install pyarrow/datasets over
  the network. Measured with `--network none`: `error resolving DNS`. The
  entrypoint now calls `/app/.venv/bin/neuralguard` directly, `PATH` carries the
  venv, and `UV_NO_SYNC=1` stops an operator-invoked `uv run` from re-resolving.
- **Base images are pinned by digest** (`python:3.11-slim`,
  `ghcr.io/astral-sh/uv`). `uv:latest` had made the dependency resolver a
  floating input in a tree that hash-pins everything else.
- `.dockerignore` added: the build context is a security boundary now that the
  image is published — `.env`, `tenants/`, `audit_logs/` and `models/` cannot
  enter a builder tarball.
- Version is reported from a single source (`neuralguard.__version__`); the
  settings default is pinned to it by a test.

### Security

- 59 advisories published since 2026-09-25 cleared from the lock; `anyio`
  bumped for CVE-2026-63374 and CVE-2026-64847.
- Blocking `pip-audit` in CI; CycloneDX SBOM signed and attested keyless; the
  image now carries provenance and SBOM attestations of its own.
- GitHub Actions pinned to commit SHAs, with Dependabot keeping them — and the
  base images — current.

### CI

- CI restored after ~7 weeks disabled, then hardened: lint, tests on 3.11 + 3.12,
  a coverage gate at **90 %** with the full semantic layer and a live Postgres
  service, security scan, SBOM generate + keyless sign/attest, boot smoke
  (uvicorn + HTTP), semantic smoke, and now the image build/sign pipeline.

### Docs

- Community-facing README with a threat-model section that states what input
  filtering provably cannot do, and per-attack-class benchmark results rather
  than aggregates.
- `docs/ROADMAP.md` as a plan-of-record with an open register; `SECURITY.md`;
  seven runbooks (appliance, MCP gateway, artifact signing, backup/restore,
  fleet, secret rotation, TLS termination).
- Three truth sweeps removed stale figures (`1,401` vectors, `1,063` tests,
  `1,288` tests, a "drill pending" that had already passed, a dependabot PR that
  had already merged).

### Known limits — not in this release, and not claimed

- **SSE streaming is refused** (`stream: true` → 422, fail-closed). Hold-back
  scanning is not built.
- **JWT is HS256 with self-issued tokens only** — no RS256/OIDC/JWKS discovery,
  no refresh tokens, no Vault/SOPS integration.
- **The HITL approval broker is per worker and in memory** — multi-replica needs
  sticky routing or a shared store. Multi-replica runtime key rotation is
  undrilled, as are prometheus-adapter custom metrics and ingress/TLS.
- **One upstream per gateway**; multi-server MCP routing is not built.
- **The i18n pattern packs have had a machine self-audit, not a native-speaker
  review** — human sign-off is outstanding and the README carries the residual.
- **No named external deployment.** Every feature above is verified against
  our own attacks, not against a stranger's traffic.

## [0.2.1] — earlier

Tagged at `aa54cc9`. Predates the research-frontier controls (NG-1 → NG-9), the
MCP gateway, the audit ordering work and the fleet campaign; see `0.3.0`.
