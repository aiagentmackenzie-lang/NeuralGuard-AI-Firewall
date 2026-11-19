# Changelog

All notable changes to NeuralGuard. Version numbers follow the tag on `main`;
every entry below is merged, CI-gated and documented in the README or a runbook.
Things that are *not* shipped are listed under "Known limits" rather than
implied by silence.

The project had no changelog before this file, and the previous tag (`v0.2.1`)
predates the research-frontier controls, the MCP gateway and the fleet campaign
— so `0.3.0` carries everything since, not just the work of the last few weeks.

## [Unreleased]

### Fixed — two model-specific budgets that only CI could fail (2026-11-19)

The multilingual model made two assumptions false on a CI runner while both
still passed on the reference Mac:

- `test_load_time_under_2s` asserted a 2 s startup for what was a 24 MB model.
  The shipped artifact is 1.11 GB and measured **2,186 ms** on the runner. The
  budget moved with the artifact (10 s local / 30 s CI) and the measured values
  are in the test, rather than a constant that quietly stopped being true.
- Five real-model test modules inherit the project-wide **30 s** per-test
  timeout, which was written for a 384-dim English model. They now declare
  `pytest.mark.timeout(600)` — the FPR-invariant test alone embeds 95 probes
  against 6,503 vectors, and it was killed at 30 s mid-suite on `main`.

Both were green locally and red in CI, which is the point: a budget tuned to the
machine you develop on is not a budget.

### Fixed — the rebuild stopped re-embedding what it already had (CI, 2026-11-19)

`rebuild_corpus_vectors.py` embedded ~5,100 paraphrase candidates, ran the benign
guard over them, threw the embeddings away and **re-embedded the survivors** —
a full second pass for identical values, ~40 % of the rebuild. On a 2-core
runner with the 768-dim model that was the difference between a job that fits and
one that gets cancelled at its ceiling (measured: main's first run hit exactly
60 minutes and was killed mid-rebuild).

`benign_blocking_mask()` returns the keep mask from the embeddings the caller
already holds; the rebuild slices instead of recomputing. Verified **bit-identical**
(`np.array_equal` on the 6,503 × 768 matrix after a full rebuild, max abs diff
0.0), not assumed — a speedup that changes which vectors survive would be a
corpus regression wearing a perf badge. Local rebuild 7 min → 3 min; the
coverage job's ceiling 60 → 90 min so a genuinely cold cache can finish.

Also fixed, found while debugging the above: three test guards checked that the
`models/embedding-onnx` **directory** existed, but that directory survives a
clone because `config.json` is tracked. A missing `tokenizer.json` therefore
surfaced as a `FileNotFoundError` inside a "skip when the model is absent"
guard — 18 errors on a clean tree instead of 18 documented skips. The guards now
check `model.onnx` and `tokenizer.json`, the two files the engine actually loads.

### Changed — the semantic layer is multilingual (P1, 2026-11-19)

`all-MiniLM-L6-v2` was an English-trained encoder doing second-line duty for ten
languages. The capability matrix measured it blind on canonical PT/JA/VI/AR
attacks and inverted on Chinese. Replaced, by measurement rather than
reputation, with **`intfloat/multilingual-e5-base`** (768-dim) after benchmarking
four candidates at a *matched operating point* — the same 6.32 % guard escalate
share, because comparing models at one model's absolute thresholds made E5
collapse the guard (it dropped 6,512 of 6,513 corpus vectors and then "caught"
100 % of everything against the single survivor).

| Metric (semantic layer) | MiniLM-L6 | e5-base |
|:--|--:|--:|
| Multilingual attack recall (114 items, 11 languages) | 34 % | **60 %** |
| Multilingual benign fired (76 held out) | 22 % | **10 %** |
| Multilingual benign blocked | 4.0 % | **1.2 %** |
| NG-6 guard blocked / escalated | 0/95, 6.32 % | 0/95, 6.32 % |
| Whole-detector recall (`core`) | 73 % | **85 %** |

- Thresholds are model-specific and now settings (0.868 escalate / 0.88 BLOCK,
  calibrated on the rebuilt corpus), because 0.60/0.75 were MiniLM's scale and
  using them for E5 produced the all-block failure mode above.
- E5 needs asymmetric `query: `/`passage: ` prefixes, so embeddings carry a
  **role** (`semantic_query_prefix` / `semantic_passage_prefix`); passing the
  wrong one degrades recall with no error.
- `scripts/export_onnx.py` now defaults to `ScannerSettings().semantic_model` so
  the artifact and the runtime cannot disagree.
- The engine no longer hard-codes 384 dimensions or the `token_type_ids` input
  tensor: the dimension comes from the model and the input feed from the graph.
  XLM-R-style encoders reject the extra tensor with INVALID_ARGUMENT.
- **Quantization measured and rejected**: int8 is 278 MB and 25.7 ms vs 1.11 GB
  and 36.1 ms, but at equal guard precision it lost ~15 points of recall
  (44/114 vs 61/114). The model's usable margin on this corpus is ~0.02 and
  quantization noise is the same order — so the smaller artifact is not free and
  is not shipped.
- Cost stated, not hidden: ONNX median single-prompt embed 5.1 ms → 36.1 ms. The
  latency budget test (50 ms strict) and the eval suite's per-test timeout moved
  with the measurement.
- **Not fixed by this change:** the NG-6 hard-negative class still blocks 15/50
  at the *pattern* layer (benign text quoting an attack, with no notion of
  quotation). Chinese still fires on 50 % of benign ZH. Both are recorded as the
  next measured work, and neither is a model problem.

### Added — the capability matrix, per layer and per language (P0, 2026-11-18)

Every detection number this project published was English-only or aggregated
across languages, which is how a detector that is 91 % accurate in English and
blind in Vietnamese still prints one respectable figure.

`benchmarks/detection_eval/` measures each layer in isolation (structural,
pattern, semantic) and the deterministic core, per language, per attack family,
and split everyday benign from NG-6 hard negatives. Two numbers for benign, not
one: **fired** (not `allow` — the product cost, includes the judge) and
**blocked** (BLOCK only — the definition the published SLO uses). Conflating
them is how an honest SLO becomes a misleading headline.

Corpora: 87 machine-authored multilingual attacks (10 languages × 4 families +
7 agentic/obfuscation), 27 existing English attacks, 76 multilingual benign, the
45 F12 benign and the 50 NG-6 hard negatives, 10 controls. Provenance is a field
on every item; the README says plainly that machine-authored corpora are
directional, not authority. Committed baseline + ratchet test so a layer's
caught-rate cannot fall and its benign block-rate cannot rise unnoticed.

What it found on the build that introduced it (`main` = `426dc52`):

- **Core, per language:** EN 91 % caught / 22 % benign blocked; ZH and KO
  **100 % caught and 100 % blocked** — a detector saying yes to everything, not
  a score. PT is the weakest recall (**38 %**) despite being the only language
  ever given a human-authored benign pass.
- **Semantic layer:** reproduces the published SLO exactly (0/95 blocked,
  6/95 escalated = 6.32 %) and is blind in PT/JA/VI/AR (0 % caught). For ZH its
  signal is inverted: 88 % of attacks caught only by over-match, 100 % of benign
  blocked.
- **The published guarded FPR is a semantic-layer number.** The pattern layer
  BLOCKs **15/95 (15.79 %)** on the same probes — the hard-negative class
  (benign text *quoting* an attack) has no notion of quotation and was never in
  the A1 gate, which scores 45 plain benign prompts. `docs/FPR_SLO.md` now says
  this in its own section with the measured table; the README's A1 block says it
  too. Measured work, not a wording fix.

### Added — the README's test count is finally a measurement (docs)

`tests/unit/test_docs_consistency.py` compared the README against
`docs/FPR_SLO.md` (corpus size) and against the loaded rule packs (rule count).
It did not compare the headline test count against anything — which is why that
number sat **47 tests stale** on `main` (1,518 claimed, 1,565 collected) after
two releases that added contracts. The same class of drift the file exists to
catch, in the one slot with no guard.

`TestTestCountClaim` now reads the claimed count from the README and compares it
to what pytest collects, delegated to a `--collect-only` subprocess so the
number comes from the runner rather than from arithmetic in the test.

### Fixed — the `DAN` token no longer matches inside ordinary words (i18n + JB-001)

`JB-ZH-003`, `JB-JA-003` and `JB-KO-003` opened with `(?i)(?:DAN|…)` — no word
boundary — while the English `JB-001` had already written `\bDAN\b`. Because
every pattern runs against every prompt regardless of language, a French
sentence was blocked by a **Chinese** rule id: `Mode développeur désactivé dans
le navigateur.` (`da**n**s`), `Le danseur a abandonné la scène.`, and the English
name in `Dan is our new account manager.` all came back BLOCK at 0.88-0.96.

Found by `scripts/i18n_fp_sweep.py`, a new mechanical instrument: 76 benign
everyday sentences across the 10 languages plus 10 positive controls, run
through the real `PatternScanner`, exit 2 if a control fails to fire so an
instrument failure cannot look like a clean result. It measured **39/76 benign
probes firing (51 %)** — with Portuguese at **0/8**, the one language that had
ever been probed before, which is what makes the other nine a finding rather
than a style preference.

- i18n packs now use `(?-i:\bDAN\b)`; `JB-001` drops its global
  case-insensitivity, so the canonical upper-case token fires and the name
  `Dan` does not. Stated trade-off: lower-case "dan mode" no longer fires.
- Guards written and run red first:
  `TestCrossLanguageDanFalsePositive`, `test_person_named_dan_is_not_a_jailbreak`.
- Deterministic gates re-run on the changed tree and unchanged: **A1 ASR 0.00 %
  / FPR 0.00 % / exact-match 100 %**, **NG-5 Unsafe-ASR 38.89 %, Safe-ASR
  12.50 %, hard BLOCKs 0**, **B4 baseline 0.00 % / guarded 0.00 %**.
- The remaining ~39 candidate false positives are documented per rule family in
  `docs/i18n_native_review_request.md` and left for the native-speaker pass: each
  one needs a judgement (role-switch verb without a suspicious role,
  developer-mode named in the *disabled* state, "ignore the previous footnote",
  "repeat everything above" as a study instruction) that a regex cannot make.

### Fixed — the stale README headline (docs)

The "Current state" line now reads `1,569 tests collected (1,557 passed / 12
skipped locally), 91.16 % measured coverage` — measured, not carried forward, and
the guard caught its own first regression while this branch grew by three tests.
Deliberately unchanged: no production behaviour, no version bump. A docs-and-
guard change does not earn a release.

## [0.3.2] — 2026-11-17

### Added — the HITL store posture is readable per pod (NG-15)

`hitl_store=redis` is the one setting a multi-replica deployment must get
right, and the startup refusal cannot enforce it: it reads `SERVER_WORKERS`,
not replicas behind a Service. Until now the only check was reading the
manifest and trusting it.

`GET /v1/info` → `mcp_gateway.hitl` now reports, per pod:

- `store` (`memory` | `redis`) and `shared_across_workers` — the structural
  property, derived from the store rather than from the config that asked for
  it;
- `store_id` — a 12-hex fingerprint of the store endpoint (host, port,
  database) with **credentials and scheme excluded**, so two pods can be proven
to point at the same Redis without either disclosing a password, and rotating a
  password cannot look like moving to a different store;
- `timeout_seconds`, `ttl_seconds`, `max_pending` — the bounds a held request
  depends on.

A per-worker store now reads as `shared_across_workers: false`,
`store_id: null` instead of a footnote in a runbook. 13 contracts.

### Fixed — a store outage is a named, audited refusal (NG-15)

Found by the two-pod drill, by accident and then on purpose: with a HITL Redis
URL that could not authenticate, the escalation route raised `HitlStoreError`
out of the handler. The security property held — nothing was approved, nothing
reached the upstream — but the operator saw `500 {"error":"internal_error"}`
with **no rule id and no audit event**, so a SIEM watching the audit chain
could not see that the control was refusing every approval-gated call. The
decide route was worse: it caught `HitlError`, which `HitlStoreError`
subclasses, and answered `409 mcp_escalation_not_pending` — telling the
approver the escalation had expired, which is a different and wrong story.

Now every store-outage surface is `503 mcp_escalation_store_unavailable` with
rule id `MCP-HITL-STORE`: audited (verdict `block`, outcome
`escalation_store_unavailable`) on raise and on decide, reported but **not**
audited on the approver list path — that endpoint is polled, and one audit
event per poll would turn an outage into a flood in the tamper-evident trail.
`HitlStoreError` moved to `neuralguard.mcp.hitl` so the route layer can catch
it without the optional `redis` extra (re-exported from `hitl_redis`).
5 contracts, written and run red first.

**Drilled in-cluster, not asserted** — two pods behind a Service, dated
2026-11-17, evidence and the negative control in
[`deploy/kubernetes/README.md`](deploy/kubernetes/README.md) §"Two-pod HITL
drill" and [`docs/runbooks/mcp_gateway.md`](docs/runbooks/mcp_gateway.md).

### Added — the live API-key set is readable per pod (P2-4)

Rotating a static key in a multi-replica deployment is documented as
"env/redeploy", and the reason is that the runtime rotation API is per process.
Neither was observable from outside: after a Secret change you could watch pods
roll but could not ask a pod **which key set it is serving**, so "has the old key
been revoked everywhere?" was answered by waiting and hoping.

`GET /v1/info` → `auth` now reports, per pod:

- `key_count` and `key_set_id` — a 12-hex fingerprint over the **live** set
  (env keys merged with the durable runtime store), built from `sha256(key)` plus
  each key's tenant and workload-agent binding, sorted, so it is order-insensitive
  and never contains key material;
- `runtime_rotation_durable` — whether a runtime rotation would survive a restart;
- `jwt_enabled`, `enabled`.

Two pods reporting the same `key_set_id` serve the same credentials; differing
ids are the list of pods still accepting a credential you believe you revoked.
12 contracts (order-insensitivity, rebinding, add/remove, leak guards, live-set
behaviour, and the empty set reported as `null` rather than a fake id).

### Fixed — a failed durable rotation was applied anyway (P2-4)

Found in the rotation drill. With a keys-file volume the container user could not
write, `POST /v1/auth/keys/rotate` answered `500 internal_error` — and the key was
**live on that pod only**, persisted nowhere, and lost on restart: a split-brain
credential reported as a failure.

`RuntimeKeyStore.add`/`remove` now **persist first, then apply**, so an
unwritable store leaves the live set untouched, and the route answers
`500 {"error": "key_store_write_failed"}` — a 500 that is true. 5 contracts, red
first.

### Drilled — rotation windows with two replicas (P2-4)

Measured against two pods behind a Service, sampled ~2×/second from a separate
pod (client-visible), posture read per pod:

| Operation | Revoked/new key window | Availability cost |
|---|---|---|
| Additive (old + new) | new key usable **+12 s**, fully at **+22 s** | none |
| Removal, rolling | revoked key still valid **21 s** (~23 % of probes succeeded) | none |
| Removal, `Recreate` | revoked key valid **0 s** | **~7 s outage** |
| Runtime API, durable **shared** key store | reaches **one pod**; the other refuses it while *able to read the file* | none |

Conclusion, stated plainly: **runtime rotation is per process by construction**,
a shared keys file buys durability but not propagation (the reload happens at
boot), and revocation under a rolling update is bounded by the rollout — so a
compromised key wants `Recreate`, and a scheduled rotation wants two phases.
Evidence and commands: [`docs/runbooks/secret_rotation.md`](docs/runbooks/secret_rotation.md).

### Known limit, unchanged

A store that fails **during** a wait is still reported as
`403 mcp_escalation_timeout`, not as `MCP-HITL-STORE`: the refusal is the same
and fail-closed, but the diagnosis is coarser. Deliberately not changed here —
`RedisHitlStore.wait` returns a refusal on store error by contract, and the
route cannot tell the two apart without changing that contract.

## [0.3.1] — 2026-11-17

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

### Release verification — 0.3.1, checkable by anyone

Published by `.github/workflows/image.yml` on tag `v0.3.1` (run `44185951141`,
3/3 jobs: airgapped boot smoke, amd64+arm64 build+push, keyless sign + CycloneDX
attest + verify). Verified **outside CI** against the registry the consumer
actually pulls from:

| Check | Result |
|---|---|
| `0.3.1` and `latest` → digest | both resolve to `sha256:6d1c0022f146…` |
| Anonymous index pull (no credentials) | `HTTP 200` |
| `cosign verify` (identity-scoped, GitHub OIDC issuer) | passes; Rekor entry verified offline; payload carries the image signature **and** the `https://cyclonedx.org/bom` attestation |
| Boot from the published digest, `/v1/health` | `"version":"0.3.1"`, `environment: production` |
| Schema created by the artifact itself | `audit_events` present after first boot against an empty Postgres |
| Table dropped under the running artifact | `/v1/ready` → `audit_db:"fail"`, aggregate **`degraded`** (v0.3.0 answered `ok`/`healthy` here) |
| Boot against an unreachable Postgres, `INIT_ATTEMPTS=2` | **two** `db_init_attempt_failed` then `db_init_failed` (v0.3.0 logged once and never retried) |

So both deployment-path fixes and the honest readiness report are properties of
the artifact, not only of the tree.

### Known limits — unchanged by this release

- The Kubernetes cluster drill **has been re-run against the 0.3.1 digest**
  (see `deploy/kubernetes/README.md`): the node pulled the published digest, the
  full check table passed, and the late-Postgres sequence that used to require a
  `rollout restart` on 0.3.0 converged on its own (attempts 1–2 failed, Service
  restored at 22:23:31Z, attempt 5 created the schema, `RESTARTS 0`).
- Multi-replica runtime key rotation, prometheus-adapter metrics and ingress/TLS
  remain undrilled; SSE streaming is still refused `422`.
- The shared HITL store closes the *broker* half of the multi-replica gap, and
  was drilled across **two OS processes** with a memory-store negative control.
  It has **not** been drilled across two pods behind a Service; that is the
  remaining evidence.
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
