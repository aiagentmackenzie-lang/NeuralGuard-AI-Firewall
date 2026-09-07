# NeuralGuard Roadmap

> **Source of truth for what ships next.** This is a tracked (public) roadmap.
> The gitignored `PRODUCTION_HARDENING_PLAN.md` is the internal ledger of
> what already landed; this doc is what is **planned**.
>
> **Last updated:** 2026-09-07 · **Status: Sprints A/B/C and the P2
> enterprise track are ALL SHIPPED (v0.2.0/v0.2.1).** The roadmap below is
> retained as the plan-of-record; every phase carries a status banner. What
> ships next is driven by the open register in
> `PRODUCTION_HARDENING_PLAN.md`: i18n native-speaker sign-off (P2-11,
> humans), the K8s cluster drill (P2-6), SSE hold-back streaming (demand-
> driven), RS256/OIDC + Vault/SOPS residuals (P2-4), and cross-worker audit
> ordering (WORM/DB-sequence).

The P0 + P1 production-readiness sweep is closed (see the merge commit
`a40d6f2` and `PRODUCTION_HARDENING_PLAN.md`). What remains is the
**differentiator track** (Sprint A + B) and the **enterprise track**
(Sprint C). Sprints A and B are the work that turns NeuralGuard from "a
solid AI firewall" into "the AI firewall with evidence no open-source
competitor has" — and they lift **two** portfolio projects at once
(NeuralGuard + NeuralStrike).

---

## Sprint A — NeuralGuard ↔ NeuralStrike benchmark harness

**One-line:** Measure NeuralGuard's detection efficacy against
NeuralStrike's attack modules and publish an Attack-Success-Rate (ASR)
table. No open-source AI firewall ships this. It is the single highest-
leverage portfolio piece left.

**Why first:** It is cheaper than Phase 3, it produces a defensible number
for interviews/client calls ("my firewall blocks N% of my own offensive
framework's attacks"), and it gives Phase 3 a measurement harness to
prove multi-turn detection works — otherwise Phase 3 is unverifiable
vapor.

**Lives in:** `benchmarks/ng_vs_ns/` inside NeuralGuard. NeuralStrike is
invoked as the attacker (via its Python API or CLI; not a submodule — a
`pip install -e ../NeuralStrike` dev dependency, documented in the
harness README).

### Phases

- **A1 — Deterministic attack corpus + regression gate.** A curated,
  labeled set of attack prompts (`benchmarks/attack_corpus.jsonl`) each
  tagged with `expected_verdict` (block-family) and the NeuralStrike
  module it represents. A harness that runs them against `/v1/evaluate`
  and asserts NeuralGuard returns the expected verdict. This is a
  **NeuralGuard regression suite** independent of NeuralStrike and lands
  first as a CI gate. Success: ASR on the labeled corpus = 0% (no attack
  that should BLOCK is ALLOWed), FPR on a benign corpus < 2%.

- **A2 — NeuralStrike attacker integration.** Wire NeuralStrike's
  `JailbreakForge` (iterative Attacker-mutated prompts), `ContextPoison`,
  and `MCPInterceptor` as live attackers against NeuralGuard in three
  configurations:
  1. `pattern-only` (deterministic baseline)
  2. `pattern + semantic` (ONNX similarity)
  3. `pattern + semantic + judge` (full pipeline)
  Record ASR, FPR, and p95 latency per configuration. Success: a
  monotonic ASR drop across the three configs (each layer helps), and a
  documented ASR number per NeuralStrike module.

- **A3 — Publish + nightly CI.** A markdown results table in the README
  (`## Benchmarks: NeuralGuard vs NeuralStrike`) generated from the
  harness JSON, plus a nightly `.github/workflows/bench.yml` that re-runs
  and **fails on ASR regressions** (the same pattern as the perf gate).
  Success: a green nightly bench job and a dated results table in the
  repo.

### Honest non-goals (Sprint A)
- Not a neutral third-party benchmark — NeuralStrike is also Raphael's.
  The README will state this plainly: "attacker and defender are by the
  same author; this measures defense-in-depth, not independence."
- No human red-team verdicts — the harness uses the pattern scanner's
  own labels + a small hand-labeled holdout for ground truth.
- Does not cover Phase 3 multi-turn attacks until Sprint B lands.

### Deliverables
- `benchmarks/ng_vs_ns/harness.py` + `attack_corpus.jsonl` + `benign_corpus.jsonl`
- `benchmarks/ng_vs_ns/README.md` (how to run, the same-author caveat)
- `.github/workflows/bench.yml` (nightly ASR regression gate)
- README section with the results table

---

## Sprint B — Phase 3: Agent Guardian (the moat)

**One-line:** Multi-turn detection, prompt-template analysis, and a
dedicated memory-poisoning rule. This is the capability commercial AI
firewalls mostly **don't** have — the real moat.

**Why second:** It needs Sprint A's harness to prove it works (multi-turn
detection is otherwise unverifiable). It is also the bigger engineering
surface, so it deserves the measurement harness first.

### Phases

- **B1 — ConversationState + multi-turn detection.** ✅ SHIPPED & MERGED
  (`c96008e`). `AgentGuardianScanner` (`scanners/agent_guardian.py`) keyed on a
  request `session_id`. Bounded per-session sliding window (in-memory per
  worker; Redis backend option designed for a B1+ follow-up). Detects delayed /
  garden-path injection (AG-DELAYED-001, BLOCK), role drift / persona erosion
  (AG-DRIFT-001, BLOCK), gradual system-prompt extraction (AG-EXT-ACCUM-001,
  ESCALATE), gradual memory poisoning (AG-MEM-ACCUM-001, ESCALATE). Fail-closed
  on state-store errors. Config: `agent_guardian.enabled`,
  `session_window_turns`, `backend`, thresholds.

- **B2 — Prompt-template analyzer.** ✅ SHIPPED & MERGED (`77d35da`).
  `neuralguard analyze-template` CLI + `POST /v1/analyze/template` endpoint —
  static injection-sink analysis (no LLM call): untrusted-variable
  interpolation, missing delimiter fences, ambiguous instruction precedence,
  action-adjacent variables, raw structured-data injection. `--fail-on-high`
  CI gate. `src/neuralguard/analysis/template_analyzer.py`.

- **B3 — ASI06 dedicated rule + canary unstub.** ✅ SHIPPED & MERGED
  (`63ec379`).
  - **Memory poisoning (ASI06) — dedicated single-turn T-MEM rules**
    (`scanners/pattern.py`, MEM-001..004): MEM-001 HIGH→BLOCK catches explicit
    memory/RAG store writes ("store this into your long-term memory", "save
    to the knowledge base", "write to the vector database", RAG / context-
    store / core-instructions). MEM-002/003/004 MEDIUM→SANITIZE catch
    conditional future-behavior, persistent belief poisoning, and persistent
    self-rule directives ("from now on, when asked X, do Y"; "always treat X
    as Y"; "permanently adopt the rule"). The Agent Guardian scanner (B1) still
    catches the *cross-turn accumulation* via AG-MEM-ACCUM-001 — T-MEM adds
    the dedicated single-turn surface. Distinct from JB-010 (jailbreak-framed
    benign-turn poisoning), which remains in the JB category.
  - **Pattern count:** 50+ → 54+ across 9 categories (was 8; the `MEMORY_POISONING`
    category now stands on its own alongside ROLEPLAY, EXFILTRATION, etc.).
  - **Residual FPR (documented honestly):** MEM-002/003 match benign persistent-
    preference statements ("from now on, when asked for a summary, respond in
    bullets"). Intentional, bounded to SANITIZE never BLOCK — a triage signal
    rather than a content-modifying BLOCK.
  - **Canary token verification** (`canary_leaked`, previously stubbed `false`)
    — unstubbed with deterministic HMAC-SHA256 canaries. `CanaryManager`
    (HMAC-SHA256 of `session_id|label`, keyed by server secret, base32
    80-bit, `NGCANARY-...` prefix). Bounded labels A..H (≤8 per session).
    Deterministic mint + detect — no server-side token storage; mint and
    detect re-derive from `session_id`. Safe-by-default: `check_leak` returns
    None when disabled/misconfigured/empty-session (additive signal that
    never raises); mint raises on disabled/misconfigured (fail-closed).
  - **Wiring:** `CanarySettings` (NEURALGUARD_CANARY_*) → `NeuralGuardConfig.canary`
    (`enabled`, `secret` ≥32 chars enforced in production, `token_count` 1..8).
    `main.py` builds the `CanaryManager` on `app.state` only when enabled; the
    production lifespan refuses to start with no/short secret in production.
  - **API surface:** `POST /v1/canary/mint` (503 disabled, 422 bad session /
    bad count) + unstubbed `canary_leaked` field in `/v1/scan/output`. On a
    canary leak the response surfaces a `CANARY-LEAK-001` finding under
    `SYSTEM_PROMPT_EXTRACTION` (HIGH, BLOCK) with redacted evidence; the
    verdict is forced to BLOCK and the reason string is appended to the
    response. The canary check runs BEFORE the dispatcher so it can drive the
    403 itself. Non-200 `/v1/scan/output` now returns the full `ScanOutputResponse`
    body (`canary_leaked` + `redacted_output` + `findings`) at the action
    status code.
  - **CLI:** `neuralguard canary-mint <session_id> [--count N] [--json]`.
  - **Tests:** 98 new — `tests/unit/test_canary.py` (25) +
    `tests/unit/test_pattern_memory.py` (34) + `tests/unit/test_canary_api.py` (12)
    + 4 prod-gate cases in `tests/unit/test_app_lifespan.py` + 4 CLI cases in
    `tests/unit/test_cli.py`. Branch gate: 658 passed / 1 skipped locally,
    ruff + mypy clean, py_compile OK on all 12 touched files. Integrity
    verified: no memory/RAG/canary phrase leaks in non-target files; em-dashes
    preserve house style (already used in `agent_guardian.py`).

- **B4 — Benchmark integration.** ✅ SHIPPED & MERGED (`45a08d2`).
  - **Multi-turn harness** (`benchmarks/ng_vs_ns/multiturn_harness.py`):
    replays curated multi-turn sequences targeting
    `AG-DELAYED-001` / `AG-DRIFT-001` / `AG-EXT-ACCUM-001` /
    `AG-MEM-ACCUM-001`, plus a live `AgentPivot.exploit_delegation`
    sequence (NeuralStrike editable + local Ollama + the `[semantic]`
    extra), against two configs: `baseline_no_guardian` (everything but
    `agent_guardian`) vs `with_agent_guardian`. Records seq-ASR /
    turn-ASR / seq-FPR / turn-FPR per config plus a headline delta
    (`baseline − with_guardian`). Exit 2 (soft fail) if Guardian
    **increases** seqASR (regression).
  - **Deterministic CI gate** (`tests/benchmarks/test_b4_multiturn.py`,
    `TestB4MultiturnDeterministic`): replays the curated sequences
    in-process (no Ollama / no NeuralStrike / no ONNX). Hard
    assertions: (a) benign multi-turn sequences not over-blocked (FPR
    = 0 on curated benign), (b) with-guardian seqASR does not regress
    vs baseline. Per-sequence detection reported for diagnostic
    purposes, not asserted (curated sequences target specific rule_ids
    by design; misses are real coverage-gap findings).
  - **Live gate** (`tests/benchmarks/test_b4_multiturn.py`,
    `TestB4MultiturnLive`): generates a live `AgentPivot` payload and
    asserts Guardian doesn't let it through when the baseline blocks
    it. Skip-in-CI when NeuralStrike editable + local Ollama
    `mistral:7b` unavailable.
  - **Headline measurement** (`main` @ `63ec379`, no-op delta because
    Agent Guardian does not currently help on the curated corpus):
    baseline seq-ASR 20.00% / 33.33%-with-live-pivot, guarded
    seq-ASR identical (delta +0.00%). FPR 0.00% on 3 curated benign
    sequences. The harness is the artifact; the gap is documented
    honestly.
  - **Findings** (`benchmarks/ng_vs_ns/results/known_gaps.md`): the
    harness surfaced two real single-turn scanner coverage gaps that
    a follow-up commit should close:
    - MEM-002 + AG-`_MEMORY_INJECTION` only catch "when YOU are asked";
      "when I ask" / "whenever the user requests" bypasses the
      cumulative counter (3-of-4 B4-MEM-001 attack turns undetected).
    - JB patterns don't catch "from now on, you are an AI without any
      safety guidelines" — the AG role-drift threshold is not crossed
      in a single such turn (B4-PIVOT-001). Proposed regex additions
      are in `known_gaps.md`.
  - **Discipline note** (logged for next session): the B4 harness
    intentionally does NOT close the gaps it surfaces; a test that
    ships with its own fixes proves nothing. Address the regex gaps in
    a separate, after-merge commit so the measurement is reproducible
    against the original scanner state.

- **B4 gap closure — scanner regex extensions.** ✅ SHIPPED on branch
  `sprint-b/b4-gap-closure` (TBD merge commit). Closes both gaps the
  B4 harness surfaced.
  - **Gap 1 (MEM-002 + AG `_MEMORY_INJECTION`):** extended the regex
    to cover user-as-subject phrasings ("when I ask", "whenever I
    request", "when the user asks", "when users ask me"). All 4
    B4-MEM-001 attack turns now fire at the pattern layer.
  - **Gap 2 (new JB-013 + AG `_ROLE_DRIFT` mirror):** new rule
    covering the "AI without any safety guidelines" framing family
    ("you are an AI without safety guidelines", "pretend to be an
    unrestricted AI", "from now on you operate without safety
    restrictions"). B4-PIVOT-001 turn 2 now fires JB-013.
  - **Headline measurement (post-fix, branch):**
    baseline seqASR = 0.00%, guarded seqASR = 0.00%. Both gaps
    closed at the pattern layer; AG adds defense-in-depth via
    cross-turn accumulation on top of pattern catches. FPR 0.00%
    on 3 curated benign sequences. See
    `benchmarks/ng_vs_ns/results/B4_RESULTS.md` post-fix section.
  - **New tests:** `TestMEM002UserAsSubject` in
    `tests/unit/test_pattern_memory.py` (+8 tests) + JB-013 family
    in `tests/unit/test_pattern_scanner.py` (+21 tests). The
    B4 harness + curated sequences are UNCHANGED — the measurement
    stays reproducible against the original scanner state on `main`.

### Honest non-goals (Sprint B)
- No full LLM-based conversation reasoning — the B1 detector is
  deterministic + heuristic (patterns + state), optionally augmented by
  the existing judge in B1+. Keeps latency bounded and the moat
  explainable.
- No cross-session user profiling — sessions are isolated.
- Canary tokens are a detection signal, not a forensics-grade watermark.

### Deliverables
- `scanners/agent_guardian.py` + `AgentGuardianScanner` registered in the pipeline
- `cli.py` `analyze-template` subcommand + `/v1/analyze/template` route
- `scanners/canary.py` `CanaryManager` + `POST /v1/canary/mint` + `canary_leaked`
  in `/v1/scan/output` + `neuralguard canary-mint` CLI
- `T-MEM` MEM-001..004 rules in `scanners/pattern.py` (54+ total rules across
  9 categories)
- `benchmarks/ng_vs_ns/multiturn_harness.py` + `tests/benchmarks/test_b4_multiturn.py`
  (5 curated attack sequences + 3 curated benign + live AgentPivot; deterministic
  CI gate + live skip-in-CI gate). Findings in
  `benchmarks/ng_vs_ns/results/known_gaps.md`.
- Tests (unit + a multi-turn redteam fixture) + benchmark extension
- README + `PRODUCTION_HARDENING_PLAN.md` updates; re-score in
  `Security_Portfolio_Reference.md`

---

## Sprint C — Enterprise track (P1-2 + P2)

Post-moat. **ALL SHIPPED** — see the per-item banners; details + residuals
in `PRODUCTION_HARDENING_PLAN.md`.

- **P1-2 — Per-tenant config.** ✅ SHIPPED & MERGED (`aadae4d`). ``tenants/<id>.yaml|json`` override files loaded into an
  in-memory `TenantConfigRegistry` keyed by tenant id; per-tenant
  RPM/burst overrides + per-tenant enable/disable for the three optional
  scanners (Agent Guardian, Semantic, Judge). Structural + Pattern are
  mandatory and CANNOT be disabled per-tenant. Override resolution is
  ``None = inherit global`` (every field defaults to None, so a partial
  tenant file degrades to the global config, never to an unsafe zero).
  Unknown-tenant miss is fail-OPEN to the global default (never a 403 —
  denying on a config miss is a self-inflicted DoS). The tenant config is
  a CEILING for the client ``request.scanners`` field: a client may
  narrow but never widen past the tenant + global registration.
  Hot-reload via directory-mtime poll (background task; parse error keeps
  last-good + logs — never blanks the registry, never raises into the
  request path). Read-only surface: ``GET /v1/tenants`` +
  ``GET /v1/tenants/{id}`` (auth-gated, tenant-binding-enforced, no
  secrets) + ``neuralguard tenants list|info <id>`` CLI. YAML tenant
  files require the optional ``[tenants]`` extra (PyYAML); ``.json``
  tenant files work with no extra. Production lifespan refuses to start
  if tenant mode is on, a YAML file is present, and PyYAML is not
  installed. Tests: +62 (config model, registry + hot-reload, pipeline
  ceiling enforcement, rate-limit per-tenant, API + lifespan gates, CLI).
  Branch gate: 759 passed / 1 skipped, ruff + mypy clean.
- **P2-4 — JWT/OAuth2 + key rotation API.** ✅ SHIPPED (2026-09-05, v0.2.0).
  HS256 short-lived JWTs (alg allowlist, exp enforced) via `POST
  /v1/auth/token`; runtime rotation via `POST /v1/auth/keys/rotate`
  (admin-tenant only, durable via `NEURALGUARD_AUTH_KEYS_FILE`, atomic 0600
  writes; runtime-only rotation refused in production). Residuals (demand-
  driven follow-ups, not claimed): RS256/OIDC discovery (JWKS infra),
  refresh tokens, Vault/SOPS integration.
- **P2-6 — Kubernetes artifacts.** ✅ SHIPPED (2026-09-05) — namespace,
  ConfigMap, Secret template, Redis (requirepass via Secret), Postgres
  StatefulSet, Deployment + Service, HPA; kubeconform 10/10 strict-valid;
  hardened secret posture in v0.2.1. **Cluster drill PENDING** (never
  applied to a real cluster — see `deploy/kubernetes/README.md`).
- **P2-5 — SBOM/image signing (cosign).** ✅ SHIPPED (2026-09-05) — CI signs
  + attests the SBOM keyless (identity-scoped verify in-job, bundles
  uploaded); local key-based flow proven end-to-end. Registry image signing
  documented as an ops step (`docs/runbooks/artifact_signing.md` §4) — CI
  builds no image, so an image-signing claim would be vapor.
- **P2 — Restore 90% CI coverage gate.** ✅ CLOSED (2026-09-05) — CI
  regenerates the ONNX model and rebuilds the corpus from tracked sources;
  the full suite runs and 90% is enforced (pyproject `fail_under = 90`).
- **P2-7 — SIEM/alert routing.** ✅ SHIPPED (2026-09-05) — `SiemRouter`:
  Splunk HEC (native), generic JSON webhook, SecurityScarletAI (ECS
  IngestEvent mapping); BLOCK-rate spike detector (edge-triggered +
  cooldown); enabled-without-sink refuses in production (F23 closed in
  v0.2.1 to also cover the scarletai-only posture).

---

## Ordering and gates

1. **Sprint A** first (cheapest, produces the measurement harness + a
   portfolio differentiator immediately).
2. **Sprint B** second (uses Sprint A's harness to prove multi-turn
   detection; the real moat).
3. **Sprint C** as enterprise demand requires.

Every phase gated on the existing CI bar: `ruff + ruff format + mypy +
pytest + coverage floor (86% during the sprints; 90% since P2 closed) +
boot-smoke`. Sprint A added a nightly bench gate; Sprint B extended it.
No phase was "done" until the gate was green on `main` and the portfolio
reference re-scored.

## Post-roadmap execution (2026-09-04 → 2026-09-07, v0.2.0/v0.2.1)

Beyond the Sprints above, the production-hardening sweep executed against a
verified issue ledger (F1–F23) — see the gitignored
`PRODUCTION_HARDENING_PLAN.md` for the closed-items ledger and the live open
register. Highlights landed on `main`:

- **Correctness sweep (F1–F8, F13–F15):** AG-before-Pattern ordering (F2),
  anchorless memory-poisoning regexes (F3), the Agent Guardian Redis
  session store (F4 — was a silent no-op), the `NEURALGUARD_*` env-name
  rename + unknown-key refuse gate (F5), user-role-only scanning with an
  explicit `scan_all_roles` opt-in (F6), dead-knob close-outs (F7), canary
  doc contradiction (F8), CI/workflow dedupe (F13), `audit-verify` CLI
  (F14), verdict-header-on-200 + release tagging (F15).
- **Judge modernization (F10):** configurable timeout, egress gate
  (`NEURALGUARD_SCANNER_JUDGE_ALLOW_EGRESS`), random data fences around
  judged text, concurrency semaphore, startup warmup, 27B re-measurement of
  `judge_resolves_escalate` (A2_RESULTS). Pattern-budget (F11). Corpus
  hygiene + 5.4× augmentation (F12; 7,623 vectors).
- **Standalone appliance (F9):** `POST /v1/proxy/chat/completions`, compose
  profile, runbook, boot-drill verified; streaming refused 422 fail-closed.
- **P2 enterprise track:** P2-3 (ASI04/ASI10 dedicated rules; 123 patterns
  total) · P2-4 (JWT + rotation API) · P2-5 (cosign keyless) · P2-6 (K8s
  artifacts + HPA; drill pending) · P2-7 (SIEM + ScarletAI) · P2-8
  (pure-ASGI middleware — the global exception handler genuinely backstops,
  proven by test) · P2-9 (coverage headroom) · P2-10 (Ed25519 signing +
  JSONL/pg chain verification, live-fire proven; pg INSERT ship-blocker
  found and fixed) · P2-11 (i18n machine self-audit done; HUMAN sign-off
  pending — `docs/i18n_native_review_request.md`).
- **v0.2.1 fix batch (2026-09-07):** F23 scarletai sink gate, postgres
  event_sig at rest + INSERT fix, `.env.example` full operator surface, CI
  timeout-minutes everywhere, hardened appliance/K8s secrets, judge scope
  consistency (F6), `system_prompt_hash` vapor removal (F7 close-out),
  `neuralguard audit-verify --pg-url` + live-fire tests enforced per-PR in
  CI.

## v0.3+ research frontier — decision register (2026-09-07)

Evidence base: the 2026-09-07 research sweep (peer-reviewed 2025–26 sources:
USENIX Security 2026, arXiv, KDD 2026, ICLR 2025, OWASP GenAI, MCP spec —
synthesis in the machine-local `docs/RESEARCH_FRONTIER_2026.md` ledger;
durable conclusions graduate into this register). Three paradigm shifts:
(1) input detection has a provable ceiling (Ball et al. 2025; controlled-release
prompting scored 92–100% ASR against production LLMs at USENIX Security 2026);
(2) the guard itself is the attack surface (Prompt Overflow ~100% bypass,
arXiv:2605.23196; mutation robustness; FPR weaponization, KDD 2026);
(3) agent security moved to protocol + provenance enforcement (CaMeL,
MCP 2026-07-28, signed tool-manifest integrity). Every item below carries its
evidence anchor; statuses update per the "plan, not a promise" rule.

| # | Item | Priority | Effort | Evidence anchor | Status |
|---|---|---|---|---|---|
| NG-1 | Reasoning/intermediate-token output scanning (opt-in, fail-closed on unscannable payloads) | P1 | ~3–5 d | USENIX '26 §6.2 | ✅ Shipped 2026-09-07 (`NEURALGUARD_PROXY_OUTPUT_REASONING_SCAN`) |
| NG-2 | README threat-model honesty section w/ USENIX citation | P0 | ~½ d | USENIX '26 §8 | ✅ Shipped 2026-09-07 (README threat-model section) |
| NG-3 | Decode-then-activate Agent Guardian signal (session-scoped, deterministic) | P1 | ~2–3 d | USENIX '26 §3–4 | ✅ Shipped 2026-09-07 (`AG-DECODE-001`) |
| NG-4 | Overflow-resistant contiguity-gated windowed aggregation | **P0** | ~2–3 d | arXiv:2605.23196 §6.2 | ✅ Shipped 2026-09-07 (`SEM-OVERFLOW-001` / `SEM-W-***`) |
| NG-5 | 16-mutation red-team gate (Unsafe-ASR + Safe-ASR, nightly, NeuralStrike pairing) | **P0** | ~2–3 d | HF study 2026-05 | ✅ Shipped 2026-09-07 (`benchmarks/ng_vs_ns/mutation_harness.py` + nightly bench job; measured baseline: Unsafe-ASR 56.94% / Safe-ASR 12.50% / 0 hard BLOCKs — see the honest-baseline note) |
| NG-6 | Published per-tenant FPR SLOs (corpus-hygiene hard negatives in semantic rebuild) | P1 | ~1–2 d | KDD '26; A2 history | ✅ Shipped 2026-09-07 (`corpus/benign_hard_negatives.jsonl` 50 NotInject-style probes, rebuild hard-negative guard, boot FPR self-check + `/v1/info` surface, `TenantScannerOverrides.semantic_block_threshold` dial; **measured guarded FPR 0.00% across 95 probes** — see `docs/FPR_SLO.md`) |
| NG-7 | MCP gateway: signed tool-inventory baselining + rug-pull detect/block (reuse P2-10 Ed25519 + audit chains) | **P0** | ~1–2 wk | MCP 2026-07-28; MDPI FI 18(5):243 | ✅ Shipped 2026-09-07 (`neuralguard.mcp`: canonical catalog hashing, Ed25519-signed baselines, strict/advisory baseliner — drift withholds the catalog + poisoned state refuses all calls; unknown-tool executes refused in every mode; drift events in the P2-10 audit chain; runbook `docs/runbooks/mcp_gateway.md`) |
| NG-8 | Header-based per-tool Intent Gate (`Mcp-Method`/`Mcp-Name` before body parse) | P0 (with NG-7) | ~1 wk | MCP 2026-07-28; OWASP ASI02 | ✅ Shipped 2026-09-07 (pre-parse per-tool/per-method allow/deny/escalate via `McpToolPolicy` on `TenantConfig.mcp`; most-restrictive-wins resolution; header/body mismatch = smuggling BLOCK; ESCALATE = refuse + audit, HITL callback is future work) |
| NG-9 | Provenance-lite egress binding (not a mini-CaMeL; fail-closed, opt-in) | P2 | ~1 wk | CaMeL 2503.18813 | ✅ Shipped 2026-09-07 (`neuralguard.mcp.provenance`: session taint windows keyed on `Mcp-Session-Id`, 4-word-shingle + long-token fingerprinting, tenant `egress_tools` classification, off/warn/block modes — off by default; pre-forward ordering guaranteed; runbook section in `docs/runbooks/mcp_gateway.md`) |

Ordering guidance: NG-1–NG-4 shipped first (small, gate-provable, defend against
attack classes that beat every tested open-weight guard). NG-5 shipped with the
calibrated baseline + a free detector fix it exposed (below). NG-6 shipped
next (2026-09-07) — the cheap honesty gate, published in `docs/FPR_SLO.md`.
cheap honesty gate), then NG-7 + NG-8 as the differentiator sprint. All
additions inherit the existing CI bar (ruff + format + mypy strict + pytest +
90% coverage floor + A1 gate + boot smoke). Calibration note (NG-4): the
windowed pass defaults (`θ_b=0.60`, decision 0.30, min_run 2) are calibrated
to the A2 corpus reality that benign prompts can match at 0.60-0.74; the gate
flags ESCALATE (judge-resolvable), never a silent pass — tune via
`NEURALGUARD_SCANNER_SEMANTIC_OVERFLOW_*` with A1/A2 measurement.

NG-5 honest-baseline note (2026-09-07): the 16-operator gate measures the
DETERMINISTIC layers (pattern-only, no ONNX/Ollama — CI-able, same contract
as A1). Day-one measurement: aggregate Unsafe-ASR **56.94%**, with the
de-mutation-class operators as documented gaps (dot_interleave /
word_reversal 100%; homoglyphs / leetspeak / char_duplication /
vowel_stretch / accent_homoglyph_mix 96.3%; emoji_interleave 70.4%;
token_splitting 48.2%; alternating_case 14.8%) and the normalizer-covered
operators at 0.00% (fullwidth, whitespace_swap, diacritics — the latter via
a NEW Latin combining-mark fold the gate itself exposed and shipped: NFKD
kept stray marks between ASCII letters, breaking every literal keyword
regex; the fold strips marks following ASCII letters in the detection copy
only and never touches load-bearing marks on non-ASCII bases, e.g.
Devanagari/Arabic). Safe-ASR **12.50%** = 90/720 evals, ALL from the two
invisible-char operators via the DESIGNED STRUCT-004 posture (strip +
sanitized delivery, zero hard BLOCKs — enforced as a gate pin). Cross-script
homoglyph folding is deliberately NOT shipped blanket-folded (it would
corrupt legitimate mixed-script text, e.g. Russian+English) — it feeds the
i18n/normalizer work item; the semantic layer (A2) is the designed second
line for de-mutation-class gaps. Gate constants: Unsafe-ASR ≤ baseline+1%,
Safe-ASR < 13%, zero BLOCKs — regressions fail the nightly bench job and the
pytest gate.

---

*Authored 2026-06-27 by Mackenzie 🔍. Status refreshed 2026-09-07 (v0.2.1
docs flush). This is a plan, not a promise — scope and ordering adjust with
evidence from each phase.*