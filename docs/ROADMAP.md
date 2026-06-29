# NeuralGuard Roadmap

> **Source of truth for what ships next.** This is a tracked (public) roadmap.
> The gitignored `PRODUCTION_HARDENING_PLAN.md` is the internal ledger of
> what already landed; this doc is what is **planned**.
>
> **Last updated:** 2026-06-29 · **Baseline:** `main` @ `45a08d2` (B4
> merge) → B3 (`63ec379`) → pre-B3 (`72b2ff3`). **Sprint B (Phase 3
> Agent Guardian) COMPLETE:** B1+B2 (`489b94e`), B3 (`63ec379`),
> B4 (`45a08d2`). 661 passed / 1 skipped. B4 surfaced 2 real scanner
> coverage gaps documented in
> `benchmarks/ng_vs_ns/results/known_gaps.md` for follow-up commits.
> The A2 semantic-FPR corroboration-gate fix is merged (`218f3b9`).
> Branch `sprint-b/b4-gap-closure` closes both gaps (TBD merge).

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

- **B3 — ASI06 dedicated rule + canary unstub.** ✅ SHIPPED on branch
  `sprint-b/b3-canary-tmem` tip, awaiting merge into main (next session
  merge gate).
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

- **B4 — Benchmark integration.** ✅ SHIPPED on branch
  `sprint-b/b4-multiturn-harness` (off `main` @ `63ec379`), awaiting merge
  gate.
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

Post-moat. Not blocking; pick these up for specific customer/enterprise
requirements.

- **P1-2 — Per-tenant config.** `tenants/<id>.yaml` → per-tenant
  RPM/burst/scanner overrides. `TenantSettings` exists but is unwired.
- **P2-4 — JWT/OAuth2 + key rotation API.** Static API keys today; add
  short-lived JWT/OIDC + a rotation endpoint (Vault/SOPS integration).
- **P2-6 — Kubernetes artifacts.** Helm chart / manifests + HPA on the
  metrics.
- **P2-5 — SBOM/image signing (cosign).** SBOM is generated but not
  attested; image not signed.
- **P2 — Restore 90% CI coverage gate.** Regenerate the ONNX model in CI
  (needs the `semantic-export` extra / torch) so the semantic tests run
  and the full-suite coverage clears 90% honestly.
- **P2-7 — SIEM/alert routing.** Escalation webhook exists; add
  structured alerting on sustained BLOCK spikes to Splunk/ELK/Sentinel.

---

## Ordering and gates

1. **Sprint A** first (cheapest, produces the measurement harness + a
   portfolio differentiator immediately).
2. **Sprint B** second (uses Sprint A's harness to prove multi-turn
   detection; the real moat).
3. **Sprint C** as enterprise demand requires.

Every phase gates on the existing CI bar: `ruff + mypy + pytest + 86%
coverage + boot-smoke`. Sprint A adds a nightly bench gate; Sprint B
extends it. No phase is "done" until the gate is green on `main` and the
portfolio reference is re-scored.

---

*Authored 2026-06-27 by Mackenzie 🔍. This is a plan, not a promise —
scope and ordering adjust with evidence from each phase.*