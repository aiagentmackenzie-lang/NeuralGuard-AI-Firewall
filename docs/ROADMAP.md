# NeuralGuard Roadmap

> **Source of truth for what ships next.** This is a tracked (public) roadmap.
> The gitignored `PRODUCTION_HARDENING_PLAN.md` is the internal ledger of
> what already landed; this doc is what is **planned**.
>
> **Last updated:** 2026-06-29 · **Baseline:** `main` @ 72b2ff3 → merge-incoming
> `sprint-b/b3-canary-tmem` (tip) — 🥇 S/91, 658 tests (CI-realistic;
> 579 → 658 with the semantic model + B3 additions), P0+P1 closed
> (production-ready for single-worker + Redis-backed multi-worker deploys),
> Sprint A complete (NG↔NS benchmark harness), Phase 3 Agent Guardian
> B1+B2+B3 shipped (B3 = ASI06 dedicated T-MEM rules + canary token
> verification, awaiting merge into main). B4 (multi-turn benchmark
> integration) is the remaining Sprint B phase. The A2 semantic-FPR
> corroboration-gate fix is merged (218f3b9).

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

- **B4 — Benchmark integration.** ⏳ NOT STARTED — next phase after the B3
  merge lands on main. Extend Sprint A's harness with NeuralStrike's
  `AgentPivot` (multi-agent lateral movement, `exploit_delegation(agent_from,
  agent_to, malicious_instruction)` in
  `NeuralStrike/src/neuralstrike/modules/exploit/agent_pivot.py`) plus
  delayed-injection multi-turn sequences. Measure Agent Guardian's ASR delta
  vs the B1-disabled baseline. Success: a measurable ASR reduction with
  `agent_guardian.enabled=true`. Plan: `benchmarks/ng_vs_ns/multiturn_harness.py`
  (live, skip-guarded like A2 — needs NeuralStrike editable + local Ollama +
  the [semantic] extra) AND `tests/benchmarks/test_b4_multiturn.py` with a
  DETERMINISTIC multi-turn regression gate (CI-able, no model dependency) plus
  a live gate that skips in CI. Same-author caveat (attacker + defender = same
  repo) is reiterated in the harness README.

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