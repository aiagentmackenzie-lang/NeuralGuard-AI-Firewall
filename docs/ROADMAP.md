# NeuralGuard Roadmap

> **Source of truth for what ships next.** This is a tracked (public) roadmap.
> The gitignored `PRODUCTION_HARDENING_PLAN.md` is the internal ledger of
> what already landed; this doc is what is **planned**.
>
> **Last updated:** 2026-06-28 · **Baseline:** `main` @ 489b94e — 🥇 S/91,
> 560 tests (CI-realistic; 579 with the semantic model), P0+P1 closed
> (production-ready for single-worker + Redis-backed multi-worker deploys),
> Sprint A complete (NG↔NS benchmark harness), Phase 3 Agent Guardian B1+B2
> shipped. The A2 semantic-FPR corroboration-gate fix is merged (218f3b9).

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

- **B3 — ASI06 dedicated rule + canary unstub.** ⏳ NOT STARTED (paused for API
  limits; resumable next session).
  - **Memory poisoning (ASI06)** is corpus-assisted only today (Agent Guardian
    catches the *accumulation* via AG-MEM-ACCUM-001). Add a dedicated
    single-turn heuristic for RAG/memory-write poisoning (injected "remember
    that..." / "from now on, when asked X, do Y" / "store this into the
    knowledge base") to the pattern scanner (`T-MEM` rules). Document residual
    risk honestly.
  - **Canary token verification** (`canary_leaked`, currently stubbed `false`):
    mint per-session canary tokens (HMAC of session_id + server secret), inject
    into the system prompt, detect them in `/v1/scan/output` as a system-prompt
    exfiltration signal.

- **B4 — Benchmark integration.** ⏳ NOT STARTED. Extend Sprint A's harness
  with NeuralStrike's `AgentPivot` (multi-agent lateral movement) and a
  multi-turn attack script (delayed-injection sequences). Measure Agent
  Guardian's ASR delta vs the B1-disabled baseline. Success: a measurable ASR
  reduction with `agent_guardian.enabled=true`.

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
- ASI06 dedicated rule + canary verification in `scan/output`
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