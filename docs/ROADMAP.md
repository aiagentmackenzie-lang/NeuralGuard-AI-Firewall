# NeuralGuard Roadmap

> **Plan-of-record for what ships next.** Shipped capability lives in the
> [README](../README.md); this document tracks what is *planned* and the
> evidence behind each decision. The internal working ledger that recorded
> how shipped items landed is machine-local and deliberately not part of
> the public repo.
>
> **Plan, not a promise:** scope and ordering adjust with evidence from
> each phase. **Last updated:** 2026-09-21.

---

## Shipped (summary)

All of the following are merged, CI-gated, and documented in the
[README](../README.md) — listed here so the open register below has context.

| Area | Status |
|---|---|
| Layered pipeline (structural → pattern → semantic ONNX → judge → output validation) | Shipped |
| Agent Guardian — multi-turn detection (delayed injection, role drift, gradual extraction/memory poisoning) | Shipped |
| Canary token mint/verify (HMAC-SHA256, no server-side token store) | Shipped |
| Prompt-template analyzer (static injection-sink analysis, CI-able) | Shipped |
| MCP gateway — per-tool intent gate + Ed25519-signed tool-catalog baselines (rug-pull refusal) | Shipped |
| Appliance proxy in front of any OpenAI-compatible upstream | Shipped |
| Per-tenant configuration (fail-safe overlay, hot-reload, mandatory core layers) | Shipped |
| JWT bearer auth + runtime key rotation | Shipped |
| Hash-chained audit (JSONL/Postgres) + Ed25519 event signing + chain verification | Shipped |
| SIEM routing (Splunk HEC / webhook / SecurityScarletAI) + block-spike alerting | Shipped |
| Kubernetes manifests + HPA (schema-validated; cluster drill pending — see open register) | Shipped, drill pending |
| SBOM + keyless cosign signing/attestation | Shipped |
| Reasoning-token output scan, decode-then-activate detection, overflow-resistant windowed aggregation | Shipped |
| Benchmark suite: deterministic regression gate, 16-operator mutation gate, live attacker benchmark, multi-turn harness | Shipped |
| Fleet deployment: co-resident SIEM + the NeuralStrike purple loop (live-fire receipts) | Shipped |

Shipped items carry their detailed status in the README's capability table
and the benchmark section; this roadmap does not duplicate them.

---

## Open register — what ships next

The current open items, in priority order. Nothing below is claimed as
done until it is merged, CI-gated, and documented.

| Item | What it is | Why it matters | Status |
|---|---|---|---|
| **SSE hold-back streaming** | Streaming requests are currently refused fail-closed (`422`) by design. Hold-back scanning would let streams start after the prefix is scanned | Real deployments increasingly stream; the refusal is honest but blocks adoption | Planned (demand-driven) |
| **Kubernetes cluster drill** | Manifests + HPA are schema-validated offline but have never been applied to a live cluster | Schema-valid ≠ applied; the drill is the evidence | Planned |
| **JWT residuals** | RS256/OIDC discovery (JWKS), refresh tokens, Vault/SOPS secret-manager integration | Enterprise auth completeness | Planned (demand-driven) |
| **Cross-worker audit ordering** | Per-worker chains are hash-authenticated; **global** write ordering needs a WORM sink or DB sequence | Full-order tamper evidence | Planned |
| **i18n native-speaker sign-off** | The rule packs had a machine self-audit (all 5 defects it found were fixed and re-verified); one language needs a human native-speaker pass (~10 min worksheet in [`i18n_native_review_request.md`](i18n_native_review_request.md)); the other languages stay deferred with an honest README residual | Machine self-audit is not human review — say so | Open (human-gated) |
| **Context-aware homoglyph folding** | The mutation-robustness gate measured the cross-script homoglyph gap (~96% for that class); blanket folding would corrupt legitimate mixed-script text, so the fix is deliberately context-aware | Closes the largest documented deterministic-layer gap | Planned (feeds the i18n/normalizer item) |
| **HITL callback for MCP ESCALATE** | The MCP gateway currently refuses + audits on ESCALATE; a human-in-the-loop callback would complete the loop | Completeness of the intent-gate story | Under consideration |

---

## Research evidence base

Planning decisions are evidence-anchored. The three results that most shape
the roadmap:

1. **Input detection has a provable ceiling.** No input filter running
   significantly faster than the model it protects can universally separate
   adversarial from benign prompts — Ball et al. (2025),
   [arXiv:2507.07341](https://arxiv.org/abs/2507.07341); made practical by
   controlled-release prompting at
   [USENIX Security 2026](https://www.usenix.org/system/files/usenixsecurity26-fairoze.pdf)
   (92–100% ASR against production models; 14 open-weight guards scored
   0.00–0.23 against those attacks).
2. **The guard itself is the attack surface.** Prompt Overflow fragments one
   instruction across an overlong prompt and beat every guard tested
   ([arXiv:2605.23196](https://arxiv.org/html/2605.23196)); mutation
   robustness and false-positive weaponization (KDD 2026) attack the
   detector, not the model.
3. **Agent security is moving to protocol + provenance enforcement.**
   [CaMeL](https://arxiv.org/abs/2503.18813) for control-flow integrity,
   the MCP specification's signed-tool-manifest direction, and the OWASP
   Agentic Top 10 all point the same way — which is why the MCP gateway,
   signed baselines, and egress taint-tracking shipped.

Full synthesis notes live in a machine-local research ledger; durable
conclusions graduate into this register with citations, as above.

---

## How decisions get made

- Every open item carries a why, an effort estimate, and an evidence anchor
  before it ships — speculation is labeled as such.
- Nothing is "done" until the CI bar is green on `main`: ruff + format +
  mypy strict + tests + the 90% coverage floor + the deterministic
  regression gate + boot smoke.
- Benchmarks publish per-attack-class results, never aggregates alone
  (benchmark aggregates do not predict deployment security — see the
  evidence base).
- This roadmap states boundaries as deliberately as features: the honest
  limits live in the [README threat model](../README.md#threat-model-what-input-filtering-can-and-cannot-do)
  and [SECURITY.md](SECURITY.md).

## Contributing to the roadmap

Open an issue or PR against this file. Proposals should name: the problem
evidence, the expected gate (how we would know it works), and the failure
mode it does **not** address. We would rather add an honest open item than
a shipped-sounding aspiration.

---

*This is a plan, not a promise — scope and ordering adjust with evidence
from each phase.*