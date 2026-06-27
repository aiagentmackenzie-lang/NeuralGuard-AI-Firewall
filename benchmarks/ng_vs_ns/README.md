# NeuralGuard ↔ NeuralStrike benchmark (Sprint A)

A benchmark that measures NeuralGuard's detection efficacy against a
labeled attack corpus and a benign corpus. **Phase A1** (this directory)
is a deterministic **NeuralGuard regression suite**, independent of
NeuralStrike. Phases A2 (live NeuralStrike attacker) and A3 (publish +
nightly `bench.yml`) build on top of it.

> ## ⚠️ Same-author caveat — read first
>
> **The attacker (NeuralStrike) and the defender (NeuralGuard) are written
> by the same author (Mackenzie / Raphael).** This benchmark measures
> **defense-in-depth and regression detection**, **NOT** neutral
> third-party independence. A 0% ASR here means "NeuralGuard catches every
> attack its own offensive sibling produces at the deterministic layer" —
> it is a strong regression signal and a portfolio differentiator, but it
> is **not** a claim that NeuralGuard resists an external red team. The
> README and any published results table state this plainly.

---

## What A1 measures

| Metric | Definition | Target |
|:--|:--|:--|
| **ASR** | Attack Success Rate — fraction of attack prompts that NeuralGuard **allowed** (`verdict=allow`). | **0%** |
| **FPR** | False Positive Rate — fraction of benign prompts that did **not** receive `allow`. | **< 2%** |
| **Exact-match** | Fraction of attacks whose actual verdict equals the labeled `expected_verdict`. Reported for visibility; not a hard gate. | informational |

### Gate semantics (important)

The pytest gate (`tests/benchmarks/test_a1_regression_gate.py`) asserts
that **no attack returns a verdict *weaker* than its `expected_verdict`**.
A drift to a *stricter* verdict (e.g. an attack labeled `sanitize` that
NeuralGuard now returns `block` for) **is allowed** — defense-in-depth
upgrades must not break the regression gate. A drift to a *weaker* verdict
(`block` → `sanitize` → `allow`) is a regression and fails the gate.

This is why the gate uses verdict-priority ordering
(`block > sanitize > escalate > quarantine > rate_limit > allow`, mirroring
`src/neuralguard/scanners/pipeline.py`) rather than exact string equality.

---

## Config — the deterministic pattern-only baseline

The harness runs in-process against the ASGI app (`httpx.ASGITransport`,
no network, no port) with a deterministic config (`benchmark_config()` in
`harness.py`):

- **Layers:** structural + pattern only. Semantic and judge are **OFF**
  (the ONNX model is gitignored; A1 is the deterministic baseline).
- **Auth:** OFF (development environment).
- **Rate limiting:** OFF (we measure detection efficacy, not throttling).
- **Audit:** OFF (no disk writes during a benchmark).

This is the **`pattern-only`** configuration that A2 will later compare
against `pattern + semantic` and `pattern + semantic + judge`.

---

## Corpus schema

### `attack_corpus.jsonl` (one JSON object per line)

| Field | Type | Description |
|:--|:--|:--|
| `id` | string | Stable ID, e.g. `ATK-PI-D-001`. |
| `prompt` | string | The attack text sent to `/v1/evaluate`. |
| `expected_verdict` | `block` \| `sanitize` | The verdict NeuralGuard's pattern layer is expected to return (HIGH/CRITICAL → `block`, MEDIUM → `sanitize`). |
| `block_family` | string | NeuralGuard threat category code (`T-PI-D`, `T-PI-I`, `T-JB`, `T-EXT`, `T-EXF`, `T-TOOL`). |
| `neuralstrike_module` | string | The NeuralStrike module the attack represents (`JailbreakForge`, `ContextPoison`, `MCPInterceptor`, `ModelExtract`, `DataExfiltrator`). |
| `expected_rule_ids` | string[] | Pattern rule IDs expected to fire (diagnostic; not hard-asserted). |
| `severity` | `high` \| `medium` \| `critical` | Expected severity of the firing rule. |
| `notes` | string | Free-text rationale / mapping note. |

### `benign_corpus.jsonl`

| Field | Type | Description |
|:--|:--|:--|
| `id` | string | Stable ID, e.g. `BEN-001`. |
| `prompt` | string | A benign prompt that should be allowed. |
| `expected_verdict` | `allow` | Always `allow`. |
| `category` | string | Coarse tag (`coding`, `writing`, `translation`, `general`, `debugging`). |

---

## How to run

### As the CI gate (pytest)

```bash
uv run pytest tests/benchmarks/ -v
```

This is wired into the `test` job of `.github/workflows/ci.yml` as the
**"Run benchmark regression gate (A1)"** step — a per-PR gate, not the
nightly `bench.yml` (that is A3).

### As a CLI (local reproduction / A3 nightly reuse)

```bash
# in-process, deterministic baseline:
uv run python -m benchmarks.ng_vs_ns.harness

# against a live deployment (used by A2/A3):
uv run python -m benchmarks.ng_vs_ns.harness --base-url http://localhost:8000
```

The CLI prints the ASR / FPR / exact-match summary, any ASR regressions,
verdict drifts, and benign false positives, then exits non-zero if
`ASR > 0` or `FPR >= 2%`.

### As a library (A2/A3 extension point)

```python
from benchmarks.ng_vs_ns.harness import run

results = await run()
assert results.asr == 0.0
assert results.fpr < 0.02
```

Pass `client=AsyncClient(base_url=...)` to target a live deployment with a
non-default config (A2's `pattern + semantic` / `pattern + semantic + judge`
configurations).

---

## Coverage and honest non-goals (A1)

A1 covers the **detection families that map to a real NeuralStrike
module** and that the **pattern scanner detects deterministically**:

| NeuralStrike module | NeuralGuard family | Rules exercised |
|:--|:--|:--|
| JailbreakForge | T-PI-D, T-JB | PI-D-001..008, JB-001..007, JB-012 |
| ContextPoison | T-PI-I, T-JB (persistence) | PI-I-001/002/005, JB-010 |
| MCPInterceptor | T-TOOL | TOOL-001/004/005 |
| ModelExtract | T-EXT, T-PI-D (output manipulation) | EXT-001/002/003, PI-D-008 |
| DataExfiltrator | T-EXF | EXF-003/005/006/007/009 |

**Intentionally NOT in A1:**

- **`T-DOS` (reasoning DoS / cost abuse)** — no NeuralStrike module maps
  to it cleanly (NeuralStrike has no DoS/cost-abuse module). DOS detection
  is covered by the unit test suite (`tests/unit/test_pattern_scanner.py`)
  and will be paired with a NeuralStrike DoS module when one exists.
- **`T-ENC` (encoding evasion)** — ENC patterns are supplementary and
  lower-confidence (the unit tests assert only that the verdict is *some*
  value in `allow/sanitize/block`, not a specific one). Including flaky
  entries would make the gate non-deterministic. EvasionSuite entries will
  land in A2 alongside live NeuralStrike `EvasionSuite` runs.
- **NeuralStrike modules not represented as single prompts** — `LLMRecon`,
  `FunctionHijack`, `AgentPivot`, `AgentC2` are protocol-level or
  multi-turn. They belong in **Sprint B's B4** (multi-turn benchmark
  extension with `AgentPivot`), not a single-prompt regression corpus.

### Non-goals carried over from the roadmap

- **Not a neutral third-party benchmark** — same author (see caveat above).
- **No human red-team verdicts** — ground truth is the pattern scanner's
  own rule labels, cross-checked against the verdict the pipeline actually
  returns.
- **No multi-turn coverage** until Sprint B lands.

---

## Current baseline

```
Attacks : 27  | ASR = 0.00%  | exact-match = 100.00%
Benign  : 45  | FPR = 0.00%  (threshold < 2.00%)
PASS    : True
```

Reproduce with `uv run python -m benchmarks.ng_vs_ns.harness`.