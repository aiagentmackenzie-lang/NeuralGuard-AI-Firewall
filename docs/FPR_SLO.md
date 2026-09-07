# NG-6 — The Guarded FPR SLO (published contract)

*Shipped 2026-09-07. Part of the 2026-09-07 research-frontier register
(`docs/ROADMAP.md`, NG-6), motivated by KDD '26 FPR-weaponization work and
the NotInject line of research: benign prompts that look like injections
are the false positives no vendor publishes.*

## What we promise

NeuralGuard's semantic layer publishes a **measured, guarded false-positive
rate (FPR)** — and lets each tenant pick their own position on the
sensitivity dial that trades detection for FPR. No commercial LLM firewall
publishes this number; for us it is cheap to measure and it is the honest
way to talk about detection.

**Definition (guarded FPR).** Embed every probe in the two guard sets below,
take each probe's *worst* (maximum) cosine similarity against the loaded
attack corpus, and count a probe as a **false positive** when that worst
match reaches the BLOCK threshold (default **0.75** — the probe would be
blocked as an attack). Guarded FPR = blocked probes / total probes.

**Guard sets (all tracked in git, all ben­ign by construction):**

| Set | File | Probes | Purpose |
|---|---|---:|---|
| F12 benign corpus | `benchmarks/ng_vs_ns/benign_corpus.jsonl` | 45 | Plain everyday prompts (coding, weather, writing). |
| **NG-6 hard negatives** | `corpus/benign_hard_negatives.jsonl` | 50 | **Benign look-alikes**: incident reports quoting attacks, awareness-training material, defensive tooling requests, policy/compliance text, research discussion. NotInject-style — the prompts that defeat naive keyword filters. |

## Measured numbers (2026-09-07, rebuilt corpus: 6,503 vectors)

| Measure | Benign (45) | Hard negatives (50) | Combined (95) |
|---|---:|---:|---:|
| **Guarded FPR (would BLOCK)** | **0.00%** | **0.00%** | **0.00%** |
| Escalate share (ambiguous zone 0.60–0.74, judge-resolvable) | 4.44% (2) | 8.00% (4) | 6.32% (6) |
| Worst probe similarity | 0.747 | 0.696 | 0.747 |

Reading the numbers honestly:

- **0.00% guarded FPR** is enforced three ways at once: the rebuild's
  hard-negative guard *drops* any corpus vector that would BLOCK a guard
  probe (`scripts/rebuild_corpus_vectors.py`), the tracked
  `TestPublishedInvariant.test_guarded_fpr_is_zero_on_published_corpus`
  re-measures it per-PR in CI, and the runtime boot self-check re-measures
  it on whatever artifact actually ships.
- The **escalate share is not zero and should not be** — security work that
  quotes attacks (HGN-001/004/005/028) lands in the ambiguous zone by
  design. ESCALATE is judge-resolvable (`judge_resolves_escalate=true` +
  a measured-reliable judge dropped the equivalent A2 benign FPR to 0.00%),
  and fail-closed escalation for ambiguous security-team traffic is the
  posture we choose to sell. This is the number FPR-weaponization research
  says attackers will try to inflate; it is measured, published, and gated.
- These are **corpus-time measurements against fixed guard sets**, not a
  production claim: production traffic cannot be truthfully labeled at
  runtime. Guard sets grow with real feedback; the SLO is re-measured on
  every boot and every rebuild.

## The SLO knobs

| Setting (env) | Default | Meaning |
|---|---|---|
| `NEURALGUARD_SCANNER_SEMANTIC_FPR_SLO` | `2.0` (percent) | Maximum tolerated guarded FPR across both guard sets. Default 2.0% has real headroom over the measured 0.00% for corpus growth + embedding-model drift. |
| `NEURALGUARD_SCANNER_SEMANTIC_FPR_SLO_ENFORCE` | `false` | When `true`, startup **fails** if the measured guarded FPR exceeds the SLO. Opt-in on purpose: an SLO breach is a quality failure (more false positives than promised), not a safety failure — failing boot by default would trade availability for a number. Operators who promise the SLO contractually should enable it. |
| `NEURALGUARD_SCANNER_SEMANTIC_BENIGN_GUARD_PATH` | `benchmarks/ng_vs_ns/benign_corpus.jsonl` | Guard set 1. |
| `NEURALGUARD_SCANNER_SEMANTIC_HARD_NEGATIVES_PATH` | `corpus/benign_hard_negatives.jsonl` | Guard set 2. |

Boot behavior: when the semantic layer is registered, the lifespan runs the
measurement, logs `fpr_slo_measured` with the full breakdown, and stores the
report for `GET /v1/info` (`semantic_fpr_slo` field — auth-protected). A
missing guard file or empty corpus reports the metric as **unavailable
(`null`), never as zero**.

## The per-tenant sensitivity dial

A published SLO is only meaningful if a tenant can *choose* their trade-off.
Tenant config files (`tenants/` directory, Sprint C C1 model) accept:

```yaml
tenant_id: acme
scanners:
  semantic_block_threshold: 0.65   # omit (or null) = inherit the global 0.75
```

- **Lower** (down to `0.60`, the ESCALATE floor) → more sensitive: matches in
  `[0.60, threshold)` escalate to the judge instead of allowing. Higher
  detection recall, higher measured FPR for that tenant.
- **Higher** (up to `0.95`) → fewer false positives, weaker semantic BLOCKs
  (hybrid scoring and the judge still see the evidence).
- Bounded at `[0.60, 0.95]`: a threshold below the ESCALATE floor would map
  matches the corpus search does not even surface; one above `0.95` is
  decorative. The scanner **re-validates** the value at scan time (defense
  in depth against a bad tenant file) and falls back to the global
  threshold with a loud log on any violation.

The dial changes only the **verdict mapping** — the corpus search threshold
stays at the ESCALATE floor so the ambiguous zone is always surfaced for
hybrid scoring + the judge, exactly as the global pipeline behaves.

## Maintenance rules

1. **New benign look-alike?** Add it to `corpus/benign_hard_negatives.jsonl`
   with an `HGN-###` id and a category from the five documented ones. If the
   rebuild then drops vectors, that is the guard working — review the dropped
   vectors before accepting the loss.
2. **A bare quoted attack with no benign framing does not belong in the
   guard set** — that is the attack itself, and the pipeline's job is to
   catch it (escalate at minimum).
3. **Never relax the CI invariant test** (`test_guarded_fpr_is_zero_on_published_corpus`)
   without a documented SLO change and Raphael's sign-off.
4. Re-measure after any embedding-model or corpus change; update the tables
   above with the new date. This file is the source of truth for the
   published numbers.