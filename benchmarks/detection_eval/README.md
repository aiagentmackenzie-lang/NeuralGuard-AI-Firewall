# Detection-capability evaluation (P0)

*Introduced 2026-11-18. This is the instrument that answers "where is the
detector good?" — per layer, per language, per attack family. It does not
answer "is it good?", and no document should pretend it does.*

## Why it exists

Every detection number this project published before today was either
English-only (`docs/FPR_SLO.md`: 95 probes, all English) or aggregated across
languages. Aggregation is how a detector that is 91 % accurate in English and
blind in Vietnamese still prints a respectable single figure. The 2026-11-18
i18n sweep made the gap visible; this harness makes it measurable and
repeatable.

## What it measures

For each layer in isolation — `structural`, `pattern`, `semantic` — and for
`core` (the three together, arbitrated by the same strictness order the
pipeline uses):

| Number | Definition | Why it is that definition |
|:--|:--|:--|
| **caught** | attack whose verdict is *not* `allow` | identical to the A1 gate's definition of "not an attack success" — block/sanitize/escalate all count |
| **fired** | benign item whose verdict is *not* `allow` | the product cost: blocked, or sent to the judge (latency + spend, and a judge outage turns it into a refusal) |
| **blocked** | benign item that reaches `BLOCK` | the **published SLO** definition (`docs/FPR_SLO.md` counts a probe only when it would block, and reports the escalate share separately) |

Both benign numbers are reported side by side because conflating them is how a
narrow, honest SLO becomes a misleading headline. They are not interchangeable
and this file will not use one as if it were the other.

Benign items are split by kind — `everyday` vs `hard_negative` — because
adversarially benign prompts (incident reports *quoting* attacks, awareness
training, defensive tooling requests) answer a different question from "does it
block the weather question".

## What it deliberately does not measure

- **The LLM judge.** It needs an upstream model: non-deterministic, not
  CI-able. Reported as unevaluated rather than counted as a win.
- **Rate limiting, auth, audit.** Policy layers, drilled elsewhere.
- **Multi-turn.** `tests/benchmarks/test_b4_multiturn.py` covers that.

## The corpora, and how much authority they have

| File | Items | Provenance |
|:--|--:|:--|
| `corpus_multilingual_attacks.jsonl` | 87 | **Machine-authored 2026-11-18**, 10 languages × 4 families (PI-D, JB, EXT, CR) + 7 language-neutral agentic/obfuscation items. **Not human-reviewed.** |
| `benchmarks/ng_vs_ns/attack_corpus.jsonl` | 27 | The existing English A1 regression corpus (reused, not duplicated) |
| `corpus_multilingual_benign.jsonl` | 76 | Machine-authored everyday sentences, 10 languages (imported from the i18n sweep so there is one copy) |
| `benchmarks/ng_vs_ns/benign_corpus.jsonl` | 45 | The F12 English benign corpus behind the published SLO |
| `corpus/benign_hard_negatives.jsonl` | 50 | NG-6 English hard negatives behind the published SLO |
| `corpus_controls.jsonl` | 10 | Canonical attacks, one per language — the instrument's own calibration |

**Authority: directional, not authoritative.** The multilingual items were
written by an AI, so the recall figures describe *that corpus*, and a human or
red-team-authored expansion is the next credibility step, not an optional one.
What these corpora *can* do is be reproducibly wrong in public, which is
strictly better than being vaguely right in private.

## Baseline and ratchet

`baseline.json` is the committed measurement of the build that introduced the
harness (2026-11-18, `main` = `426dc52`). `tests/benchmarks/test_detection_eval.py`
fails when a layer's overall caught-rate drops or its benign block-rate rises
beyond a 1-point tolerance against that baseline — a ratchet, so the numbers
cannot silently rot in the flattering direction. **Raising the baseline is a
commit with the new numbers in the message**, which is the point.

The same test enforces the cross-check with the published SLO: semantic-only
`blocked` on the 50 hard negatives must stay **0**. If it ever is not, one of
the two documents is lying and the build stops.

## Measured 2026-11-19 (`main` = `426dc52` + the multilingual semantic layer)

**Before** = all-MiniLM-L6-v2 (384-dim, English-trained) at its own thresholds
0.60/0.75. **After** = multilingual-e5-base (768-dim, ONNX fp32) at thresholds
calibrated to the *same published guard escalate share* (6.3 %), block 0.88.

Attack detection rate (caught / total), semantic layer, per language:

| model | AR | DE | EN | ES | FR | JA | KO | PT | RU | VI | ZH | ALL |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 | 0 % | 12 % | 50 % | 38 % | 25 % | 0 % | 88 % | 0 % | 25 % | 0 % | 88 % | **34 %** |
| multilingual-e5-base | 50 % | 100 % | 47 % | 88 % | 75 % | 38 % | 38 % | 50 % | 50 % | 62 % | 100 % | **60 %** |

Benign fired (not `allow`) on the 76 held-out multilingual probes:

| model | all | everyday | hard negatives | blocked (BLOCK only) |
|:--|--:|--:|--:|--:|
| all-MiniLM-L6-v2 | 22 % | — | — | 4.0 % |
| multilingual-e5-base | **10 %** | 12/121 | 5/50 | **1.2 %** |

Whole detector (`core` = structural + pattern + semantic):

| model | attacks caught | benign fired | benign blocked | NG-6 guard blocked | guard escalate |
|:--|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 | 73 % | 41 % | 41 % | 0/95 | 6.32 % |
| multilingual-e5-base | **85 %** | 37 % | 28.6 % | 0/95 | 6.3 % |

By family (core, E5): JB 96 %, PI-D 88 %, EXF 100 %, PI-I 100 %, MEM 100 %,
TOOL 60 %, OBF 67 %, EXT 43 %, CR 45 %. Controls: 10/10.

### What changed, and what did not

- **The multilingual column is no longer carried by regex alone.** Semantic
  recall went 34 % → 60 % on the same items at the same published escalate
  share, and benign firing *halved*. Ten of eleven languages improved; the one
  that did not is English semantic recall (50 % → 47 %), where the pattern layer
  already covers the corpus and E5's longer context dilutes instead of sharpening.
- **Chinese is still the weak language.** 100 % of ZH attacks are now caught,
  but 50 % of benign ZH still fires and 25 % is BLOCKed. The inversion is gone
  (benign no longer scores above attacks); the discrimination is not yet good.
  That is the honest state, and it is a native-speaker question, not a threshold.
- **The hard-negative class was untouched by the model swap** — 15/50 still
  block at the *pattern* layer, because that layer matches quoted attacks with no
  notion of quotation. 15.79 % remains the number to beat, and it is not a
  model problem.
- **Cost, measured:** ONNX median single-prompt embed 5.1 ms → 36.1 ms (fp32),
  artifact 1.11 GB. int8 quantization was measured and **rejected**: at equal
  guard precision it lost ~15 points of recall (44/114 vs 61/114 at 4.2 %
  guard escalate) because the model's usable margin on this corpus is ~0.02 and
  quantization noise is the same order. Latency budget (100 ms) and the
  per-test timeout moved with the measurement rather than the measurement being
  ignored.

## Regenerating

```bash
python -m benchmarks.detection_eval.harness                 # matrix on stdout
python -m benchmarks.detection_eval.harness --json out.json  # full report
python -m benchmarks.detection_eval.harness --layer semantic --language ZH
```

Exit code **3** if a canonical control attack is missed by *every* layer — the
instrument or the build is broken and no number in the run is worth reading.
