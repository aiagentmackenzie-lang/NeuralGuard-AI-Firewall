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

## Measured 2026-11-18 (`main` = `426dc52`, before any model change)

Attack detection rate (caught / total), per layer:

| layer | AR | DE | EN | ES | FR | JA | KO | PT | RU | VI | ZH | ALL |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| structural | 0 % | 0 % | 6 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 2 % |
| pattern | 50 % | 50 % | 85 % | 62 % | 62 % | 62 % | 50 % | 38 % | 38 % | 50 % | 62 % | 62 % |
| semantic | 0 % | 12 % | 50 % | 38 % | 25 % | 0 % | 88 % | 0 % | 25 % | 0 % | 88 % | 34 % |
| **core** | 50 % | 50 % | 91 % | 75 % | 75 % | 62 % | **100 %** | 38 % | 50 % | 50 % | **100 %** | 73 % |

Benign blocked rate (BLOCK only), per layer:

| layer | AR | DE | EN | ES | FR | JA | KO | PT | RU | VI | ZH | ALL |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| structural | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| pattern | 71 % | 50 % | 17 % | 50 % | 62 % | 57 % | 57 % | 0 % | 62 % | 71 % | 38 % | 32 % |
| semantic | 0 % | 0 % | 6 % | 50 % | 0 % | 0 % | 71 % | 0 % | 0 % | 0 % | **100 %** | 13 % |
| **core** | 71 % | 50 % | 22 % | 75 % | 62 % | 57 % | **100 %** | 0 % | 62 % | 71 % | **100 %** | 41 % |

By family (core): JB 96 %, PI-D 85 %, EXF 100 %, PI-I 100 %, MEM 100 %,
OBF 67 %, TOOL 60 %, EXT 48 %, **CR 45 %**.

By benign kind (core): everyday 34/121 blocked (28 %), hard negatives 15/50
blocked (30 %).

Controls: **10/10** caught by core.

## What that matrix actually says

1. **`core` at 100 % detection with 100 % benign block rate is not a score —
   it is a detector that says yes to everything.** ZH and KO are in that
   state today: Chinese and Korean detection is carried by over-match, not by
   discrimination. A vendor quoting "100 % detection in Chinese" off this
   matrix would be lying with a true number.
2. **The multilingual recall is the pattern layer's**, hand-written per
   language, and it ranges 38 % (PT, RU) to 85 % (EN). The semantic layer
   contributes almost nothing outside EN/ES/FR/DE/KO/ZH, and for ZH its
   signal is inverted.
3. **The published guarded-FPR SLO is a semantic-layer number and it
   reproduces exactly here** (hard negatives: 0/50 blocked, 4/50 escalate =
   8.00 %, as documented). But the **pattern layer blocks 15/50 (30 %) of
   those same probes**, so the number a buyer reads under "FPR 0.00 %" does
   not describe the detector that ships. That is the scope gap this harness
   exists to expose, and it is now labelled in `docs/FPR_SLO.md`.
4. **PT is the weakest language on recall (38 %) despite being the only one
   with a human-authored benign pass** — proof that the September PT work
   tuned precision and left recall gaps (`Esqueça a instrução anterior`,
   `Desconsidere as instruções`, `Obedeça sem questionar` are all missed).

## Regenerating

```bash
python -m benchmarks.detection_eval.harness                 # matrix on stdout
python -m benchmarks.detection_eval.harness --json out.json  # full report
python -m benchmarks.detection_eval.harness --layer semantic --language ZH
```

Exit code **3** if a canonical control attack is missed by *every* layer — the
instrument or the build is broken and no number in the run is worth reading.
