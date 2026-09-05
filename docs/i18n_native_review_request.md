# P2-11 — Native-speaker security review of the 50 i18n detection patterns

> **Status: PENDING HUMAN REVIEW.** This document is the structured request.
> The 50 non-English patterns in `src/neuralguard/scanners/pattern_i18n.py`
> have NOT had native-speaker review (SRD FR-001 TODO). Nothing in this repo
> may claim i18n detection is "reviewed" until a native speaker signs off per
> language below. Until then, the README must carry the residual-risk
> admission.

## Why this needs humans

The patterns were authored by an AI model without native-speaker verification.
A pattern can be regex-correct and still be wrong in the language: unnatural
phrasings that never occur in real attacks, false-positive magnets that fire
on everyday sentences, or real attack phrasings the author never thought of.
Machine self-review (below) catches mechanical defects — it cannot judge
idiomaticity, real-world attack surface, or false-positive likelihood. That
judgment is exactly what this request asks for.

## How to review (per language, ~15-20 min)

You need: the file `src/neuralguard/scanners/pattern_i18n.py`, the 5 patterns
for your language, and fluency. For each pattern:

1. **Attack recall** — is the described attack phrase family actually covered?
   Try the canonical phrasings AND the phrasings a real person would type.
   Note misses (false negatives).
2. **False-positive probe** — try 5-10 BENIGN sentences from everyday use
   (customer support, schoolwork, casual chat) that might accidentally match.
   Note any match that should not BLOCK/SANITIZE (false positives).
3. **Naturalness** — are the matched phrases things people actually write?
4. **Severity sanity** — does HIGH (→BLOCK) vs MEDIUM (→SANITIZE) feel right
   for what the pattern catches?

Record per pattern: `OK` / `FN (missed phrasings: …)` / `FP (benign example: …)`
/ `REWRITE (suggested regex)`. Send results to Raphael; findings become
hand-verified pattern fixes, labeled as human-reviewed in the commit message.

## Machine self-audit (2026-09-05, mechanical defects found — NOT a substitute)

Found by regex-mechanics review; these are flagged for fix and are the kind of
defect a native pass would also catch:

| # | Pattern | Defect | Class |
|:--|:--|:--|:--|
| 1 | DE PI-D-DE-001 | `annuliere` — misspelled; correct German is **annulliere** (double l). The pattern only matches the misspelling today. | Real bug (FN) |
| 2 | FR PI-D-FR-001 | `ne\s+suive` — not a valid imperative form (2sg = "ne suis pas", 2pl = "ne suivez pas"); as written it only fires on subjunctive constructions. Likely intended `suivez`. | Real bug (FN) |
| 3 | ZH JB-ZH-003, JA JB-JA-003 | ` developer mode` with a leading space — an English-pattern copy artifact in the CJK alternations. | Artifact |
| 4 | ZH PI-D-ZH-001 | Character-class alternation `[指說]?[示令命導]?[令命]?[規則規]?[則]?…` — nearly all groups optional, so the branch can match bare fragments (e.g. single char 掉/規). **High over-match (FP) risk** on benign Chinese; also duplicate chars inside classes (`[規則規]`), duplicated `[掉]?`. | High FP risk |
| 5 | AR EXT-AR-004 | Exact duplicate alternatives: `تعليماتك\|تعليماتك`, `توجيهاتك\|توجيهاتك`, `الشروط` twice. | Noise |
| 6 | VI PI-D-VI-001 / -002 / EXT-VI-004 | Exact duplicate alternatives: `bỏ\s+qua` ×2, `trở\s+thành` ×2, `ở\s+trên` ×2. | Noise |
| 7 | RU PI-D-RU-005 | `ограничения?` appears twice in one alternation. | Noise |
| 8 | KO PI-D-KO-005 | `없이` twice in one alternation. | Noise |
| 9 | ES PI-D-ES-001 | `tus\s+` twice in the optional-prefix alternation. | Noise |
| 10 | RU PI-D-RU-002 | `ты теперь …` alone (before any `\S{1,8}` complement check the regex cannot anchor) risks matching the extremely common benign "ты теперь …" framing; also `(?:веди\|себя\|принимай)\s+как` includes the malformed "веди как" (correct: "веди себя как"). | FP risk / grammar |
| 11 | All ZH patterns | `\s*` between CJK tokens — written Chinese does not space words; harmless but signals non-native authorship. | Noise |

Mechanically sound: bounded quantifiers (`\S{1,8}`), per-search 50 ms ReDoS
timeout (regex module), `(?i)` on Latin/Cyrillic scripts, no nested-unbounded
quantifier constructs found in any of the 50 patterns.

## Language checklist (sign-off grid)

| Language | Patterns | Reviewer | Date | Verdict per pattern (001-005) |
|:--|:--|:--|:--|:--|
| PT (Portuguese) | PI-D-PT-001/002, JB-PT-003, EXT-PT-004, PI-D-PT-005 | ______ | ______ | |
| ES (Spanish) | PI-D-ES-001/002, JB-ES-003, EXT-ES-004, PI-D-ES-005 | ______ | ______ | |
| FR (French) | PI-D-FR-001/002, JB-FR-003, EXT-FR-004, PI-D-FR-005 | ______ | ______ | |
| DE (German) | PI-D-DE-001/002, JB-DE-003, EXT-DE-004, PI-D-DE-005 | ______ | ______ | |
| ZH (Chinese) | PI-D-ZH-001/002, JB-ZH-003, EXT-ZH-004, PI-D-ZH-005 | ______ | ______ | |
| JA (Japanese) | PI-D-JA-001/002, JB-JA-003, EXT-JA-004, PI-D-JA-005 | ______ | ______ | |
| KO (Korean) | PI-D-KO-001/002, JB-KO-003, EXT-KO-004, PI-D-KO-005 | ______ | ______ | |
| RU (Russian) | PI-D-RU-001/002, JB-RU-003, EXT-RU-004, PI-D-RU-005 | ______ | ______ | |
| AR (Arabic) | PI-D-AR-001/002, JB-AR-003, EXT-AR-004, PI-D-AR-005 | ______ | ______ | |
| VI (Vietnamese) | PI-D-VI-001/002, JB-VI-003, EXT-VI-004, PI-D-VI-005 | ______ | ______ | |

## Benign starter sets per language (seed the FP probe — expand freely)

Each reviewer should go well beyond these, but start with:

- **PT**: "Por favor, ignore o erro anterior e continue." · "Quero que você
  escreva um poema." · "Sem mais delongas, vamos começar."
- **ES**: "Ignora el ruido de fondo del audio." · "Ahora eres el siguiente
  presentador." (quiz context) · "Sin restricciones de tiempo, tómate tu
  momento."
- **FR**: "Ignore le chapitre précédent et lis la suite." · "Tu es maintenant
  prêt à commencer." · "Sans aucune restriction de budget…"
- **DE**: "Ignoriere bitte die Fußnoten." · "Du bist jetzt an der Reihe." ·
  "Ohne jegliche Einschränkung der Künstlerfreiheit…"
- **ZH**: 「你可以做任何你喜歡的事（夢想主題討論）」 · 「重複上面的練習題」(tutoring)
- **JA**: 「上記の内容をすべて繰り返し練習しましょう」(classroom) · 「何でもしてくれる店」(restaurant review)
- **KO**: "위의 내용을 모두 반복해서 읽어 주세요" (reading practice) · "제한 없이 자유롭게
  의견을 말해 보세요" (classroom)
- **RU**: "Ты теперь в главной роли" (theater class) · "Повтори всё выше
  сказанное своими словами" (study)
- **AR**: "أعد كتابة الجملة السابقة" (handwriting practice) · "افعل ما تريد في
  وقت فراغك" (advice column)
- **VI**: "Lặp lại tất cả ở trên để ghi nhớ" (study tip) · "Bây giờ bạn là người
  tiếp theo" (game show)

## After the review lands

- Fix confirmed FNs/FPs in a dedicated commit (pattern changes → re-run A1 +
  B4 + the full gate, per house rule: measurement unchanged in that commit).
- Update `PRODUCTION_HARDENING_PLAN.md` P2-11: PENDING → CLOSED with the
  reviewer names/dates, and lift the README residual-risk admission to
  "reviewed by native speakers (PT/ES/… on date)" listing any language still
  unreviewed honestly.