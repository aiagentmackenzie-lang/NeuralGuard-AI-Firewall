"""Benchmark multilingual embedding candidates for the semantic layer (P1).

The question this answers
-------------------------
`all-MiniLM-L6-v2` is English-trained and the semantic layer is the documented
"second line of defence" for ten languages. On 2026-11-18 the capability matrix
(`benchmarks/detection_eval/`) measured it as **blind** on canonical PT/JA/VI/AR
attacks and **inverted** on Chinese (benign text scores above attacks). This
script decides the replacement by measurement instead of by reputation.

Method
------
Apples to apples, one variable changed — the embedding model:

* the attack corpus is rebuilt from the *same tracked sources* the shipped
  vectors come from (`corpus/attack_corpus_full.jsonl` +
  `corpus/augment_checkpoint.jsonl`, replayed through the same dedup), so a
  candidate is not judged against a corpus it was not given;
* every eval item (multilingual attacks, 10-language benign, English attacks,
  F12 benign, NG-6 hard negatives) is embedded as a *query* with the model's own
  query prefix where it has one (E5 needs `query: ` / `passage: `; the
  paraphrase models and BGE-M3 do not);
* the decision variables are the ones the product already uses — the published
  escalate floor 0.60 and BLOCK threshold 0.75 — so the comparison is in the
  units of the shipped contract, not in an academic metric that nobody can act on;
* **the NG-6 benign guard is applied per model** (corpus vectors that would BLOCK
  one of the 95 published English guard probes are dropped, exactly as
  `scripts/rebuild_corpus_vectors.py` does at build time). Without it the measured
  numbers are meaningless: an unguarded 6,513-vector corpus matches *everything*,
  which the first smoke run of this script demonstrated by scoring 100 % caught
  and 100 % benign blocked — the same all-block failure mode the capability
  matrix found in Chinese.

**Known methodological limit, stated rather than hidden.** The guard probes and
the published 0.00 % SLO are the same 95 English prompts, which is the project's
existing design: the number is an operating-point guarantee by construction, not
a generalisation estimate. It is therefore reported here as **in-sample**, and
the model comparison is decided on the **held-out** multilingual benign set (76
probes, disjoint from the guard), with English attacks still counted.

Reported per model, per language: attacks caught (>= escalate floor), attacks
blocked (>= BLOCK), benign fired and benign blocked. Selection rule, stated
before the numbers are looked at:

    1. multilingual benign BLOCK rate must stay <= the incumbent's (0 % on the
       English guard sets, <= 4 % across the 10 languages);
    2. subject to (1), maximise the *worst* language's attack caught-rate.

Rule 2 is deliberately "worst language", not average: an average is how the
incumbent passed for a year while being blind in four languages.

Usage
-----
    python scripts/benchmark_embedding_models.py
    python scripts/benchmark_embedding_models.py --models intfloat/multilingual-e5-base
    python scripts/benchmark_embedding_models.py --json /tmp/emb_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.detection_eval.corpus import load_attacks, load_benign

ESCALATE_FLOOR = 0.60
BLOCK_THRESHOLD = 0.75
LANGUAGES = ("PT", "ES", "FR", "DE", "ZH", "JA", "KO", "RU", "AR", "VI", "EN")
BATCH = 64


@dataclass(frozen=True)
class Candidate:
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""


DEFAULT_CANDIDATES = (
    Candidate("sentence-transformers/all-MiniLM-L6-v2"),
    Candidate("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    Candidate("intfloat/multilingual-e5-small", query_prefix="query: ", passage_prefix="passage: "),
    Candidate("intfloat/multilingual-e5-base", query_prefix="query: ", passage_prefix="passage: "),
)


def load_corpus_texts() -> list[str]:
    """The exact texts the shipped 6,503-vector corpus is built from.

    Replays the dedup in ``scripts/rebuild_corpus_vectors.py`` (originals first,
    then checkpointed paraphrases in parent order, dropping duplicates) without
    reproducing its embedding or its benign guard — we are measuring what a
    model can separate, and the guard belongs to whichever model ships.
    """
    originals = [
        json.loads(line)["text"]
        for line in (REPO / "corpus" / "attack_corpus_full.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    checkpoint: dict[int, dict[str, Any]] = {}
    for line in (
        (REPO / "corpus" / "augment_checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if line.strip():
            row = json.loads(line)
            checkpoint[int(row["index"])] = row

    seen = {text.strip().lower() for text in originals}
    texts = list(originals)
    for parent_idx in sorted(checkpoint):
        for paraphrase in checkpoint[parent_idx].get("paraphrases", []):
            key = paraphrase.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(paraphrase)
    return texts


def embed_all(model: Any, texts: list[str], prefix: str, label: str) -> Any:
    import numpy as np

    out: list[Any] = []
    started = time.time()
    for i in range(0, len(texts), BATCH):
        chunk = [prefix + t for t in texts[i : i + BATCH]]
        out.append(
            model.encode(
                chunk, batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False
            )
        )
        if i and i % (BATCH * 20) == 0:
            done = i / len(texts)
            print(
                f"    {label}: {i}/{len(texts)} ({100 * done:.0f}%) "
                f"{time.time() - started:.0f}s elapsed",
                flush=True,
            )
    return np.vstack(out).astype(np.float32)


def _percentile_threshold(values: Any, target_share: float) -> float:
    """Threshold at which ``target_share`` of ``values`` are >= threshold."""
    import numpy as np

    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return 1.0
    k = round((1.0 - target_share) * arr.size)
    k = max(0, min(k, arr.size - 1))
    return float(arr[k])


def _rates(
    attacks: list[Any],
    eval_benign: list[Any],
    atk_sim: Any,
    ben_sim: Any,
    escalate: float,
    block: float,
) -> dict[str, Any]:
    per_lang: dict[str, Any] = {}
    for lang in LANGUAGES:
        atk_idx = [i for i, a in enumerate(attacks) if a.language == lang]
        ben_idx = [i for i, b in enumerate(eval_benign) if b.language == lang]
        caught = sum(1 for i in atk_idx if atk_sim[i] >= escalate)
        blocked_atk = sum(1 for i in atk_idx if atk_sim[i] >= block)
        fired = sum(1 for i in ben_idx if ben_sim[i] >= escalate)
        blocked = sum(1 for i in ben_idx if ben_sim[i] >= block)
        per_lang[lang] = {
            "attacks": len(atk_idx),
            "caught": caught,
            "blocked": blocked_atk,
            "caught_rate": round(caught / len(atk_idx), 4) if atk_idx else None,
            "benign": len(ben_idx),
            "benign_fired": fired,
            "benign_blocked": blocked,
            "benign_fired_rate": round(fired / len(ben_idx), 4) if ben_idx else None,
            "benign_block_rate": round(blocked / len(ben_idx), 4) if ben_idx else None,
            "atk_median_sim": round(float(statistics.median([atk_sim[i] for i in atk_idx])), 4)
            if atk_idx
            else None,
            "ben_median_sim": round(float(statistics.median([ben_sim[i] for i in ben_idx])), 4)
            if ben_idx
            else None,
        }

    multilingual = [lang for lang in LANGUAGES if lang != "EN"]
    worst = min(multilingual, key=lambda lang: per_lang[lang]["caught_rate"] or 0.0)
    total_atk = sum(per_lang[lang]["attacks"] for lang in LANGUAGES)
    total_caught = sum(per_lang[lang]["caught"] for lang in LANGUAGES)
    total_ben = sum(per_lang[lang]["benign"] for lang in LANGUAGES)
    total_ben_blocked = sum(per_lang[lang]["benign_blocked"] for lang in LANGUAGES)
    total_ben_fired = sum(per_lang[lang]["benign_fired"] for lang in LANGUAGES)
    return {
        "escalate_threshold": round(escalate, 4),
        "block_threshold": round(block, 4),
        "overall": {
            "attacks": total_atk,
            "caught": total_caught,
            "caught_rate": round(total_caught / total_atk, 4),
            "benign": total_ben,
            "benign_fired": total_ben_fired,
            "benign_blocked": total_ben_blocked,
            "benign_fired_rate": round(total_ben_fired / total_ben, 4) if total_ben else None,
            "benign_block_rate": round(total_ben_blocked / total_ben, 4) if total_ben else None,
            "worst_language": worst,
            "worst_language_caught_rate": per_lang[worst]["caught_rate"],
        },
        "by_language": per_lang,
    }


def evaluate_candidate(candidate: Candidate, corpus_texts: list[str]) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    print(f"\n=== {candidate.name} ===", flush=True)
    started = time.time()
    model = SentenceTransformer(candidate.name)

    print(f"  embedding corpus ({len(corpus_texts)} texts)...", flush=True)
    corpus = embed_all(model, corpus_texts, candidate.passage_prefix, "corpus")

    attacks = load_attacks()
    benign = load_benign()
    # Benign evaluation EXCLUDES the English guard probes: they drive the guard,
    # so scoring them as held-out evidence would be leakage (the project's own
    # in-sample convention is reported separately).
    eval_benign = [b for b in benign if b.language != "EN"]
    guard_texts = [b.prompt for b in benign if b.language == "EN"]

    print(
        f"  embedding {len(attacks)} attacks, {len(eval_benign)} benign, {len(guard_texts)} guard...",
        flush=True,
    )
    atk = embed_all(model, [a.prompt for a in attacks], candidate.query_prefix, "attacks")
    ben = embed_all(model, [b.prompt for b in eval_benign], candidate.query_prefix, "benign")
    guard = embed_all(model, guard_texts, candidate.query_prefix, "guard")

    raw_guard_sims = (guard @ corpus.T).max(axis=1)

    # ── The reason this benchmark needs two operating points ──────────────
    # 0.60 / 0.75 are MiniLM's thresholds. E5's similarity scale sits much
    # higher, so scoring E5 at MiniLM thresholds made *every* corpus vector look
    # like a BLOCK for a guard probe: the guard dropped 6,512 of 6,513 vectors
    # and the model then "caught" 100 % of everything against a single surviving
    # vector. That is the all-block failure mode, dressed as a win. Comparing
    # models therefore requires a matched operating point, not a shared constant.
    #
    #   A. shipped contract — the absolute 0.60 / 0.75 this product ships;
    #   B. matched cost — thresholds calibrated so this model reproduces the
    #      published escalate share (6.32 % of the guard set) and blocks none of
    #      it. Recall is then compared at equal user cost.
    matched_escalate = _percentile_threshold(raw_guard_sims, 6.0 / len(guard_texts))
    matched_block = float(max(raw_guard_sims.max() + 1e-6, matched_escalate + 1e-3))

    operating_points: dict[str, Any] = {}
    guard_report: dict[str, Any] = {}
    kept_by_op: dict[str, int] = {}
    for label, (escalate, block) in {
        "A_shipped_absolute": (ESCALATE_FLOOR, BLOCK_THRESHOLD),
        "B_matched_cost": (matched_escalate, matched_block),
    }.items():
        keep = (guard @ corpus.T < block).all(axis=0)
        kept = corpus[keep]
        kept_by_op[label] = int(keep.sum())
        atk_sim = (atk @ kept.T).max(axis=1)
        ben_sim = (ben @ kept.T).max(axis=1)
        guard_kept = (guard @ kept.T).max(axis=1)
        guard_report[label] = {
            "probes": len(guard_texts),
            "blocked": int((guard_kept >= block).sum()),
            "escalated": int((guard_kept >= escalate).sum()),
            "corpus_kept": int(keep.sum()),
            "corpus_dropped": int((~keep).sum()),
        }
        operating_points[label] = _rates(attacks, eval_benign, atk_sim, ben_sim, escalate, block)

    result = {
        "model": candidate.name,
        "query_prefix": candidate.query_prefix,
        "corpus_size": len(corpus_texts),
        "seconds": round(time.time() - started, 1),
        "guard_similarity": {
            "max": round(float(raw_guard_sims.max()), 4),
            "median": round(float(statistics.median(raw_guard_sims.tolist())), 4),
            "min": round(float(raw_guard_sims.min()), 4),
        },
        "guard": guard_report,
        "operating_points": operating_points,
    }

    for label, op in operating_points.items():
        ov = op["overall"]
        print(
            f"  [{label}] thr escalate={op['escalate_threshold']} block={op['block_threshold']} | "
            f"corpus kept {guard_report[label]['corpus_kept']}/{len(corpus_texts)} | "
            f"guard blocked {guard_report[label]['blocked']} escalated {guard_report[label]['escalated']} | "
            f"caught {ov['caught']}/{ov['attacks']} ({100 * ov['caught_rate']:.0f}%), "
            f"benign fired {ov['benign_fired']}/{ov['benign']}, blocked {ov['benign_blocked']}, "
            f"worst {ov['worst_language']} {100 * (ov['worst_language_caught_rate'] or 0):.0f}%",
            flush=True,
        )
    print(f"  [{result['seconds']}s]", flush=True)
    return result


def print_comparison(results: list[dict[str, Any]]) -> None:
    for label in ("A_shipped_absolute", "B_matched_cost"):
        print(f"\n\n=== ATTACK CAUGHT RATE — {label} ===")
        print("model".ljust(56) + "".join(lang.rjust(6) for lang in LANGUAGES) + "  WORST")
        for r in results:
            op = r["operating_points"][label]
            row = r["model"].ljust(56)
            for lang in LANGUAGES:
                rate = op["by_language"][lang]["caught_rate"]
                row += (f"{100 * rate:4.0f}%" if rate is not None else "    -").rjust(6)
            row += f"  {op['overall']['worst_language']} {100 * (op['overall']['worst_language_caught_rate'] or 0):.0f}%"
            print(row)

        print(f"\n=== BENIGN (76 multilingual, held out) — {label} ===")
        print("model".ljust(56) + "fired%".rjust(8) + "blocked%".rjust(10))
        for r in results:
            ov = r["operating_points"][label]["overall"]
            print(
                f"{r['model']:<56}{100 * (ov['benign_fired_rate'] or 0):7.1f}%"
                f"{100 * (ov['benign_block_rate'] or 0):9.1f}%"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models", action="append", help="model id (repeatable); default = 4 candidates"
    )
    ap.add_argument("--json", help="write full results here")
    ap.add_argument("--corpus-limit", type=int, help="truncate corpus (smoke runs only)")
    args = ap.parse_args(argv)

    candidates = tuple(Candidate(m) for m in args.models) if args.models else DEFAULT_CANDIDATES

    corpus_texts = load_corpus_texts()
    if args.corpus_limit:
        corpus_texts = corpus_texts[: args.corpus_limit]
    print(f"corpus texts: {len(corpus_texts)}")

    results = [evaluate_candidate(c, corpus_texts) for c in candidates]
    print_comparison(results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
