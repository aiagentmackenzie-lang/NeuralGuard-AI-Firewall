"""Detection-capability evaluation — per language, per layer, per family (P0).

What this answers
-----------------
Not "is the firewall good?" — but *where* it is good, measured one layer at a
time, one language at a time. Every previous number this project published was
either English-only (`docs/FPR_SLO.md`: 95 probes) or aggregated across
languages, which hides a detector that is 98 % accurate in English and blind in
Vietnamese. This harness refuses to aggregate: it reports the matrix.

What it measures
----------------
For each **layer in isolation** (structural, pattern, semantic) and for the
deterministic core they form together:

* **caught** — attacks whose verdict is not ``allow`` (block / sanitize /
  escalate all count as caught, which is the same definition A1 uses);
* **fired** — benign items whose verdict is not ``allow`` (a false positive);
* both broken down by language, by attack family, and by benign kind
  (``everyday`` vs ``hard_negative``, kept apart because hard negatives are
  adversarially benign and answer a different question).

Deliberately excluded
---------------------
* **The LLM judge** — it needs an upstream model, which makes the numbers
  non-deterministic and non-CI-able. It is reported as unevaluated rather than
  quietly counted as a win.
* **Rate limiting, auth, audit** — policy layers, not detection. They are
  drilled elsewhere.

Reading the output honestly
---------------------------
The committed corpora are **machine-authored and not human-reviewed**
(see ``corpus.py``). They are a *baseline*: reproducible, versioned, extensible,
and honest about their provenance. They are not an authority, and no README
claim should be upgraded to "verified" on their strength alone.

Usage
-----
    python -m benchmarks.detection_eval.harness
    python -m benchmarks.detection_eval.harness --json detection_eval.json
    python -m benchmarks.detection_eval.harness --layer semantic --language ZH

Exit codes
----------
0  report produced
3  a control attack was missed by *every* layer — the instrument or the build is
   broken and no number below is worth reading
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.detection_eval.corpus import (
    Attack,
    Benign,
    load_attacks,
    load_benign,
    load_controls,
)
from neuralguard.config.settings import ScannerSettings
from neuralguard.models.schemas import EvaluateRequest, Verdict
from neuralguard.scanners.pattern import PatternScanner
from neuralguard.scanners.structural import StructuralScanner
from neuralguard.semantic.similarity import SimilarityScanner

# Same strictness ordering the pipeline uses, so "combined" means what the
# product means by combined.
_PRIORITY: dict[Verdict, int] = {
    Verdict.BLOCK: 6,
    Verdict.SANITIZE: 5,
    Verdict.ESCALATE: 4,
    Verdict.QUARANTINE: 3,
    Verdict.RATE_LIMIT: 2,
    Verdict.ALLOW: 0,
}


@dataclass
class LayerStats:
    """Counts for one layer over one slice of the corpus.

    Two distinct numbers, because they answer different questions and this
    project has already been bitten by blending them:

    * ``fired``   — verdict is not ALLOW. The *product* cost: blocked, or
      handed to the judge (ESCALATE) which is latency and spend, and a judge
      outage turns it into a refusal.
    * ``blocked`` — verdict is BLOCK (or stricter). The *published SLO*
      definition: ``docs/FPR_SLO.md`` counts a benign probe as a false
      positive only when it reaches the BLOCK threshold, and documents the
      escalate share separately (8 % on its hard negatives). Reporting one
      number as if it were the other is how an honest SLO becomes a
      misleading headline.
    """

    total: int = 0
    fired: int = 0
    blocked: int = 0

    @property
    def rate(self) -> float:
        return self.fired / self.total if self.total else 0.0

    @property
    def block_rate(self) -> float:
        return self.blocked / self.total if self.total else 0.0

    def merge(self, other: LayerStats) -> None:
        self.total += other.total
        self.fired += other.fired
        self.blocked += other.blocked


@dataclass
class Report:
    layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    controls: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"corpus": self.corpus, "layers": self.layers, "controls": self.controls}


class LayerRunner:
    """One scanner, isolated, so a number can be attributed to a layer."""

    def __init__(self, layer: str) -> None:
        self.layer = layer
        settings = ScannerSettings()
        if layer == "structural":
            self._scanners = [StructuralScanner(settings)]
        elif layer == "pattern":
            self._scanners = [PatternScanner(settings)]
        elif layer == "semantic":
            self._scanners = [SimilarityScanner(ScannerSettings(semantic_enabled=True))]
        elif layer == "core":
            self._scanners = [
                StructuralScanner(settings),
                PatternScanner(settings),
                SimilarityScanner(ScannerSettings(semantic_enabled=True)),
            ]
        else:  # pragma: no cover - argparse restricts this
            raise ValueError(f"unknown layer {layer!r}")
        self._request = EvaluateRequest

    def verdict(self, prompt: str) -> Verdict:
        worst = Verdict.ALLOW
        for scanner in self._scanners:
            result = scanner.safe_scan(self._request(prompt=prompt))
            if _PRIORITY[result.verdict] > _PRIORITY[worst]:
                worst = result.verdict
        return worst


LAYERS: tuple[str, ...] = ("structural", "pattern", "semantic", "core")


def _stats_by(
    runner: LayerRunner, items: list[Attack] | list[Benign], key: str
) -> dict[str, LayerStats]:
    out: dict[str, LayerStats] = defaultdict(LayerStats)
    for item in items:
        verdict = runner.verdict(item.prompt)
        bucket = str(getattr(item, key))
        out[bucket].total += 1
        if verdict is not Verdict.ALLOW:
            out[bucket].fired += 1
        if _PRIORITY[verdict] >= _PRIORITY[Verdict.BLOCK]:
            out[bucket].blocked += 1
    return dict(out)


def evaluate(layers: tuple[str, ...] = LAYERS) -> Report:
    attacks = load_attacks()
    benign = load_benign()
    controls = load_controls()
    report = Report()

    report.corpus = {
        "attacks": len(attacks),
        "attacks_by_language": len({a.language for a in attacks}),
        "benign": len(benign),
        "benign_everyday": sum(1 for b in benign if b.kind == "everyday"),
        "benign_hard_negative": sum(1 for b in benign if b.kind == "hard_negative"),
    }

    for layer in layers:
        runner = LayerRunner(layer)
        by_language = _stats_by(runner, attacks, "language")
        by_family = _stats_by(runner, attacks, "family")
        benign_lang = _stats_by(runner, benign, "language")
        benign_kind = _stats_by(runner, benign, "kind")

        all_attacks = LayerStats()
        for stats in by_language.values():
            all_attacks.merge(stats)
        all_benign = LayerStats()
        for stats in benign_lang.values():
            all_benign.merge(stats)

        report.layers[layer] = {
            "attacks": {
                "total": all_attacks.total,
                "caught": all_attacks.fired,
                "rate": round(all_attacks.rate, 4),
                "by_language": {
                    lang: {
                        "total": s.total,
                        "caught": s.fired,
                        "rate": round(s.rate, 4),
                        "blocked": s.blocked,
                        "block_rate": round(s.block_rate, 4),
                    }
                    for lang, s in sorted(by_language.items())
                },
                "by_family": {
                    fam: {"total": s.total, "caught": s.fired, "rate": round(s.rate, 4)}
                    for fam, s in sorted(by_family.items())
                },
            },
            "benign": {
                "total": all_benign.total,
                "fired": all_benign.fired,
                "rate": round(all_benign.rate, 4),
                "blocked": all_benign.blocked,
                "block_rate": round(all_benign.block_rate, 4),
                "by_language": {
                    lang: {
                        "total": s.total,
                        "fired": s.fired,
                        "rate": round(s.rate, 4),
                        "blocked": s.blocked,
                        "block_rate": round(s.block_rate, 4),
                    }
                    for lang, s in sorted(benign_lang.items())
                },
                "by_kind": {
                    kind: {
                        "total": s.total,
                        "fired": s.fired,
                        "rate": round(s.rate, 4),
                        "blocked": s.blocked,
                        "block_rate": round(s.block_rate, 4),
                    }
                    for kind, s in sorted(benign_kind.items())
                },
            },
        }

        if layer == "core":
            missed = [c.id for c in controls if runner.verdict(c.prompt) is Verdict.ALLOW]
            report.controls = {
                "total": len(controls),
                "missed_by_core": missed,
                "missed_by_language": sorted({m.split("-")[1] for m in missed if "-" in m}),
            }

    return report


def _print_matrix(report: Report) -> None:
    langs = sorted(
        {lang for layer in report.layers.values() for lang in layer["attacks"]["by_language"]}
        | {lang for layer in report.layers.values() for lang in layer["benign"]["by_language"]}
    )

    print("=== ATTACK DETECTION RATE, per layer × language ===")
    header = "layer".ljust(11) + "".join(lang.rjust(7) for lang in langs) + "   ALL"
    print(header)
    for name, data in report.layers.items():
        row = name.ljust(11)
        for lang in langs:
            stats = data["attacks"]["by_language"].get(lang)
            row += (f"{100 * stats['rate']:5.0f}%" if stats else "     -").rjust(7)
        row += f"{100 * data['attacks']['rate']:6.0f}%"
        print(row)

    print("\n=== BENIGN FALSE-POSITIVE RATE, per layer × language ===")
    print(header)
    for name, data in report.layers.items():
        row = name.ljust(11)
        for lang in langs:
            stats = data["benign"]["by_language"].get(lang)
            row += (f"{100 * stats['rate']:5.0f}%" if stats else "     -").rjust(7)
        row += f"{100 * data['benign']['rate']:6.0f}%"
        print(row)

    print("\n=== CORE, by attack family ===")
    for fam, stats in report.layers["core"]["attacks"]["by_family"].items():
        print(f"  {fam:<6} {stats['caught']:>3}/{stats['total']:<3} {100 * stats['rate']:5.1f}%")

    print("\n=== CORE, benign by kind (fired = not allow, blocked = BLOCK only) ===")
    for kind, stats in report.layers["core"]["benign"]["by_kind"].items():
        print(
            f"  {kind:<14} fired {stats['fired']:>3}/{stats['total']:<3} {100 * stats['rate']:5.1f}%"
            f" | blocked {stats['blocked']:>3}/{stats['total']:<3} {100 * stats['block_rate']:5.1f}%"
        )

    controls = report.controls
    print(
        f"\ncontrols (the instrument's own calibration): "
        f"{controls['total'] - len(controls['missed_by_core'])}/{controls['total']} caught by core"
    )
    if controls["missed_by_core"]:
        print(
            "  missed: "
            + ", ".join(controls["missed_by_core"])
            + " — a canonical attack missed by every pattern must be explained, not tolerated"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the full report to this path")
    ap.add_argument("--layer", action="append", choices=LAYERS, help="limit to these layers")
    args = ap.parse_args(argv)

    report = evaluate(tuple(args.layer) if args.layer else LAYERS)
    _print_matrix(report)

    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nreport written: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
