"""Gates for the detection-capability evaluation (P0).

Three jobs:

1. **Corpus integrity** — every language the product claims is represented, on
   both sides, with unique ids. A matrix with a missing column is how a blind
   language hides.
2. **Ratchet against `benchmarks/detection_eval/baseline.json`** — a layer's
   overall caught-rate may not fall and its benign block-rate may not rise
   beyond tolerance. Raising the baseline is a commit with the numbers in the
   message; that is the entire point of committing it.
3. **Cross-check with the published SLO** — `docs/FPR_SLO.md` promises a
   *guarded* (BLOCK-only) FPR of 0.00 % for the semantic layer on the NG-6 hard
   negatives. This harness measures that independently. If the two ever
   disagree, one of the documents is lying and the build stops here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.detection_eval.corpus import LANGUAGES, load_attacks, load_benign, load_controls
from benchmarks.detection_eval.harness import evaluate

_BASELINE = Path("benchmarks/detection_eval/baseline.json")
_TOLERANCE = 0.01  # rates are deterministic; the tolerance is for rounding only

# The matrix embeds 300+ items against a 6,503-vector corpus. The project-wide
# 30 s per-test timeout was set when the semantic layer was a 384-dim English
# model running in well under a second; on a CI runner this suite takes longer,
# and a timeout is not a detection result. Measured on the runner before
# raising: 3 x Timeout (>30.0s) with the suite otherwise green.
pytestmark = pytest.mark.timeout(900)

requires_model = pytest.mark.skipif(
    not (Path("models/attack_vectors.npy").exists() and Path("models/embedding-onnx").exists()),
    reason="rebuilt corpus + ONNX model not present (CI order: rebuild first)",
)


@pytest.fixture(scope="module")
def report() -> dict:
    return evaluate().to_dict()


class TestCorpusIntegrity:
    def test_every_claimed_language_has_attacks_and_benign(self) -> None:
        attacks = load_attacks()
        benign = load_benign()
        for lang in LANGUAGES:
            n_atk = sum(1 for a in attacks if a.language == lang)
            n_ben = sum(1 for b in benign if b.language == lang)
            assert n_atk >= 7, f"{lang} has {n_atk} attack items; the matrix needs >= 7"
            assert n_ben >= 7, f"{lang} has {n_ben} benign items; the matrix needs >= 7"

    def test_ids_are_unique(self) -> None:
        ids = [a.id for a in load_attacks()] + [b.id for b in load_benign()]
        assert len(ids) == len(set(ids)), "duplicate corpus ids make a failure unattributable"

    def test_controls_are_one_per_language(self) -> None:
        controls = load_controls()
        assert len(controls) == 10
        assert {c.language for c in controls} == set(LANGUAGES) - {"EN"}


@requires_model
class TestRatchet:
    def test_no_layer_regressed_against_the_committed_baseline(self, report: dict) -> None:
        assert _BASELINE.exists(), (
            "baseline.json is missing; regenerate with "
            "`python -m benchmarks.detection_eval.harness --json benchmarks/detection_eval/baseline.json`"
        )
        baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))

        for layer, data in report["layers"].items():
            base = baseline["layers"][layer]
            caught, base_caught = data["attacks"]["rate"], base["attacks"]["rate"]
            assert caught >= base_caught - _TOLERANCE, (
                f"{layer}: attack caught-rate fell {base_caught:.4f} -> {caught:.4f}. "
                "A recall regression is a product regression: restore it or record why "
                "in the baseline commit."
            )
            blocked, base_blocked = data["benign"]["block_rate"], base["benign"]["block_rate"]
            assert blocked <= base_blocked + _TOLERANCE, (
                f"{layer}: benign blocked-rate rose {base_blocked:.4f} -> {blocked:.4f}. "
                "Raise the baseline only with the new numbers in the message."
            )

    def test_published_slo_still_reproduces(self, report: dict) -> None:
        """docs/FPR_SLO.md promises 0.00% guarded FPR on the NG-6 hard negatives."""
        semantic = report["layers"]["semantic"]["benign"]["by_kind"]["hard_negative"]
        assert semantic["blocked"] == 0, (
            f"the semantic layer now BLOCKs {semantic['blocked']}/{semantic['total']} published "
            "hard negatives; docs/FPR_SLO.md claims 0.00%. One of them is wrong."
        )

    def test_controls_are_caught_by_core(self, report: dict) -> None:
        missed = report["controls"]["missed_by_core"]
        assert not missed, (
            f"canonical attacks missed by every layer: {missed}. A control that does not fire "
            "means the instrument — not the detector — is the thing to fix first."
        )
