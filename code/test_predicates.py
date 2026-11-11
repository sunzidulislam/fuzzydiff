"""Tests of the prompt-to-predicate translation (python -I test_predicates.py).

1. Exhaustive check against the gold programs of the template suites over a slot grid.
2. Targeted cases: unsupported constructions (manner adverbs, counts, physical quantities,
   unknown modifiers) never create degree constraints; context PPs create no constraints;
   negation, visibility, and spatial relations map to F7, F1, F9.
Writes translation_check.json (constraint precision/recall and exact-match rate per suite).
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predicates import (Constraint, gold_color_mixture, gold_size, gold_soft_attribute,  # noqa: E402
                        score, translate)

OBJ = ["mug", "teapot", "vase", "chair", "lamp", "towel", "candle", "bottle", "kitten", "pillow"]


def suite(name, cases):
    res = [score(translate(prompt), gold) for gold, prompt in cases]
    n = len(res)
    out = {"n": n,
           "precision": sum(r["precision"] for r in res) / n,
           "recall": sum(r["recall"] for r in res) / n,
           "exact": sum(r["exact"] for r in res) / n}
    bad = [prompt for (gold, prompt), r in zip(cases, res) if not r["exact"]]
    assert not bad, f"{name}: {len(bad)} mismatches, e.g. {bad[:3]}"
    return out


def main():
    results = {}
    hedges = ["barely", "slightly", "somewhat", "moderately", "very", "extremely", "gently", "mildly"]
    attrs = ["dark", "bright", "pale", "vivid", "red", "blue", "folded", "twisted"]
    cases = [gold_soft_attribute("a", h, a, o) for h, a, o in itertools.product(hedges, attrs, OBJ)]
    results["(i) soft attribute"] = suite("i", cases)

    colors = ["greenish-gray", "turquoise-green", "coral-orange", "bluish-purple", "pinkish-white"]
    cases = [gold_color_mixture(c1, o1, c2, o2)
             for c1, c2 in itertools.permutations(colors, 2)
             for o1, o2 in itertools.permutations(OBJ[:5], 2)]
    results["(ii) color mixture"] = suite("ii", cases)

    comps = ["larger", "bigger", "smaller", "taller", "shorter"]
    cases = [gold_size(o1, h, c, o2) for o1, o2 in itertools.permutations(OBJ, 2)
             for h in hedges[:6] for c in comps]
    results["(iii) size"] = suite("iii", cases)

    # targeted cases
    p = translate("a gently twisted rope around a pole")
    assert not [c for c in p.constraints if c.tier == 3], p
    assert "adverb:gently" in p.unsupported and p.context == ["pole"]
    p = translate("a few fast cars")
    assert {"count:a few", "physical:fast"} <= set(p.unsupported)
    assert not [c for c in p.constraints if c.tier == 3]
    p = translate("a candle slightly shorter than a bottle on a wooden shelf")
    assert Constraint("F8", ("candle", "bottle"), "slightly", "height", -1) in p.constraints
    assert p.context == ["shelf"] and "context-modifier:wooden" in p.unsupported
    assert not [c for c in p.constraints if "shelf" in c.args]
    p = translate("a barely visible deer")
    assert Constraint("F1", ("deer",), "barely", "visibility", +1) in p.constraints
    assert Constraint("presence", ("deer",), None, None, +1, tier=1) in p.constraints
    p = translate("almost no snow")
    assert Constraint("F7", ("snow",), "extremely", "occupancy", -1) in p.constraints
    p = translate("a mug left of a teapot")
    assert Constraint("F9", ("mug", "teapot"), None, "left", +1) in p.constraints
    p = translate("a slightly red mug and a blue teapot")
    fams = sorted(c.family for c in p.constraints)
    assert fams.count("binding") == 2 and fams.count("leakage") == 2 and fams.count("F6") == 1, fams
    p = translate("a very dimly lit room")
    assert Constraint("F2a", ("room",), "very", "scene_lightness", -1) in p.constraints
    # GenEval's frame names no object: the program is the scene's own
    framed, plain = translate("a photo of a red cake and a blue bench"), translate("a red cake and a blue bench")
    assert framed.constraints == plain.constraints and framed.objects == ["cake", "bench"]
    assert "photo" not in translate("A photo of a cow").objects
    results["targeted cases"] = "pass"

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translation_check.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
