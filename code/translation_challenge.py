"""Author-written challenge set for the prompt-to-predicate translation (paper: Tab. translation).

Each item gives a prompt and its gold constraint program, written from the intended meaning
(not from the implementation's output). Categories: comparisons (orientation, hedges, copula,
unlisted intensifiers), negation (soft, hard, 'without', negated attributes), color-name
ordering ('greenish-gray' vs 'grayish-green', spaced and suffix forms), attributes/context, and
constructions outside the template grammar (relative clauses, adjective coordination,
attribute comparatives, counts). The last group documents what the reference implementation
does NOT handle; free-form prompts need the dependency parser.
Usage: python -I translation_challenge.py   (writes translation_challenge.json)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predicates import Constraint as C, Program, score, translate  # noqa: E402


def P(objects, cons, unsupported=()):
    p = Program(objects=list(objects))
    p.constraints = list(cons) + [C("presence", (q,), None, None, +1, tier=1) for q in objects]
    p.unsupported = list(unsupported)
    return p


def B(a, q):
    return C("binding", (a, q), None, None, +1, tier=2)


def Lk(a, q, r):
    return C("leakage", (a, q, r), None, None, -1, tier=2)


ITEMS = [
    # ---- comparisons
    ("comparison", "a cat slightly larger than a dog", P(["cat", "dog"], [C("F8", ("cat", "dog"), "slightly", "area", +1)])),
    ("comparison", "a dog slightly smaller than a cat", P(["dog", "cat"], [C("F8", ("dog", "cat"), "slightly", "area", -1)])),
    ("comparison", "a mug very much bigger than a spoon", P(["mug", "spoon"], [C("F8", ("mug", "spoon"), None, "area", +1)], ["token:much"])),
    ("comparison", "a vase much larger than a cup", P(["vase", "cup"], [C("F8", ("vase", "cup"), None, "area", +1)], ["adverb:much"])),
    ("comparison", "the tree is slightly taller than the house", P(["tree", "house"], [C("F8", ("tree", "house"), "slightly", "height", +1)])),
    ("comparison", "a lamp moderately shorter than a shelf", P(["lamp", "shelf"], [C("F8", ("lamp", "shelf"), "moderately", "height", -1)])),
    ("comparison", "a kitten barely smaller than a pillow on a sofa", P(["kitten", "pillow"], [C("F8", ("kitten", "pillow"), "barely", "area", -1)])),
    ("comparison", "a boat wider than a car", P(["boat", "car"], [C("F8", ("boat", "car"), None, "width", +1)])),
    ("comparison", "a slightly dark mug larger than a teapot", P(["mug", "teapot"], [C("F2a", ("mug",), "slightly", "lightness", -1), C("F8", ("mug", "teapot"), None, "area", +1), B("dark", "mug")])),
    ("comparison", "a bottle is very much smaller than a barrel", P(["bottle", "barrel"], [C("F8", ("bottle", "barrel"), None, "area", -1)], ["token:much"])),
    # ---- negation
    ("negation", "almost no snow", P([], [C("F7", ("snow",), "extremely", "occupancy", -1)])),
    ("negation", "no clouds", P([], [C("F7", ("clouds",), None, "occupancy", -1)])),
    ("negation", "a street without cars", P(["street"], [C("F7", ("cars",), None, "occupancy", -1)])),
    ("negation", "a mug that is not red", P(["mug"], [], ["negated:red"])),
    ("negation", "a field with almost no flowers", P(["field"], [C("F7", ("flowers",), "extremely", "occupancy", -1)])),
    ("negation", "a barely visible deer", P(["deer"], [C("F1", ("deer",), "barely", "visibility", +1)])),
    ("negation", "a sky without birds above a lake", P(["sky"], [C("F7", ("birds",), None, "occupancy", -1)])),
    ("negation", "the car is not dark", P(["car"], [], ["negated:dark"])),
    # ---- color-name ordering
    ("color order", "a greenish-gray mug", P(["mug"], [C("F6", ("mug",), "slightly", ("gray", "green"), +1)])),
    ("color order", "a grayish-green mug", P(["mug"], [C("F6", ("mug",), "slightly", ("green", "gray"), +1)])),
    ("color order", "a greenish gray mug", P(["mug"], [C("F6", ("mug",), "slightly", ("gray", "green"), +1)])),
    ("color order", "a blue-green vase", P(["vase"], [C("F6", ("vase",), "moderately", ("green", "blue"), +1)])),
    ("color order", "a green-blue vase", P(["vase"], [C("F6", ("vase",), "moderately", ("blue", "green"), +1)])),
    ("color order", "a reddish mug", P(["mug"], [C("F6", ("mug",), "slightly", ("achromatic", "red"), +1)])),
    ("color order", "a bluish-purple bowl and a reddish-brown jar",
     P(["bowl", "jar"], [C("F6", ("bowl",), "slightly", ("purple", "blue"), +1), C("F6", ("jar",), "slightly", ("brown", "red"), +1),
                         B("bluish-purple", "bowl"), B("reddish-brown", "jar"), Lk("bluish-purple", "bowl", "jar"), Lk("reddish-brown", "jar", "bowl")])),
    ("color order", "a very red mug and a slightly blue teapot",
     P(["mug", "teapot"], [C("F6", ("mug",), "very", ("achromatic", "red"), +1), C("F6", ("teapot",), "slightly", ("achromatic", "blue"), +1),
                           B("red", "mug"), B("blue", "teapot"), Lk("red", "mug", "teapot"), Lk("blue", "teapot", "mug")])),
    # ---- attributes and context
    ("attribute/context", "the mug is slightly dark", P(["mug"], [C("F2a", ("mug",), "slightly", "lightness", -1)])),
    ("attribute/context", "a very pale vase on a wooden table", P(["vase"], [C("F2a", ("vase",), "very", "chroma", -1)], ["context-modifier:wooden"])),
    ("attribute/context", "a gently twisted rope around a pole", P(["rope"], [], ["adverb:gently"])),
    ("attribute/context", "a somewhat folded towel", P(["towel"], [C("F2b", ("towel", "folded"), "somewhat", "siglip", +1)])),
    ("attribute/context", "a very dimly lit room", P(["room"], [C("F2a", ("room",), "very", "scene_lightness", -1)])),
    ("attribute/context", "a few fast cars", P(["cars"], [], ["count:a few", "physical:fast"])),
    # ---- outside the template grammar (expected to need the dependency parser)
    ("out of grammar", "a mug that is slightly dark", P(["mug"], [C("F2a", ("mug",), "slightly", "lightness", -1)])),
    ("out of grammar", "a slightly dark and very vivid mug", P(["mug"], [C("F2a", ("mug",), "slightly", "lightness", -1), C("F2a", ("mug",), "very", "chroma", +1)])),
    ("out of grammar", "a mug slightly darker than the teapot", P(["mug", "teapot"], [], ["comparative-attribute:darker"])),
    ("out of grammar", "two slightly red mugs", P(["mugs"], [C("F6", ("mugs",), "slightly", ("achromatic", "red"), +1)], ["count:two"])),
    ("out of grammar", "a teapot next to a slightly larger mug", P(["teapot", "mug"], [C("F8", ("mug", "teapot"), "slightly", "area", +1)])),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    by_cat, fails = {}, []
    for cat, prompt, gold in ITEMS:
        try:
            pred = translate(prompt)
            sc = score(pred, gold)
        except Exception as e:  # a crash counts as a failure
            sc, pred = {"precision": 0.0, "recall": 0.0, "exact": False}, f"error: {e}"
        d = by_cat.setdefault(cat, {"n": 0, "exact": 0, "precision": 0.0, "recall": 0.0, "wrong": 0, "degraded": 0})
        if not isinstance(pred, str):
            # safety: any emitted tier-2/3 constraint or object that is not in the gold program
            extra = [c for c in pred.constraints if c.tier > 1 and c not in set(gold.constraints)]
            d["wrong"] += int(bool(extra) or any(q not in gold.objects for q in pred.objects))
            d["degraded"] += int(pred.degraded)
        d["n"] += 1
        d["exact"] += int(sc["exact"])
        d["precision"] += sc["precision"]
        d["recall"] += sc["recall"]
        if not sc["exact"]:
            fails.append({"category": cat, "prompt": prompt})
    for d in by_cat.values():
        d["precision"] /= d["n"]
        d["recall"] /= d["n"]
        d["exact_rate"] = d["exact"] / d["n"]
    out = {"by_category": by_cat, "failures": fails,
           "in_grammar_exact": sum(d["exact"] for c, d in by_cat.items() if c != "out of grammar")
           / sum(d["n"] for c, d in by_cat.items() if c != "out of grammar")}
    with open(os.path.join(here, "translation_challenge.json"), "w") as f:
        json.dump(out, f, indent=2)
    first = {}
    fr = os.path.join(here, "translation_challenge_firstrun.json")
    if os.path.exists(fr):   # results of the first run, before fixing the bugs this set exposed
        first = {c: d["exact"] / d["n"] for c, d in json.load(open(fr))["by_category"].items()}
    rows = [f"{c} ({d['n']}) & {d['precision']:.2f} & {d['recall']:.2f} & "
            f"{first.get(c, float('nan')):.2f} $\\to$ {d['exact_rate']:.2f} & {d['wrong']} / {d['degraded']}\\\\"
            for c, d in by_cat.items()]
    with open(os.path.join(here, "translation_rows.tex"), "w") as f:
        f.write("% Generated by translation_challenge.py; do not edit.\n")
        f.write("\\def\\translationChallengeRows{%\n" + "\n".join(rows) + "\n}\n")
        f.write("\\def\\translationChallengeN{" + str(len(ITEMS)) + "}\n")
        f.write("\\def\\translationFailures{" + "; ".join(f"``{x['prompt']}''" for x in fails).replace("_", "\\_") + "}\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
