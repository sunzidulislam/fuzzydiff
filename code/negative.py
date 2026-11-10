"""Negative prompt derived from the parsed program: the outcomes that violate its constraints.

  F8        the reversed relation ("a shoe bigger than a kettle"), and "the same size" unless the
            hedge asks for a small difference (center below SAME_SIZE_MIN)
  presence  each object alone ("only a kettle"), when the prompt has two or more objects
  binding   the colours swapped between two bound objects ("a blue mug and a red teapot")

Used only by the *_neg arms; plain SD3 keeps the empty negative prompt.
"""
from typing import Dict

SAME_SIZE_MIN = 0.35            # hedge center from which "the same size" counts as a violation


def negative_prompt(program, centers: Dict[str, float], default_center: float = 0.7) -> str:
    parts = []
    for c in program.constraints:
        if c.family != "F8":
            continue
        q, r = c.args
        larger, smaller = (q, r) if c.orientation > 0 else (r, q)
        parts.append(f"a {smaller} bigger than a {larger}")
        if (centers.get(c.hedge, default_center) if c.hedge else default_center) >= SAME_SIZE_MIN:
            parts.append(f"a {q} and a {r} of the same size")
    if len(program.objects) >= 2:
        parts += [f"only a {q}" for q in program.objects]
    bound = [c.args for c in program.constraints if c.family == "binding"]
    if len(bound) == 2 and bound[0][0] != bound[1][0]:
        (c1, o1), (c2, o2) = bound
        parts.append(f"a {c2} {o1} and a {c1} {o2}")
    return ", ".join(parts)
