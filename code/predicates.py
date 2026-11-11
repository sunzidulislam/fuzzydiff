"""Prompt-to-predicate translation (paper: Sec. 3.1, App. "Prompt-to-Predicate Translation").

A prompt y is translated by a deterministic function tau(y) into a *constraint program*
    Pi(y) = (Q, K, U)
  Q: object mentions (head noun with its span),
  K: typed constraints k = (family, args, hedge, orientation, tier),
  U: constructions recognized but outside the guided inventory (logged, never guided).

Every object in Q gets a presence constraint (tier 1). Every attribute bound to an object gets a
binding constraint (tier 2) when the prompt mentions more than one object. Degree constraints
(tier 3) are emitted only for constructions in the closed inventory below; hedges come from a
closed lexicon (paper: Tab. hedge targets). Unknown modifiers never create constraints.

This module implements the *template grammar* of the evaluation suites, for which the gold
program is known by construction (`gold_*` functions); free-form prompts (domain-specific
section) use a dependency parser that emits the same Program type (not included here).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---- closed lexicons -------------------------------------------------------------------------
# hedge surface form -> canonical hedge (ordinal order of the paper's hedge table)
HEDGES = {
    "barely": "barely", "a little": "a little", "faintly": "a little", "slightly": "slightly",
    "relatively": "somewhat", "somewhat": "somewhat", "moderately": "moderately",
    "considerably": "considerably", "very": "very", "mostly": "very", "nearly": "nearly",
    "extremely": "extremely", "almost": "extremely",
}
HEDGE_ORDER = ["barely", "a little", "slightly", "somewhat", "moderately", "considerably",
               "very", "nearly", "extremely"]
# degree adverbs that are NOT in the lexicon (no acceptance data) and manner adverbs
UNSUPPORTED_ADVERBS = {"gently", "mildly", "softly", "quickly", "slowly", "loosely", "tightly",
                       "much", "far", "way", "significantly", "hugely"}  # intensifiers without acceptance data

# photometric attributes (F2a): quantity and polarity on that quantity
PHOTOMETRIC = {
    "dark": ("lightness", -1), "dim": ("lightness", -1), "bright": ("lightness", +1),
    "pale": ("chroma", -1), "faded": ("chroma", -1), "muted": ("chroma", -1),
    "vivid": ("chroma", +1), "saturated": ("chroma", +1),
    "dimly lit": ("scene_lightness", -1), "brightly lit": ("scene_lightness", +1),
    "lit": ("scene_lightness", +1),           # hedged: a moderately lit room
}
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "gray", "grey",
          "white", "black", "turquoise", "coral", "teal", "beige", "navy", "olive", "violet", "cyan"}
# open-vocabulary attributes admitted *individually* after both validity tests (F2b)
OPEN_VOCAB = {"folded", "curved", "twisted", "wrinkled", "cracked", "rusty"}
# comparatives (F8): orientation o and the measured dimension
COMPARATIVES = {
    "larger": (+1, "area"), "bigger": (+1, "area"), "smaller": (-1, "area"),
    "taller": (+1, "height"), "shorter": (-1, "height"),
    "wider": (+1, "width"), "narrower": (-1, "width"),
}
SPATIAL = {"left of": "left", "right of": "right", "above": "above", "below": "below",
           "beside": "beside", "next to": "beside"}
COUNTS = {"a few", "several", "many", "two", "three", "four", "some"}
PHYSICAL = {"fast", "heavy", "hot", "cold", "loud", "slow", "light-weight"}
ARTICLES = {"a", "an", "the"}
BOUNDARIES = {"and", "than", "on", "in", "with", "without", "is", "are", "looks", "appears", "seems", "under", "near", "at", "of", "beside", "above",
              "below", "next", "left", "right", "to", "around", "behind", "inside", "over", "by",
              "from", "against", "into", "onto", "beneath", "along", "among", "through", "across",
              "that", "which"}

COPULA = {"is", "are", "looks", "appears", "seems"}

TIER = {"presence": 1, "binding": 2, "leakage": 2}


@dataclass(frozen=True)
class Constraint:
    family: str                     # presence | binding | leakage | F1 | F2a | F2b | F6 | F7 | F8 | F9
    args: Tuple[str, ...]           # object heads (and attribute word for binding)
    hedge: Optional[str] = None     # canonical hedge, or None (default interval of the family)
    quantity: Optional[str] = None  # e.g. lightness, chroma, area, height, (c1, c2) for F6
    orientation: int = 1            # polarity / comparison direction o
    tier: int = 3


@dataclass
class Program:
    objects: List[str] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)   # nouns of scene context (no constraints)
    degraded: bool = False                             # grammar violated -> presence-only fallback

    def key(self):
        return (tuple(self.objects), frozenset(self.constraints), tuple(sorted(self.unsupported)))


# ---- tokenization with multi-word lexicon entries ---------------------------------------------
_MULTI = sorted([k for k in list(HEDGES) + list(PHOTOMETRIC) + list(SPATIAL) + list(COUNTS)
                 if " " in k], key=len, reverse=True)


def tokenize(prompt: str) -> List[str]:
    s = prompt.lower().strip().rstrip(".")
    for m in _MULTI:
        s = re.sub(rf"\b{re.escape(m)}\b", m.replace(" ", "_"), s)
    return [t.replace("_", " ") for t in s.split()]


def _color_term(tok: str):
    """'greenish-gray' -> ('gray', 'green', 'ish'); 'turquoise-green' -> ('green', 'turquoise', 'compound');
    'red' -> ('red', None, 'single'); else None."""
    if "-" in tok:
        a, b = tok.split("-", 1)
        if a.endswith("ish") and _ish_color(a) and b in COLORS:
            return (b, _ish_color(a), "ish")
        if a in COLORS and b in COLORS:
            return (b, a, "compound")
        return None
    return (tok, None, "single") if tok in COLORS else None


def _ish_color(tok):
    """'reddish' -> 'red', 'greenish' -> 'green', 'bluish' -> 'blue'; else None."""
    if tok.endswith("ish"):
        stem = tok[:-3]
        for cand in (stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else None):
            if cand in COLORS:
                return cand
    return None


def _degree_run_to_comparative(toks, j):
    """True if toks[j:] starts with one or more hedges/intensifiers followed by a comparative."""
    k = j
    while k < len(toks) and (toks[k] in HEDGES or toks[k] in UNSUPPORTED_ADVERBS):
        k += 1
    return k > j and k < len(toks) and toks[k] in COMPARATIVES


def _parse_np(toks, i):
    """Parse [art] [hedge|attr|color|unknown]* head; returns (head, mods, i_next)."""
    if i < len(toks) and toks[i] in ARTICLES:
        i += 1
    mods = []
    j = i
    while j < len(toks) and toks[j] not in BOUNDARIES and toks[j] not in COMPARATIVES \
            and toks[j] not in SPATIAL \
            and not _degree_run_to_comparative(toks, j):
        j += 1
    if j == i:
        return None, [], i
    head = toks[j - 1]
    mods = toks[i:j - 1]
    return head, mods, j


def _modifiers(prog: Program, head: str, mods: List[str]):
    """Turn NP modifiers into constraints; returns attribute words bound to `head`."""
    bound = []
    hedge = None
    k = 0
    while k < len(mods):
        m = mods[k]
        if m == "not" and k + 1 < len(mods):            # negated attribute: recognized, never guided
            prog.unsupported.append(f"negated:{mods[k + 1]}")
            k += 2
            hedge = None
            continue
        if _ish_color(m) and k + 1 < len(mods) and _color_term(mods[k + 1]) and "-" not in mods[k + 1]:
            m = f"{m}-{mods[k + 1]}"                     # 'greenish gray' == 'greenish-gray'
            k += 1
        elif _ish_color(m) and not _color_term(m):       # 'reddish' alone == 'slightly red'
            base = _ish_color(m)
            bound.append(m)
            if hedge != "__unsupported__":
                prog.constraints.append(Constraint("F6", (head,), hedge or "slightly", ("achromatic", base), +1))
            hedge = None
            k += 1
            continue
        if m in HEDGES:
            hedge = HEDGES[m]
        elif m in UNSUPPORTED_ADVERBS:
            prog.unsupported.append(f"adverb:{m}")
            hedge = "__unsupported__"
        elif m in COUNTS:
            prog.unsupported.append(f"count:{m}")
        elif m in PHYSICAL:
            prog.unsupported.append(f"physical:{m}")
        elif m in PHOTOMETRIC:
            q, pol = PHOTOMETRIC[m]
            bound.append(m)
            if hedge != "__unsupported__":
                prog.constraints.append(Constraint("F2a", (head,), hedge, q, pol))
            hedge = None
        elif m in OPEN_VOCAB:
            bound.append(m)
            if hedge != "__unsupported__":
                prog.constraints.append(Constraint("F2b", (head, m), hedge, "siglip", +1))
            hedge = None
        elif _color_term(m):
            base, tint, kind = _color_term(m)
            bound.append(m)
            if hedge == "__unsupported__":
                pass
            elif kind == "single":
                # hedged single color: axis from achromatic (same L*) to the color prototype
                if hedge is not None:
                    prog.constraints.append(Constraint("F6", (head,), hedge, ("achromatic", base), +1))
            elif kind == "ish":       # 'greenish-gray': gray dominant, green tint
                prog.constraints.append(Constraint("F6", (head,), hedge or "slightly", (base, tint), +1))
            else:                     # 'turquoise-green': balanced blend
                prog.constraints.append(Constraint("F6", (head,), hedge or "moderately", (base, tint), +1))
            hedge = None
        else:
            prog.unsupported.append(f"modifier:{m}")
        k += 1
    return bound


_LEXICAL = None


def _is_lexical(w):
    """True for words that the grammar uses as modifiers/operators and can never be a noun head."""
    global _LEXICAL
    if not w:
        return True                       # a missing head is a grammar violation
    if _LEXICAL is None:
        _LEXICAL = (set(HEDGES) | set(PHOTOMETRIC) | COLORS | OPEN_VOCAB | set(COMPARATIVES) | COPULA
                    | UNSUPPORTED_ADVERBS | ARTICLES | {"not", "no", "visible", "than"})
    return w in _LEXICAL or bool(_color_term(w)) or bool(_ish_color(w))


def _degrade(prog: Program, reason: str) -> Program:
    """Deterministic fallback for prompts outside the grammar: keep presence for heads that are
    genuine nouns, drop every degree, binding, and leakage constraint (attachment is uncertain)."""
    objs = [q for q in prog.objects if not _is_lexical(q)]
    out = Program(objects=objs, unsupported=prog.unsupported + [f"degraded:{reason}"], context=prog.context,
                  degraded=True)
    out.constraints = [Constraint("presence", (q,), None, None, +1, tier=1) for q in objs]
    return out


# GenEval writes every prompt as "a photo of <scene>". The frame names no object, so it is removed before
# parsing; the image is still generated from the full prompt.
PHOTO_FRAME = "a photo of "


def translate(prompt: str) -> Program:
    """Translate, then validate against the grammar; violations degrade to presence only."""
    if prompt.lower().startswith(PHOTO_FRAME):
        prompt = prompt[len(PHOTO_FRAME):]
    try:
        prog = _translate(prompt)
    except Exception as e:                       # any parse failure degrades, never raises
        return _degrade(Program(unsupported=["unparsed"]), f"parse error ({type(e).__name__})")
    toks = tokenize(prompt)
    if "than" in toks and not any(c.family == "F8" for c in prog.constraints):
        # e.g. attribute comparatives ('darker than'): the word before 'than' is not a noun
        before_than = {toks[k - 1] for k, w in enumerate(toks) if w == "than" and k > 0}
        prog.objects = [q for q in prog.objects if q not in before_than]
        return _degrade(prog, "unsupported comparative")
    bad_heads = [q for q in prog.objects + prog.context if _is_lexical(q)]
    bad_args = [a for c in prog.constraints if c.family in ("F7",) for a in c.args if _is_lexical(a)]
    leftovers = [u for u in prog.unsupported if u.startswith("token:") or u == "unparsed"]
    if bad_heads or bad_args or leftovers:
        return _degrade(prog, "lexical head" if bad_heads or bad_args else "unconsumed tokens")
    return prog


def _translate(prompt: str) -> Program:
    """Template grammar of the evaluation suites:
         NP (and NP)*  |  NP [hedge] COMP than NP [context]  |  NP SPATIAL NP  |  'almost no' NP
       with NP = [art] modifier* head and optional context PPs ('on a wooden shelf')."""
    toks = tokenize(prompt)
    prog = Program()
    bound = {}
    i = 0
    ambiguous = False
    neg = None
    if toks[:2] == ["almost", "no"]:
        neg, i = "extremely", 2
    elif toks[:1] == ["no"]:
        neg, i = "none", 1
    head, mods, i = _parse_np(toks, i)
    if head is None:
        prog.unsupported.append("unparsed")
        return prog
    if neg:
        # soft negation (F7): bounded occupancy of the whole image; the negated concept is NOT an
        # object, so it gets no presence constraint
        prog.constraints.append(Constraint("F7", (head,), None if neg == "none" else neg, "occupancy", -1))
        prog.unsupported.extend(f"negated-modifier:{m}" for m in mods)
        bound[head] = []
    elif mods[:2] == ["barely", "visible"]:     # 'barely visible X': presence + low visibility
        prog.objects.append(head)
        prog.constraints.append(Constraint("F1", (head,), "barely", "visibility", +1))
        bound[head] = _modifiers(prog, head, mods[2:])
    else:
        prog.objects.append(head)
        bound[head] = _modifiers(prog, head, mods)

    while i < len(toks):
        t = toks[i]
        if t in COPULA:                          # 'the mug is slightly dark' / 'X is larger than Y'
            j = i + 1
            while j < len(toks) and toks[j] not in BOUNDARIES and toks[j] not in COMPARATIVES \
                    and not (toks[j] in HEDGES and j + 1 < len(toks) and toks[j + 1] in COMPARATIVES):
                j += 1
            if j > i + 1:
                bound[head] = bound.get(head, []) + _modifiers(prog, head, toks[i + 1:j])
            i = j
            continue
        if t in ("that", "which") and i + 1 < len(toks) and toks[i + 1] in COPULA:
            i += 1
            continue
        if t == "with" and toks[i + 1:i + 3] == ["almost", "no"] or t == "with" and toks[i + 1:i + 2] == ["no"]:
            k = i + 3 if toks[i + 1] == "almost" else i + 2
            h2, m2, i = _parse_np(toks, k)
            if h2:
                prog.constraints.append(Constraint("F7", (h2,), "extremely" if toks[k - 2] == "almost" else None,
                                                   "occupancy", -1))
            ambiguous = True
            continue
        if t == "without":                       # 'a street without cars': F7 on the whole image
            h2, m2, i = _parse_np(toks, i + 1)
            if h2:
                prog.constraints.append(Constraint("F7", (h2,), None, "occupancy", -1))
                prog.unsupported.extend(f"negated-modifier:{m}" for m in m2)
            ambiguous = True                     # a following PP could attach to the negated noun
            continue
        if t == "and":
            ambiguous = False
            h2, m2, i = _parse_np(toks, i + 1)
            prog.objects.append(h2)
            bound[h2] = _modifiers(prog, h2, m2)
        elif t in COMPARATIVES or _degree_run_to_comparative(toks, i):
            run = []
            while toks[i] not in COMPARATIVES:
                run.append(toks[i])
                i += 1
            unsup = [w for w in run if w in UNSUPPORTED_ADVERBS]
            prog.unsupported.extend(f"adverb:{w}" for w in unsup)
            # a lexicon hedge counts only if no unlisted intensifier modifies the comparative;
            # otherwise the magnitude is unknown and only the direction is kept
            hedge = HEDGES[run[-1]] if run and not unsup and run[-1] in HEDGES else None
            o, dim = COMPARATIVES[toks[i]]
            if i + 1 >= len(toks) or toks[i + 1] != "than":
                prog.unsupported.append(f"token:{toks[i]}")   # comparative without 'than' -> degrade
                i += 1
                continue
            h2, m2, i = _parse_np(toks, i + 2)
            prog.objects.append(h2)
            bound[h2] = _modifiers(prog, h2, m2)
            prog.constraints.append(Constraint("F8", (head, h2), hedge, dim, o))
        elif t in SPATIAL and ambiguous:          # attachment ambiguous -> context, no relation
            h2, m2, i = _parse_np(toks, i + 1)
            if h2:
                prog.context.append(h2)
                prog.unsupported.append(f"ambiguous-attachment:{t}")
        elif t in SPATIAL:
            h2, m2, i = _parse_np(toks, i + 1)
            prog.objects.append(h2)
            bound[h2] = _modifiers(prog, h2, m2)
            prog.constraints.append(Constraint("F9", (head, h2), None, SPATIAL[t], +1))
        elif t in BOUNDARIES:            # context PP: 'on a wooden shelf' -> context noun, no constraint
            h2, m2, i = _parse_np(toks, i + 1)
            if h2:
                prog.context.append(h2)
                prog.unsupported.extend(f"context-modifier:{m}" for m in m2)
        else:
            prog.unsupported.append(f"token:{t}")
            i += 1

    for q in prog.objects:
        prog.constraints.append(Constraint("presence", (q,), None, None, +1, tier=1))
    multi = len(prog.objects) > 1
    for q, attrs in bound.items():
        for a in attrs:
            if multi:
                prog.constraints.append(Constraint("binding", (a, q), None, None, +1, tier=2))
                for r in prog.objects:
                    if r != q and _color_term(a):
                        prog.constraints.append(Constraint("leakage", (a, q, r), None, None, -1, tier=2))
    return prog


# ---- gold programs of the template suites (known by construction) -----------------------------
def gold_soft_attribute(art, hedge, attr, obj):
    """Exp. (i): '{art} {hedge} {attr} {obj}'."""
    p = Program(objects=[obj])
    if attr in PHOTOMETRIC:
        q, pol = PHOTOMETRIC[attr]
        if hedge in HEDGES:
            p.constraints.append(Constraint("F2a", (obj,), HEDGES[hedge], q, pol))
    elif attr in OPEN_VOCAB:
        if hedge in HEDGES:
            p.constraints.append(Constraint("F2b", (obj, attr), HEDGES[hedge], "siglip", +1))
    elif attr in COLORS:
        if hedge in HEDGES:
            p.constraints.append(Constraint("F6", (obj,), HEDGES[hedge], ("achromatic", attr), +1))
    if hedge in UNSUPPORTED_ADVERBS:
        p.unsupported.append(f"adverb:{hedge}")
    p.constraints.append(Constraint("presence", (obj,), None, None, +1, tier=1))
    return p, f"{art} {hedge} {attr} {obj}"


def gold_color_mixture(c1, o1, c2, o2):
    """Exp. (ii): 'a {c1} {o1} and a {c2} {o2}' with c = 'Xish-Y' or 'X-Y'."""
    p = Program(objects=[o1, o2])
    for c, o in ((c1, o1), (c2, o2)):
        base, tint, kind = _color_term(c)
        p.constraints.append(Constraint("F6", (o,), "slightly" if kind == "ish" else "moderately", (base, tint), +1))
    for q in (o1, o2):
        p.constraints.append(Constraint("presence", (q,), None, None, +1, tier=1))
    for c, q, r in ((c1, o1, o2), (c2, o2, o1)):
        p.constraints.append(Constraint("binding", (c, q), None, None, +1, tier=2))
        p.constraints.append(Constraint("leakage", (c, q, r), None, None, -1, tier=2))
    return p, f"a {c1} {o1} and a {c2} {o2}"


def gold_size(o1, hedge, comp, o2):
    """Exp. (iii): 'a {o1} {hedge} {comp} than a {o2}'."""
    o, dim = COMPARATIVES[comp]
    p = Program(objects=[o1, o2])
    p.constraints.append(Constraint("F8", (o1, o2), HEDGES.get(hedge), dim, o))
    for q in (o1, o2):
        p.constraints.append(Constraint("presence", (q,), None, None, +1, tier=1))
    return p, f"a {o1} {hedge} {comp} than a {o2}"


def score(pred: Program, gold: Program):
    """Constraint-level precision/recall and exact match (paper: Tab. translation)."""
    P, G = set(pred.constraints), set(gold.constraints)
    tp = len(P & G)
    prec = tp / len(P) if P else 1.0
    rec = tp / len(G) if G else 1.0
    return {"precision": prec, "recall": rec, "exact": P == G and pred.objects == gold.objects}
