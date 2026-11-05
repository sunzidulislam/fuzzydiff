"""Evaluation prompt suites and their hash-based groups (paper: Tab. data_splits).

A base prompt is the prompt with its hedge removed. A fixed hash of the base prompt decides its group,
so generating more prompts later cannot move a base prompt between groups or leak it into test.
Objects listed as test-only appear only in test prompts, flagged `unseen` and reported separately.
"""
from __future__ import annotations

import hashlib
import itertools
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_config import ROOT  # noqa: E402
from predicates import HEDGES, translate  # noqa: E402

VOCAB_PATH = os.path.join(ROOT, "configs", "vocab.yaml")
SALT = "fuzzydiff-v1"
FIXED_DEV = {"plain_color", "m1_tune", "m1_eval", "m1_ladder"}
_SINGLE_HEDGES = {h for h in HEDGES if " " not in h}


@lru_cache(maxsize=1)
def vocab():
    with open(VOCAB_PATH) as f:
        return yaml.safe_load(f)


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


def base_prompt(prompt: str) -> str:
    """Remove the hedge and repair the article that preceded it."""
    tokens = prompt.split()
    out: List[str] = []
    for i, t in enumerate(tokens):
        if t in _SINGLE_HEDGES and not (t == "moderately" and i + 1 < len(tokens) and tokens[i + 1] == "lit"):
            continue
        out.append(t)
    for i in range(len(out) - 1):
        if out[i] in ("a", "an"):
            out[i] = _art(out[i + 1])
    return " ".join(out)


def assign_group(base: str, salt: str = SALT) -> str:
    digest = hashlib.sha256(f"{salt}|{base}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2 ** 64
    edge = 0.0
    shares = vocab()["group_shares"]
    for group, share in shares.items():
        edge += share
        if u < edge:
            return group
    return list(shares)[-1]


@dataclass(frozen=True)
class PromptRow:
    prompt: str
    base: str
    group: str
    suite: str
    unseen: bool = False

    @property
    def program(self):
        return translate(self.prompt)


def _row(prompt, suite, objects):
    base = base_prompt(prompt)
    unseen = bool(set(objects) & set(vocab()["test_only_objects"]))
    if suite in FIXED_DEV:
        group = "dev"
    else:
        group = "test" if unseen else assign_group(base)
    return PromptRow(prompt, base, group, suite, unseen)


def _soft():
    v = vocab()
    for h, a, o, c in itertools.product(v["hedges"], v["soft_attributes"], v["objects"] + v["test_only_objects"],
                                        v["contexts"]):
        text = f"{_art(h)} {h} {a} {o}" + (f" {c}" if c else "")
        yield text, (o,)


def _mixture():
    v = vocab()
    objects = v["objects"][:v["mixture_objects"]] + v["test_only_objects"]
    for (c1, c2), (o1, o2) in itertools.product(itertools.permutations(v["mixture_terms"], 2),
                                                itertools.permutations(objects, 2)):
        yield f"{_art(c1)} {c1} {o1} and {_art(c2)} {c2} {o2}", (o1, o2)


def _size():
    v = vocab()
    objects = v["objects"] + v["test_only_objects"]
    for (o1, o2), h, c in itertools.product(itertools.permutations(objects, 2), v["hedges"], v["comparatives"]):
        yield f"{_art(o1)} {o1} {h} {c} than {_art(o2)} {o2}", (o1, o2)


def _plain():
    v = vocab()
    for (c1, c2), (o1, o2) in itertools.product(itertools.permutations(v["plain_colors"], 2),
                                                itertools.permutations(v["plain_objects"], 2)):
        yield f"{_art(c1)} {c1} {o1} and {_art(c2)} {c2} {o2}", (o1, o2)


def _nouns(key):
    for n in vocab()[key]:
        yield f"{_art(n)} {n}", ()


def _ladder():
    v = vocab()
    for n, d in itertools.product(v["m1_ladder_nouns"], v["m1_ladder_degrees"]):
        yield f"a {d} {n}", ()


SUITES = {"soft": _soft, "mixture": _mixture, "size": _size, "plain_color": _plain,
          "m1_tune": lambda: _nouns("m1_tune"), "m1_eval": lambda: _nouns("m1_eval"), "m1_ladder": _ladder}


@lru_cache(maxsize=None)
def _enumerate(name):
    return tuple(_row(text, name, objects) for text, objects in SUITES[name]())


def enumerate_suite(name: str) -> List[PromptRow]:
    return list(_enumerate(name))


def select(suite: str, group: str, n: int, seed: int = 0, include_unseen: bool = False) -> List[PromptRow]:
    """n prompts with distinct base prompts from one group, reproducibly."""
    rows = [r for r in _enumerate(suite) if r.group == group and (include_unseen or not r.unseen)]
    rng = random.Random(f"{suite}|{group}|{seed}")
    rng.shuffle(rows)
    chosen, seen = [], set()
    for r in rows:
        if r.base not in seen:
            chosen.append(r)
            seen.add(r.base)
            if len(chosen) == n:
                return chosen
    raise ValueError(f"suite {suite} has only {len(chosen)} base prompts in group {group}, asked for {n}")


def m1_nouns():
    v = vocab()
    return set(v["m1_tune"]) | set(v["m1_eval"]) | set(v["m1_ladder_nouns"])


def suite_nouns(name: str):
    v = vocab()
    nouns = set(v["objects"]) | set(v["test_only_objects"])
    if name == "soft":
        for c in v["contexts"]:
            nouns |= set(c.split())
    return nouns
