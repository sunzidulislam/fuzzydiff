"""Tests of prompts.py (python -I test_prompts.py)."""
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import assign_group, base_prompt, enumerate_suite, m1_nouns, select, suite_nouns  # noqa: E402


def test_hedge_does_not_change_base():
    assert base_prompt("a slightly red mug") == base_prompt("a very red mug") == "a red mug"
    assert base_prompt("an extremely red mug") == "a red mug"
    assert base_prompt("a barely orange mug") == "an orange mug"
    assert base_prompt("a kitten slightly larger than a pillow") == "a kitten larger than a pillow"


def test_groups_are_stable_and_disjoint():
    assert assign_group("a red mug") == assign_group("a red mug")
    rows = enumerate_suite("soft")
    by = defaultdict(set)
    for r in rows:
        by[r.base].add(r.group)
    assert all(len(g) == 1 for g in by.values())


def test_each_setting_has_400_test_bases_and_other_groups():
    for s in ("soft", "mixture", "size"):
        rows = enumerate_suite(s)
        main = {r.base for r in rows if r.group == "test" and not r.unseen}
        assert len(main) >= 400, s
        assert {"cal", "int", "sys", "cons", "dev"} <= {r.group for r in rows}


def test_no_base_prompt_crosses_suites():
    bases = {s: {r.base for r in enumerate_suite(s)} for s in ("soft", "mixture", "size", "plain_color")}
    names = list(bases)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not bases[a] & bases[b], (a, b)


def test_test_only_objects_and_parser():
    for suite in ("mixture", "soft", "size"):
        rows = enumerate_suite(suite)
        assert any(r.unseen for r in rows)
        assert all(r.group == "test" for r in rows if r.unseen)
        sample = random.Random(0).sample(rows, 300)
        assert not any(r.program.degraded for r in sample), suite


def test_suite_programs_have_expected_families():
    r = next(r for r in enumerate_suite("mixture"))
    assert {"F6", "presence", "binding", "leakage"} <= {c.family for c in r.program.constraints}
    r = next(r for r in enumerate_suite("size"))
    assert "F8" in {c.family for c in r.program.constraints}
    assert {"presence", "binding", "leakage"} <= {c.family for c in enumerate_suite("plain_color")[0].program.constraints}


def test_size_prompts_only_ask_for_drawable_relations():
    from prompts import size_class, vocab
    v = vocab()
    assert all(any(o in m for m in v["size_classes"].values()) for o in v["objects"] + v["test_only_objects"])
    for suite in ("size", "ladder_size"):
        for r in enumerate_suite(suite):
            c = next(c for c in r.program.constraints if c.family == "F8")
            gap = size_class(c.args[0]) - size_class(c.args[1])
            assert abs(gap) <= 1 and (gap == 0 or (gap > 0) == (c.orientation > 0)), r.prompt
    prompts = {r.prompt for r in enumerate_suite("ladder_size")}
    assert "a pumpkin barely larger than a mug" in prompts and "a mug barely larger than a pumpkin" not in prompts
    assert not any(p.startswith("a kitten") and p.endswith("than a tent") and "larger" in p for p in prompts)


def test_m1_nouns_are_isolated():
    n = m1_nouns()
    others = suite_nouns("soft") | suite_nouns("mixture") | suite_nouns("size")
    assert not (n & others) and all(r.group == "dev" for r in enumerate_suite("m1_eval"))
    assert not ({r.base for r in enumerate_suite("m1_tune")} & {r.base for r in enumerate_suite("m1_eval")})
    assert len(enumerate_suite("m1_tune")) == 5 and len(enumerate_suite("m1_eval")) == 20
    assert {r.prompt for r in enumerate_suite("m1_ladder") if "bedroom" in r.prompt} == {
        "a dimly lit bedroom", "a moderately lit bedroom", "a brightly lit bedroom"}


def test_select_is_seeded_group_limited_and_excludes_unseen():
    a = select("mixture", "dev", 20, seed=1)
    assert [r.prompt for r in a] == [r.prompt for r in select("mixture", "dev", 20, seed=1)]
    assert len(a) == 20 and len({r.base for r in a}) == 20
    assert all(r.group == "dev" and not r.unseen for r in a)
    try:
        select("m1_tune", "dev", 6)
        assert False, "asking for more prompts than exist must fail"
    except ValueError:
        pass


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("prompts: pass")


if __name__ == "__main__":
    main()
