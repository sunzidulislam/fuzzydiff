"""Tests of threshold_probe.py and the provisional values it produces (python -I test_probe.py)."""
import os
import random
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from estimators import FakeSigLIP, FakeSupport  # noqa: E402
from grounding import phrase_spans  # noqa: E402
from measure import EstimatorDeps, Measure  # noqa: E402
from predicates import translate  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402
from threshold_probe import (absent_objects, attention_rows, probe_image, probe_prompts, settings_yaml,  # noqa: E402
                             summarize)

PROMPT = "a red mug and a blue teapot"


def _rows(separated=True, n=40):
    rng = random.Random(0)
    rows = []
    for i in range(n):
        base = f"prompt {i}"
        for label in (1, 0):
            centre = (2.0 if label else -6.0) if separated else 0.0
            rows.append({"type": "presence", "bin": 3, "object": "x", "label": label, "base": base,
                         "global": centre + rng.gauss(0, 1), "region": centre + rng.gauss(0, 1)})
    return rows


def _attention(share, correct, kind="objects", base="p"):
    return {"type": "attention", "kind": kind, "config": "read_blocks", "peak": 5.0, "share": share, "mass": 0.05,
            "correct": correct, "base": base}


def test_prompts_are_development_only_and_of_the_requested_number():
    rows = probe_prompts(20)
    assert len(rows) == 20 and len({r.prompt for r in rows}) == 20
    assert {r.group for r in rows} == {"dev"}
    assert {r.suite for r in rows} == {"plain_color", "mixture"}


def test_absent_objects_are_repeatable_and_never_named_in_the_prompt():
    chosen = absent_objects(PROMPT, translate(PROMPT), 3, seed=0)
    assert chosen == absent_objects(PROMPT, translate(PROMPT), 3, seed=0)
    assert len(chosen) == 3 and not {"mug", "teapot"} & set(chosen)


def test_attention_peak_is_judged_against_the_object_in_the_final_image():
    P_on, P_off = torch.zeros(4, 4), torch.zeros(4, 4)
    P_on[1, 2], P_off[3, 3] = 1.0, 1.0
    S = torch.zeros(8, 8)
    S[2:4, 4:6] = 1.0                               # grid cell (1, 2) of the coarser map
    attention = {"mug": {"P": P_on, "peak": 9.0, "share": 0.3, "mass": 0.05},
                 "red": {"P": P_off, "peak": 4.0, "share": 0.2, "mass": 0.05}}
    rows = {r["kind"]: r for r in attention_rows(attention, {"mug": S}, {"red": "mug"}, "base")}
    assert rows["objects"]["correct"] is True and rows["objects"]["share"] == 0.3
    assert rows["attributes"]["correct"] is False and rows["attributes"]["peak"] == 4.0


def test_summary_finds_separable_presence_and_reports_when_there_is_none():
    good = summarize(_rows(True), [1, 2], 0.8)["presence"]["3"]
    assert good["auc"] > 0.95 and good["present_pass"] > 0.9 and good["absent_pass"] < 0.1
    assert good["model"]["w_global"] > 0
    bad = summarize(_rows(False), [1, 2], 0.8)["presence"]["3"]
    assert bad["auc"] < 0.75


def test_summary_chooses_the_widest_attention_threshold_that_stays_accurate():
    rows = []
    for i in range(20):
        base = f"prompt {i}"
        rows += [_attention(0.3, True, base=base), _attention(0.1, False, base=base),
                 _attention(0.3, True, "attributes", base), _attention(0.1, False, "attributes", base)]
    chosen = summarize(rows, [0.05, 0.2, 0.5], 0.8)["attention"]
    assert chosen["objects"]["feasible"] and chosen["objects"]["m_min"] == 0.2
    useless = summarize([_attention(0.3, i % 2 == 0, base=f"prompt {i}") for i in range(40)], [0.05, 0.2], 0.8)["attention"]
    assert useless["objects"]["feasible"] is False


def test_summary_values_reach_the_provisional_calibration():
    rows = _rows(True)
    for i in range(20):
        base = f"prompt {i}"
        rows += [_attention(0.3, True, base=base), _attention(0.1, False, base=base),
                 _attention(0.3, True, "attributes", base), _attention(0.1, False, "attributes", base)]
    values = yaml.safe_load(settings_yaml(summarize(rows, [0.05, 0.2, 0.5], 0.8)))
    s = load_settings()
    s.provisional.update(values)
    calib = ProvisionalCalibration(s)
    assert calib.attention_thresholds("objects")[1] == 0.2 and "attention_m_min" in s.provisional_used
    present = calib.presence_logit(3, torch.tensor(2.0), torch.tensor(2.0))
    absent = calib.presence_logit(3, torch.tensor(-6.0), torch.tensor(-6.0))
    assert present > 0 > absent and "presence_models" in s.provisional_used
    assert calib.presence_logit(1, torch.tensor(2.0), torch.tensor(2.0)) == present      # nearest fitted bin


def test_probe_records_present_and_absent_objects_at_noisy_steps_and_the_final_image():
    def mask(phrase, x):
        return torch.ones(8, 8)
    s = load_settings()
    calib = ProvisionalCalibration(s)
    deps = EstimatorDeps(FakeSupport(mask), FakeSigLIP(lambda phrase, image: image.mean()), calib, s)
    b = tiny_backend()
    program = translate(PROMPT)
    cond = b.encode_prompt(PROMPT, phrase_spans(PROMPT, program))
    image, rows = probe_image(b, Measure(deps, PROMPT, cond), cond, PROMPT, program, ["vase"], seed=0, steps=4,
                              every=1, sigma_lo=0.0, sigma_hi=1.0)
    presence = [r for r in rows if r["type"] == "presence"]
    assert {(r["object"], r["label"]) for r in presence} == {("mug", 1), ("teapot", 1), ("vase", 0)}
    assert len({r["bin"] for r in presence}) >= 2 and all(r["base"] == PROMPT for r in presence)
    attention = [r for r in rows if r["type"] == "attention"]
    assert {r["kind"] for r in attention} == {"objects", "attributes"} and all(r["correct"] is True for r in attention)
    assert image.numel() > 0


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("probe: pass")


if __name__ == "__main__":
    main()
