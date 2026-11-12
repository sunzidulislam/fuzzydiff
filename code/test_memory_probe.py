"""Tests of memory_probe.py (python -I test_memory_probe.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_probe import PROMPT, module_gib, no_checkpointing, probe, report, timed  # noqa: E402
from predicates import translate  # noqa: E402
from grounding import phrase_spans  # noqa: E402
from run_experiment import _tiny_estimators  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402


def test_module_gib_counts_parameters_and_buffers_once():
    m = torch.nn.Linear(4, 4, bias=False)
    m.register_buffer("b", torch.zeros(4))

    class Holder:
        def __init__(self):
            self.model, self.again, self.size = m, m, 3

    expected = (16 + 4) * 4 / 2 ** 30
    assert abs(module_gib(m) - expected) < 1e-15 and abs(module_gib(Holder()) - expected) < 1e-15


def test_timed_reports_a_median_time():
    out = timed(lambda: sum(range(1000)), repeats=3)
    assert out["time_s"] >= 0 and "peak_gb" in out and "activation_gb" in out


def test_no_checkpointing_restores_both_switches():
    backend = tiny_backend()
    original = torch.utils.checkpoint.checkpoint
    with no_checkpointing(backend):
        assert torch.utils.checkpoint.checkpoint is not original
        assert not backend.transformer.is_gradient_checkpointing
    assert torch.utils.checkpoint.checkpoint is original
    assert backend.transformer.is_gradient_checkpointing and backend.transformer.training


def test_probe_runs_every_measurement_on_the_tiny_model():
    s = load_settings()
    backend = tiny_backend()
    seg, siglip = _tiny_estimators()
    backend.encode_prompt(PROMPT, phrase_spans(PROMPT, translate(PROMPT)))
    result = probe(backend, seg, siglip, s, repeats=1, guided_steps=3)
    for key in ("unguided_step", "readout_forward", "vae_decode", "mmdit_backward", "vae_backward",
                "naive_mmdit_backward", "naive_vae_backward", "naive_guided_step", "sar_step"):
        assert "time_s" in result[key], key
    assert result["weights_gb"]["mmdit"] > 0 and len(result["guided_steps"]) == 3
    assert "guided step, ours" in report(result)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("memory_probe: pass")


if __name__ == "__main__":
    main()
