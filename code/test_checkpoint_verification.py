"""Regression checks for checkpoint verification (CPU, no model downloads)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuzzydiff_sd3_attention import FuzzyJointAttnProcessor
from test_processor import MaterializedReference


def attention_fixture(value):
    identity = torch.nn.Identity()
    attn = SimpleNamespace(
        heads=1, to_q=identity, to_k=identity, to_v=identity,
        add_q_proj=identity, add_k_proj=identity, add_v_proj=identity,
        norm_q=None, norm_k=None, norm_added_q=None, norm_added_k=None,
        context_pre_only=True, to_out=[identity, identity],
    )
    x = torch.full((1, 2, 1), value, dtype=torch.float64, requires_grad=True)
    ctx = torch.full((1, 2, 1), -value, dtype=torch.float64)
    return attn, x, ctx


class ReferenceShareTests(unittest.TestCase):
    def test_zero_text_mass_has_finite_zero_share_and_finite_map_gradient(self):
        attn, x, ctx = attention_fixture(1000.0)
        ref = MaterializedReference(readout={'word': [0]}, valid_text=[0, 1])
        ref(attn, x, encoder_hidden_states=ctx)
        self.assertTrue(torch.isfinite(ref.shares['word']).all().item())
        self.assertEqual(ref.shares['word'].tolist(), [0.0, 0.0])
        self.assertEqual(ref.text_mass_stats, {
            'head_query_rows': 2, 'zero_rows': 2, 'floored_rows': 2, 'nonfinite_rows': 0,
        })
        grad, = torch.autograd.grad(ref.maps['word'].sum(), x)
        self.assertTrue(torch.isfinite(grad).all().item())

    def test_very_small_mass_uses_same_guard_as_custom_processor(self):
        attn, x, ctx = attention_fixture(6.0)  # exp(-72) is nonzero but below 1e-30.
        ref = MaterializedReference(readout={'word': [0]}, valid_text=[0, 1])
        ours = FuzzyJointAttnProcessor(readout={'word': [0]}, valid_text=[0, 1])
        ref(attn, x, encoder_hidden_states=ctx)
        ours(attn, x, encoder_hidden_states=ctx)
        torch.testing.assert_close(ref.shares['word'], ours.shares['word'], rtol=1e-12, atol=1e-15)

    def test_regular_share_is_half_and_confidence_statistics_are_detached(self):
        attn, x, ctx = attention_fixture(0.0)
        ref = MaterializedReference(readout={'word': [0]}, valid_text=[0, 1])
        ref(attn, x, encoder_hidden_states=ctx)
        self.assertEqual(ref.shares['word'].tolist(), [0.5, 0.5])
        self.assertFalse(ref.shares['word'].requires_grad)
        self.assertFalse(ref.text_mass.requires_grad)


class TinyVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as out:
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            subprocess.run(
                [sys.executable, '-I', str(Path(__file__).with_name('verify_checkpoint.py')),
                 '--tiny', '--out', out],
                check=True, capture_output=True, text=True, env=env,
            )
            cls.result = json.loads((Path(out) / 'checkpoint_check_tiny.json').read_text())

    def test_finite_difference_checks_the_same_differentiable_scalar(self):
        self.assertLess(self.result['fd_grad'][1], 1e-6)

    def test_tiny_reference_shares_agree_even_for_vanishing_text_mass(self):
        self.assertLess(self.result['share_val'][0], 1e-10)

    def test_finite_difference_reports_a_step_size_sweep(self):
        sweep = self.result['fd_grad_sweep']
        self.assertEqual(len(sweep), 3)
        self.assertTrue(all(row['epsilon'] > 0 for row in sweep))
        self.assertTrue(all(abs(row['relative_error']) < 1e-5 for row in sweep))
        self.assertEqual(self.result['fd_grad'], [sweep[1]['absolute_error'], sweep[1]['relative_error']])

    def test_identity_reports_repeated_noise_and_custom_comparisons(self):
        self.assertEqual(len(self.result['identity_grad_noise_floor_samples']), 2)
        self.assertEqual(len(self.result['identity_grad_samples']), 3)
        self.assertEqual(self.result['identity_grad'], [0.0, 0.0])

    def test_reference_reports_zero_and_floored_text_mass_counts(self):
        stats = self.result['text_mass_diagnostics']
        self.assertEqual(len(stats), 3)
        self.assertTrue(all(row['head_query_rows'] > 0 for row in stats.values()))
        self.assertTrue(all(row['nonfinite_rows'] == 0 for row in stats.values()))


if __name__ == '__main__':
    unittest.main()
