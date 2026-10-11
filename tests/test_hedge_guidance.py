"""Experimental hedge objectives, tested without downloading weights."""
import unittest
import tempfile
from pathlib import Path
import torch
from test_fullpipeline import load_definitions


class HedgeGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.code = load_definitions()

    def test_detection_requires_binding_and_word_boundaries(self):
        resolve = self.code['resolve_hedge_targets']
        groups = {'Mostly blue': [1, 2], 'vase': [3], 'almostly green': [4]}
        result = resolve(groups, [('Mostly blue', 'bound_to', 'vase')])
        self.assertEqual(list(result), ['Mostly blue'])
        self.assertEqual(result['Mostly blue']['target'], 0.85)
        with self.assertRaisesRegex(ValueError, 'binding'):
            resolve(groups, [])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            resolve({'slightly dusty car': [1, 2, 3], 'car': [3]},
                    [('slightly dusty car', 'bound_to', 'car')])

    def test_multiword_hedges_and_comparatives(self):
        resolve = self.code['resolve_hedge_targets']
        groups = {'a little dusty': [1, 2, 3], 'car': [4], 'slightly larger red ball': [5, 6, 7]}
        result = resolve(groups, [('a little dusty', 'bound_to', 'car'),
                                 ('slightly larger red ball', 'slightly_larger_than', 'car')])
        self.assertEqual(list(result), ['a little dusty'])
        self.assertEqual(result['a little dusty']['target'], 0.10)

    def test_band_gradient_moves_toward_target_and_leaves_band_alone(self):
        fn = self.code['hedge_band_loss']
        for value, sign in [(0.05, -1), (0.5, 1), (0.15, 0)]:
            truth = torch.tensor(value, requires_grad=True)
            fn(truth, 0.15).backward()
            self.assertEqual(int(truth.grad.sign()), sign)

    def test_target_replaces_attribute_presence_but_preserves_object_presence(self):
        attention = torch.tensor([[[0.5, 0.5]] * 4], requires_grad=True)
        targets = {'slightly dusty': {'tokens': [0], 'target': 0.15}}
        loss = self.code['compute_fuzzy_loss']({'down_cross': attention}, [[0], [1]],
                    membership_sharpness=1, t_norm='product', hedge_targets=targets)
        loss.backward()
        self.assertTrue(torch.isfinite(attention.grad).all())
        self.assertGreater(attention.grad[..., 0].sum().item(), 0)
        self.assertLess(attention.grad[..., 1].sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'share'):
            self.code['compute_fuzzy_loss']({'down_cross': attention}, [[0], [1]],
                    membership_mode='relative', hedge_targets=targets)

    def test_refinement_residual_is_signed_and_zero_in_band(self):
        residual = self.code['hedge_refinement_weights']
        targets = {'dusty': {'target': 0.15}}
        self.assertLess(residual({'dusty': 0.5, 'car': 0.3}, targets)['dusty'], 0)
        self.assertGreater(residual({'dusty': 0.01}, targets)['dusty'], 0)
        self.assertEqual(residual({'dusty': 0.15}, targets)['dusty'], 0)
        self.assertAlmostEqual(residual({'car': 0.3}, {})['car'], 0.7)

    def test_negative_refinement_is_finite_and_preserves_unconditional_attention(self):
        processor = self.code['PredicateRefinerProcessor'](
            {1: torch.ones(2, 2)}, weights={1: -0.5}, cfg=True)
        probabilities = torch.full((2, 4, 3), 1 / 3)
        changed = processor.modify_attention(probabilities, batch_size=2, heads=1)
        self.assertTrue(torch.equal(changed[0], probabilities[0]))
        self.assertTrue((changed[1, :, 1] < probabilities[1, :, 1]).all())
        self.assertTrue(torch.isfinite(changed).all())
        self.assertTrue(torch.allclose(changed.sum(-1), torch.ones(2, 4)))

    def test_sd15_sampling_consumes_targets_and_changes_seeded_output(self):
        from test_sd_fuzzy import tiny_pipeline
        import run_sd
        fuzzy = run_sd.load_fuzzy_definitions()
        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pipeline, tokenizer = tiny_pipeline(path, fuzzy,
                extra_tokens=['v', 'y', 'y</w>', 'd</w>', 'k</w>'])
            cfg, groups = run_sd.build_config(fuzzy, tokenizer, 'a very red book',
                ['very red', 'book'], 42, path, steps=2, size=64,
                sharpness=5, learning_rate=5, updates=2, t_norm='product')
            relations = run_sd.resolve_bindings(groups, [('very red', 'book')])
            plain, _ = run_sd.generate_fuzzy(pipeline, fuzzy, cfg, groups, relations=relations)
            cfg.hedge_guidance = True
            guided, store = run_sd.generate_fuzzy(pipeline, fuzzy, cfg, groups, relations=relations)
            self.assertEqual(store.hedge_targets['very red']['target'], 0.85)
            self.assertGreater(store.guidance_diagnostics[0]['update_rms'], 0)
            self.assertNotEqual(plain.tobytes(), guided.tobytes())
            repeat, _ = run_sd.generate_fuzzy(pipeline, fuzzy, cfg, groups, relations=relations)
            self.assertEqual(repeat.tobytes(), guided.tobytes())


if __name__ == '__main__':
    unittest.main()
