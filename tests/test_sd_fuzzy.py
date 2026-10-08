"""CPU regression tests for the SD 1.x backbone runner; no model downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import four_way
import run_sd

PROMPT = 'a red book'
WORDS = ['red book']


def tiny_pipeline(path, fuzzy):
    """A randomly initialized SD 1.x stack: real modules, no weights to download."""
    vocab = {'<|startoftext|>': 0, '<|endoftext|>': 1}
    for token in ['a', 'r', 'e', 'd', 'b', 'o', 'k', 'a</w>', 'red</w>', 'book</w>']:
        vocab[token] = len(vocab)
    (path / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    (path / 'merges.txt').write_text('#version: 0.2\n', encoding='utf-8')
    tokenizer = CLIPTokenizer(vocab_file=str(path / 'vocab.json'),
                              merges_file=str(path / 'merges.txt'), model_max_length=16)
    text_config = CLIPTextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=1, num_attention_heads=4,
                                 max_position_embeddings=16, bos_token_id=0, eos_token_id=1,
                                 pad_token_id=1)
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
                                down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
                                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
                                block_out_channels=(16, 32), layers_per_block=1,
                                cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=('DownEncoderBlock2D',) * 4,
                        up_block_types=('UpDecoderBlock2D',) * 4,
                        block_out_channels=(16,) * 4, layers_per_block=1,
                        norm_num_groups=8, sample_size=64)
    pipeline = run_sd.FuzzyStableDiffusionPipeline(
        vae=vae, text_encoder=CLIPTextModel(text_config), tokenizer=tokenizer, unet=unet,
        scheduler=DPMSolverMultistepScheduler(), safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    run_sd.configure(pipeline, torch.device('cpu'), fuzzy)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, tokenizer


class StableDiffusionBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cls.fuzzy = run_sd.load_fuzzy_definitions()

    def test_notebook_supplies_the_fuzzy_definitions(self):
        # The backbones share one implementation; a renamed cell must fail loudly here.
        for name in ('RunConfig', 'soft_truth', 'phrase_membership', 'compute_fuzzy_loss',
                     'phrase_truth_scores', 'AttentionStore', 'AttendExciteAttnProcessor',
                     'get_token_groups'):
            self.assertIn(name, self.fuzzy)

    def test_missing_words_flags_only_absent_phrases(self):
        with tempfile.TemporaryDirectory() as directory:
            _, tokenizer = tiny_pipeline(Path(directory), self.fuzzy)
        self.assertEqual(run_sd.missing_words(tokenizer, PROMPT, WORDS), [])
        self.assertEqual(run_sd.missing_words(tokenizer, PROMPT, ['red book', 'oak']), ['oak'])

    def test_bindings_parse_and_resolve_to_token_groups(self):
        self.assertEqual(run_sd.parse_bindings(['yellow>clock', ' red > book ']),
                         [('yellow', 'clock'), ('red', 'book')])
        for bad in ('yellowclock', '>clock', 'yellow>'):
            with self.assertRaises(ValueError):
                run_sd.parse_bindings([bad])
        groups = {'yellow': [1], 'clock': [2, 3]}
        self.assertEqual(run_sd.resolve_bindings(groups, [('yellow', 'clock')]),
                         [([1], 'bound_to', [2, 3])])
        with self.assertRaises(ValueError):
            run_sd.resolve_bindings(groups, [('yellow', 'book')])

    def test_guidance_changes_a_seeded_image_and_records_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pipeline, tokenizer = tiny_pipeline(path, self.fuzzy)
            # Stock diffusers sampling stays reachable through the inherited __call__,
            # so one checkpoint serves the plain and the guided arm.
            plain = run_sd.generate_plain(pipeline, PROMPT, seed=42, steps=2, guidance=7.5, size=64)
            self.assertEqual(plain.size, (64, 64))
            cfg, groups = run_sd.build_config(self.fuzzy, tokenizer, PROMPT, WORDS, 42, path,
                                              steps=2, guidance=7.5, size=64,
                                              learning_rate=0.2, updates=2)
            self.assertEqual(cfg.text_span, (1, len(tokenizer.encode(PROMPT)) - 1))
            self.assertEqual(cfg.token_groups, [groups['red book']])
            guided, store = run_sd.generate_fuzzy(pipeline, self.fuzzy, cfg, groups)
            self.assertEqual(guided.size, (64, 64))
            self.assertNotEqual(list(plain.getdata()), list(guided.getdata()),
                                'Fuzzy guidance did not change the image')
            self.assertEqual(len(store.guidance_diagnostics), 2)
            self.assertEqual(sorted(store.guidance_diagnostics[0]),
                             ['clipped', 'gradient_rms', 'latent_rms', 'learning_rate',
                              'loss', 'step', 'update_ratio', 'update_rms'])
            truths = self.fuzzy['phrase_truth_scores'](store, cfg.alpha, cfg.membership_sharpness)
            self.assertEqual(sorted(truths), WORDS)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in truths.values()))
            repeated, _ = run_sd.generate_fuzzy(pipeline, self.fuzzy, cfg, groups)
            self.assertEqual(list(guided.getdata()), list(repeated.getdata()),
                             'Seeded guided sampling was not repeatable')
            cfg.membership_mode = 'relative'
            relative, relative_store = run_sd.generate_fuzzy(pipeline, self.fuzzy, cfg, groups)
            self.assertEqual(relative.size, (64, 64))
            self.assertEqual(relative_store.membership_mode, 'relative')
            self.assertGreater(relative_store.guidance_diagnostics[0]['update_rms'], 0)
            self.assertNotAlmostEqual(relative_store.guidance_diagnostics[0]['loss'],
                                      store.guidance_diagnostics[0]['loss'], places=5,
                                      msg='SD 1.x ignored the selected membership mode')

    def test_four_way_grid_renders_every_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pipeline, _ = tiny_pipeline(path, self.fuzzy)
            image = run_sd.generate_plain(pipeline, PROMPT, seed=42, steps=1, guidance=1.0, size=64)
            images = {name: image for row in four_way.GRID for name in row}
            scores = {name: {'clip': {'min_part': 0.25}} for name in images}
            grid = four_way.save_grid(path, images, scores, PROMPT)
            self.assertTrue(grid.is_file())
            self.assertGreater(grid.stat().st_size, 0)

    def test_four_way_results_keep_display_order_separate_from_measured_rank(self):
        report = {'prompt': 'A slightly dusty red sports car parked on a road.', 'seed': 142,
                  'stage': 'base only, no refinement',
                  'settings': {'membership_mode': 'relative', 'fuzzy_lr': 0.2},
                  'runs': {name: {'clip': {'full': full, 'min_part': part}}
                           for name, full, part in [('sdxl_fuzzy', 0.30, 0.20),
                                                   ('sdxl_plain', 0.29, 0.21),
                                                   ('sd15_plain', 0.28, 0.19),
                                                   ('sd15_fuzzy', 0.31, 0.22)]},
                  'ranking_by_min_part': ['sd15_fuzzy', 'sdxl_plain', 'sdxl_fuzzy', 'sd15_plain'],
                  'within_backbone_delta': {'sdxl': -0.01, 'sd15': 0.03}}
        with tempfile.TemporaryDirectory() as directory:
            path = four_way.save_results(Path(directory), report)
            text = path.read_text(encoding='utf-8')
        rows = [line for line in text.splitlines() if line.startswith('| ')][2:]
        self.assertIn('Ours on SDXL + fuzzy | 3 |', rows[0])
        self.assertIn('Ours on SDXL, fuzzy off | 2 |', rows[1])
        self.assertIn('SD 1.5 backbone only | 4 |', rows[2])
        self.assertIn('SD 1.5 + our fuzzy pipeline | 1 |', rows[3])
        self.assertIn('not a quality ranking', text)


if __name__ == '__main__':
    unittest.main()
