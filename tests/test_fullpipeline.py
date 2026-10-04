"""CPU regression tests for notebook definitions; no model downloads."""
import abc
import json
import math
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate

NOTEBOOK = Path(__file__).resolve().parents[1] / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb'

def load_definitions(base_class=object):
    namespace = dict(globals(), StableDiffusionXLPipeline=base_class)
    prefixes = ('@dataclass', 'def soft_truth', 'def compute_fuzzy_loss',
                'class AttentionControl', 'class AttentionStore',
                'class AttendExciteAttnProcessor', 'class FuzzyAttendExciteSDXLPipeline',
                'def get_token_indices', 'def get_token_groups',
                'def build_object_masks', 'class PredicateRefinerProcessor')
    for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']:
        source = cell['source']
        source = ''.join(source) if isinstance(source, list) else source
        if cell['cell_type'] == 'code' and source.startswith(prefixes):
            exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
    return namespace

class NotebookRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.code = load_definitions()

    def test_guidance_attention_retains_gradient_to_latents(self):
        store = self.code['AttentionStore'](attn_res=2)
        if hasattr(store, 'begin_forward'):
            store.begin_forward(keep_grad=True)
        latents = torch.randn(1, 4, 3, requires_grad=True)
        store.forward(latents.softmax(-1), True, 'down')
        loss = self.code['compute_fuzzy_loss'](store.get_current_step_attention(), [1])
        self.assertTrue(loss.requires_grad, 'Attention disconnected fuzzy loss from latents')
        gradient = torch.autograd.grad(loss, latents)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertFalse(torch.equal(latents, latents - 0.05 * gradient))

    def test_unrelated_words_do_not_receive_spatial_constraints(self):
        maps = {'down_cross': torch.full((1, 4, 2), 0.5)}
        loss = self.code['compute_fuzzy_loss'](maps, [0, 1])
        self.assertAlmostEqual(loss.item(), 0.69314718, places=5)

    def test_attention_aggregation_handles_small_and_large_maps(self):
        store = self.code['AttentionStore'](attn_res=4)
        if hasattr(store, 'begin_forward'):
            store.begin_forward()
        store.forward(torch.ones(2, 4, 3), True, 'down')
        store.forward(torch.ones(2, 64, 3), True, 'down')
        try:
            maps = store.get_current_step_attention()
        except RuntimeError as error:
            self.fail(f'UNet resolutions could not be aggregated: {error}')
        self.assertEqual(tuple(maps['down_cross'].shape), (1, 16, 3))

    def test_tracking_covers_split_words_and_repeated_occurrences(self):
        class Tokenizer:
            model_max_length = 77
            def encode(self, text, add_special_tokens=True):
                ids = {'redwood': [10, 11], 'redwood and redwood': [10, 11, 12, 10, 11]}[text.lower()]
                return [0, *ids, 99] if add_special_tokens else ids
            def convert_ids_to_tokens(self, ids):
                return [{0: '<start>', 10: 'red', 11: 'wood</w>', 12: 'and</w>', 99: '<end>'}[i] for i in ids]
            def tokenize(self, text):
                return ['red', 'wood</w>']
        indices = self.code['get_token_indices'](Tokenizer(), 'redwood and redwood', ['redwood'])
        self.assertEqual(indices, [1, 2, 4, 5])

    def test_cfg_collection_uses_only_conditional_attention(self):
        from diffusers.models.attention_processor import Attention
        torch.manual_seed(3)
        attention = Attention(query_dim=8, cross_attention_dim=8, heads=4, dim_head=2)
        hidden = torch.randn(2, 4, 8)
        context = torch.randn(2, 3, 8)
        store = self.code['AttentionStore'](attn_res=2)
        if hasattr(store, 'begin_forward'):
            store.begin_forward(cfg=True)
        processor = self.code['AttendExciteAttnProcessor'](store, 'down')
        processor.fallback = attention.processor
        processor(attention, hidden, encoder_hidden_states=context)
        cfg_map = store.get_current_step_attention()['down_cross']
        conditional_store = self.code['AttentionStore'](attn_res=2)
        conditional_processor = self.code['AttendExciteAttnProcessor'](conditional_store, 'down')
        conditional_processor.fallback = attention.processor
        conditional_processor(attention, hidden[1:], encoder_hidden_states=context[1:])
        conditional_map = conditional_store.get_current_step_attention()['down_cross']
        self.assertTrue(torch.allclose(cfg_map, conditional_map, atol=1e-6),
                        'Unconditional attention contaminated the predicate maps')

    def test_refiner_keeps_unconditional_branch_unchanged(self):
        from diffusers.models.attention_processor import Attention
        torch.manual_seed(4)
        attention = Attention(query_dim=8, cross_attention_dim=8, heads=4, dim_head=2)
        hidden = torch.randn(2, 4, 8)
        context = torch.randn(2, 3, 8)
        baseline = attention(hidden, encoder_hidden_states=context)
        processor = self.code['PredicateRefinerProcessor']({1: torch.tensor([[1., 0.], [0., 1.]])})
        processor.fallback = attention.processor
        modified = processor(attention, hidden, encoder_hidden_states=context)
        self.assertTrue(torch.allclose(modified[0], baseline[0], atol=1e-6),
                        'Predicate refinement modified CFG negative conditioning')
        self.assertFalse(torch.allclose(modified[1], baseline[1], atol=1e-6))

    def test_checkpointed_sdxl_guidance_changes_seeded_denoising(self):
        # Real randomly initialized SDXL components exercise the complete latent loop.
        from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
        from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        code = load_definitions(StableDiffusionXLPipeline)
        code['StableDiffusionXLPipelineOutput'] = StableDiffusionXLPipelineOutput
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            vocab = {'<|startoftext|>': 0, '<|endoftext|>': 1}
            for token in ['a', 'c', 't', 'a</w>', 'c</w>', 't</w>']:
                vocab[token] = len(vocab)
            (path / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
            (path / 'merges.txt').write_text('#version: 0.2\n', encoding='utf-8')
            tokenizer = CLIPTokenizer(vocab_file=str(path / 'vocab.json'), merges_file=str(path / 'merges.txt'),
                                      model_max_length=16)
            text_config = CLIPTextConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=16, projection_dim=8,
                bos_token_id=0, eos_token_id=1, pad_token_id=1)
            unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
                down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels=(16, 32),
                layers_per_block=1, cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8,
                addition_embed_type='text_time', addition_time_embed_dim=2,
                projection_class_embeddings_input_dim=20)
            vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                down_block_types=('DownEncoderBlock2D',) * 4,
                up_block_types=('UpDecoderBlock2D',) * 4, block_out_channels=(16,) * 4,
                layers_per_block=1, norm_num_groups=8, sample_size=64)
            pipeline = code['FuzzyAttendExciteSDXLPipeline'](unet=unet, vae=vae,
                text_encoder=CLIPTextModel(text_config), text_encoder_2=CLIPTextModelWithProjection(text_config),
                tokenizer=tokenizer, tokenizer_2=tokenizer, scheduler=DPMSolverMultistepScheduler(),
                add_watermarker=False)
            for component in (unet, vae, pipeline.text_encoder, pipeline.text_encoder_2):
                component.eval().requires_grad_(False)
            unet.enable_gradient_checkpointing()
            indices = code['get_token_indices'](tokenizer, 'a cat', ['cat'])

            def sample(updates):
                config = code['RunConfig'](prompt='a cat', height=64, width=64, n_inference_steps=2,
                    max_iter_to_alter=updates, output_path=path)
                store = code['AttentionStore'](attn_res=4)
                pipeline.register_attention_control(store)
                output = pipeline(prompt='a cat', height=64, width=64, num_inference_steps=2,
                    guidance_scale=2, generator=torch.Generator().manual_seed(9), cfg=config,
                    attention_store=store, token_indices=indices, output_type='latent')
                return output.images, store

            baseline, _ = sample(0)
            guided, store = sample(1)
            repeated, _ = sample(1)
            self.assertTrue(torch.isfinite(guided).all())
            self.assertFalse(torch.allclose(baseline, guided), 'Guidance did not affect denoising')
            self.assertTrue(torch.equal(guided, repeated), 'Seeded sampling was not repeatable')
            self.assertEqual(len(store.guidance_diagnostics), 1)
            self.assertGreater(store.guidance_diagnostics[0]['update_rms'], 0)
            self.assertFalse(store.step_store, 'Guidance graphs remained in the collector')
            self.assertTrue(all(not value.requires_grad and value.device.type == 'cpu'
                                for value in store.get_average_attention().values()))
            # Exercise inherited VAE decode/postprocessing as well as the latent path.
            config = code['RunConfig'](height=64, width=64, n_inference_steps=1, max_iter_to_alter=0, output_path=path)
            result = pipeline(prompt='a cat', height=64, width=64, num_inference_steps=1, guidance_scale=1,
                              generator=torch.Generator().manual_seed(9), cfg=config)
            self.assertEqual(result.images[0].size, (64, 64))

if __name__ == '__main__':
    unittest.main()
