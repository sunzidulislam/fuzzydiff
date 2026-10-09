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
                'class AttentionStore',
                'class AttendExciteAttnProcessor', 'class FuzzyAttendExciteSDXLPipeline',
                'def get_token_indices', 'def get_token_groups',
                'def build_phrase_membership_masks', 'class PredicateRefinerProcessor')
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
        # A full 77-token context keeps attention at its real per-token scale;
        # a 3-token stand-in saturates the membership softmax and hides gradients.
        latents = torch.randn(1, 4, 77, requires_grad=True)
        store.forward(latents.softmax(-1), True, 'down')
        loss = self.code['compute_fuzzy_loss'](store.get_current_step_attention(), [1],
                                               text_span=(1, 76))
        self.assertTrue(loss.requires_grad, 'Attention disconnected fuzzy loss from latents')
        gradient = torch.autograd.grad(loss, latents)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertFalse(torch.equal(latents, latents - 0.05 * gradient))

    def test_phrase_membership_is_graded_rather_than_context_scaled(self):
        # Pooling raw attention returned ~1/context for present and absent phrases
        # alike; membership has to separate them inside [0, 1].
        context = 16
        logits = torch.full((1, 4, context), -2.0)
        logits[..., 1] = 4.0
        attention = logits.softmax(-1)
        span = (1, context - 1)
        present = self.code['phrase_membership'](attention, [1], span)
        absent = self.code['phrase_membership'](attention, [2], span)
        self.assertGreaterEqual(present.min().item(), 0.0)
        self.assertLessEqual(present.max().item(), 1.0)
        self.assertGreater(self.code['soft_truth'](present).item(), 0.9)
        self.assertLess(self.code['soft_truth'](absent).item(), 0.1)

    def test_phrase_pools_its_subtokens_into_one_predicate(self):
        # Each half of the image is grounded by a different subtoken; the phrase
        # must cover both halves rather than compete with itself.
        context = 8
        logits = torch.full((1, 4, context), -2.0)
        logits[:, :2, 1] = 2.0
        logits[:, 2:, 2] = 2.0
        attention = logits.softmax(-1)
        span = (1, context - 1)
        phrase = self.code['phrase_membership'](attention, [1, 2], span)
        subtoken = self.code['phrase_membership'](attention, [1], span)
        self.assertGreater(phrase.min().item(), 0.9)
        self.assertLess(subtoken.min().item(), 0.1)
        self.assertLessEqual(phrase.max().item(), 1.0)

    def test_phrase_truth_scores_read_the_store_the_way_generate_fills_it(self):
        # generate() records {phrase: indices} plus a text span on the store; the
        # metadata and refinement weights are read back through that shape.
        context = 12
        store = self.code['AttentionStore'](attn_res=2)
        store.token_groups = {'fast car': [1, 2], 'tree': [3]}
        store.text_span = (1, context - 1)
        logits = torch.full((1, 4, context), -2.0)
        logits[:, :2, 1] = 4.0
        logits[:, 2:, 2] = 4.0
        store.begin_forward()
        store.forward(logits.softmax(-1).expand(2, -1, -1), True, 'down')
        store.end_forward()
        truths = self.code['phrase_truth_scores'](store)
        self.assertEqual(sorted(truths), ['fast car', 'tree'])
        self.assertGreater(truths['fast car'], 0.9)
        self.assertLess(truths['tree'], 0.1)
        masks = self.code['build_phrase_membership_masks'](store, torch.device('cpu'), latent_size=4)
        self.assertEqual(sorted(masks), ['fast car'])
        self.assertEqual(tuple(masks['fast car'].shape), (4, 4))
        self.assertLessEqual(masks['fast car'].max().item(), 1.0)

    def test_refinement_leaves_well_grounded_phrases_unchanged(self):
        from diffusers.models.attention_processor import Attention
        torch.manual_seed(6)
        attention = Attention(query_dim=8, cross_attention_dim=8, heads=4, dim_head=2)
        hidden = torch.randn(2, 4, 8)
        context = torch.randn(2, 3, 8)
        baseline = attention(hidden, encoder_hidden_states=context)
        masks = {1: torch.tensor([[1., 0.], [0., 1.]])}
        processor = self.code['PredicateRefinerProcessor'](masks, weights={1: 0.0})
        processor.fallback = attention.processor
        grounded = processor(attention, hidden, encoder_hidden_states=context)
        self.assertTrue(torch.allclose(grounded, baseline, atol=1e-6),
                        'A fully grounded phrase was still reweighted')

    def test_unrelated_words_do_not_receive_spatial_constraints(self):
        maps = {'down_cross': torch.full((1, 4, 2), 0.5)}
        loss = self.code['compute_fuzzy_loss'](maps, [0, 1])
        self.assertAlmostEqual(loss.item(), 0.69314718, places=5)

    def test_product_t_norm_counts_every_phrase_not_only_the_weakest(self):
        # Both phrases have truth 0.5 here. Goedel min keeps one -log term, so only
        # the weakest phrase drives the objective; the product t-norm keeps both.
        maps = {'down_cross': torch.full((1, 4, 2), 0.5)}
        minimum = self.code['compute_fuzzy_loss'](maps, [0, 1], t_norm='min')
        product = self.code['compute_fuzzy_loss'](maps, [0, 1], t_norm='product')
        self.assertAlmostEqual(minimum.item(), 0.69314718, places=5)
        self.assertAlmostEqual(product.item(), 1.38629436, places=5)
        with self.assertRaises(ValueError):
            self.code['compute_fuzzy_loss'](maps, [0, 1], t_norm='lukasiewicz')

    def test_phrase_truth_scores_honour_the_sharpness_they_are_given(self):
        # Reporting truths at the default sharpness while guidance ran at another
        # value makes the two incomparable, which hid a real change in one run.
        context = 8
        store = self.code['AttentionStore'](attn_res=2)
        store.token_groups = {'phrase': [1]}
        store.text_span = (1, context - 1)
        logits = torch.full((1, 4, context), -2.0)
        logits[:, :2, 1] = 2.0
        store.begin_forward()
        store.forward(logits.softmax(-1).expand(2, -1, -1), True, 'down')
        store.end_forward()
        sharp = self.code['phrase_truth_scores'](store, 10.0, 100.0)['phrase']
        soft = self.code['phrase_truth_scores'](store, 10.0, 5.0)['phrase']
        self.assertNotAlmostEqual(sharp, soft, places=3)

    def test_size_relations_compare_membership_area(self):
        # Comparative size is gradable in image space, so the predicate reads the
        # membership mass each phrase holds rather than any calibrated extent.
        big = torch.ones(4, 4)
        small = torch.zeros(4, 4)
        small[0, 0] = 1.0
        self.assertGreater(self.code['soft_larger'](big, small).item(), 0.9)
        self.assertLess(self.code['soft_larger'](small, big).item(), 0.1)
        self.assertAlmostEqual(self.code['soft_larger'](big, big.clone()).item(), 0.5, places=6)
        # Scale free: doubling both areas leaves the truth unchanged.
        self.assertAlmostEqual(self.code['soft_larger'](big, small).item(),
                               self.code['soft_larger'](big * 2, small * 2).item(), places=6)

    def test_larger_than_relation_prefers_the_stated_ordering(self):
        context = 8
        span = (1, context - 1)

        def loss_for(relation):
            # Token 1 covers three of four positions; token 2 covers one.
            logits = torch.full((1, 4, context), -2.0)
            logits[:, [0, 1, 2], 1] = 4.0
            logits[:, [3], 2] = 4.0
            return self.code['compute_fuzzy_loss']({'down_cross': logits.softmax(-1)},
                                                   [[1], [2]], text_span=span,
                                                   relations=[([1], relation, [2])])

        self.assertLess(loss_for('larger_than').item(), loss_for('smaller_than').item())
        with self.assertRaises(ValueError):
            loss_for('much_larger_than')

    def test_summarize_guidance_flags_an_inert_run(self):
        # Three runs reached this state unnoticed: guidance executed, moved the
        # latent by well under 0.01 percent, and produced an image identical to
        # the unguided one. The summary has to say so before the images are read.
        def diagnostics(ratio, clipped=False):
            return [{'update_ratio': ratio, 'clipped': clipped, 'loss': 1.0} for _ in range(5)]

        inert = self.code['summarize_guidance'](diagnostics(7.4e-05))
        self.assertTrue(inert['inert'])
        self.assertIn('INERT', inert['note'])
        active = self.code['summarize_guidance'](diagnostics(3.0e-03))
        self.assertFalse(active['inert'])
        self.assertEqual(active['note'], 'guidance active')
        pinned = self.code['summarize_guidance'](diagnostics(3.0e-03, clipped=True))
        self.assertIn('ceiling', pinned['note'])
        self.assertEqual(self.code['summarize_guidance']([])['steps'], 0)

    def test_relative_membership_rescales_to_its_own_peak(self):
        membership = torch.tensor([0.0, 0.1, 0.4])
        relative = self.code['relative_membership'](membership)
        self.assertAlmostEqual(relative.max().item(), 1.0, places=6)
        self.assertAlmostEqual(relative[1].item(), 0.25, places=6)
        # An empty map must not divide by zero.
        self.assertTrue(torch.isfinite(self.code['relative_membership'](torch.zeros(3))).all())

    def test_relative_phrase_membership_allows_attribute_and_object_to_overlap(self):
        # Hand-built attention: both phrases peak on the same two car pixels.
        # Token-share membership cannot give both phrases high membership there.
        attention = torch.tensor([[[0.01, 0.4, 0.5, 0.09],
                                   [0.01, 0.2, 0.25, 0.54],
                                   [0.01, 0.04, 0.05, 0.9],
                                   [0.01, 0.0, 0.0, 0.99]]])
        for token in (1, 2):
            membership = self.code['phrase_membership'](
                attention, [token], (1, 4), mode='relative')
            self.assertTrue(torch.allclose(membership, torch.tensor([[1., 0.5, 0.1, 0.]])))
        default = self.code['phrase_membership'](attention, [1], (1, 4))
        explicit = self.code['phrase_membership'](attention, [1], (1, 4), mode='share')
        self.assertTrue(torch.equal(default, explicit))

    def test_relative_membership_handles_empty_maps_and_rejects_unknown_mode(self):
        attention = torch.zeros(1, 4, 3)
        membership = self.code['phrase_membership'](attention, [1], mode='relative')
        self.assertTrue(torch.equal(membership, torch.zeros(1, 4)))
        with self.assertRaises(ValueError):
            self.code['phrase_membership'](attention, [1], mode='unknown')

    def test_relative_binding_has_finite_gradients_for_both_phrases(self):
        attention = torch.tensor([[[0.1, 0.6, 0.2, 0.1], [0.1, 0.3, 0.4, 0.2],
                                   [0.1, 0.1, 0.6, 0.2], [0.1, 0.2, 0.1, 0.6]]],
                                 requires_grad=True)
        loss = self.code['compute_fuzzy_loss'](
            {'down_cross': attention}, [[1], [2]], text_span=(1, 4),
            relations=[([1], 'bound_to', [2])], t_norm='product', membership_mode='relative')
        gradient = torch.autograd.grad(loss, attention)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        for token in (1, 2):
            self.assertGreater(gradient[..., token].abs().sum().item(), 0)

    def test_relative_scores_and_refinement_masks_use_the_selected_mode(self):
        store = self.code['AttentionStore'](attn_res=2)
        store.token_groups = {'dusty': [1], 'car': [2]}
        store.text_span = (1, 4)
        store.membership_mode = 'relative'
        attention = torch.tensor([[[0.01, 0.4, 0.5, 0.09], [0.01, 0.2, 0.25, 0.54],
                                   [0.01, 0.04, 0.05, 0.9], [0.01, 0.0, 0.0, 0.99]]])
        store.begin_forward()
        store.forward(attention, True, 'down')
        store.end_forward()
        truths = self.code['phrase_truth_scores'](store)
        self.assertAlmostEqual(truths['dusty'], truths['car'], places=6)
        self.assertGreater(truths['dusty'], 0.99)
        masks = self.code['build_phrase_membership_masks'](store, torch.device('cpu'), latent_size=2)
        self.assertTrue(torch.allclose(masks['dusty'], torch.tensor([[1., 0.5], [0.1, 0.]])))
        self.assertTrue(torch.allclose(masks['dusty'], masks['car']))

    def test_binding_rewards_overlapping_attribute_and_object_support(self):
        # The object occupies the top half. The attribute lands either on the same
        # half (bound) or on the opposite half (leaked onto something else).
        context = 8
        span = (1, context - 1)
        relations = [([1], 'bound_to', [2])]

        def loss_for(attribute_rows):
            logits = torch.full((1, 4, context), -2.0)
            logits[:, attribute_rows, 1] = 4.0
            logits[:, [0, 1], 2] = 4.0
            return self.code['compute_fuzzy_loss']({'down_cross': logits.softmax(-1)},
                                                   [[1], [2]], relations=relations,
                                                   text_span=span)

        bound = loss_for([0, 1])
        leaked = loss_for([2, 3])
        self.assertLess(bound.item(), leaked.item(),
                        'Binding did not penalise an attribute grounded away from its object')

    def test_binding_gradient_reaches_both_operands(self):
        # Binding has to optimise the attribute even when another phrase is the
        # weakest conjunct under the Goedel t-norm. Sharpness matters here: a
        # saturated membership softmax has exactly zero gradient, so a binding at
        # the default sharpness of 100 cannot be optimised at all.
        context = 8
        logits = torch.full((1, 4, context), -2.0)
        logits[:, [2, 3], 1] = 4.0
        logits[:, [0, 1], 2] = 4.0

        def binding_gradient(sharpness):
            attention = logits.softmax(-1).requires_grad_(True)
            loss = self.code['compute_fuzzy_loss']({'down_cross': attention}, [[1], [2]],
                                                   relations=[([1], 'bound_to', [2])],
                                                   text_span=(1, context - 1),
                                                   membership_sharpness=sharpness)
            return torch.autograd.grad(loss, attention)[0]

        gradient = binding_gradient(5.0)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[..., 1].abs().sum().item(), 0)
        self.assertGreater(gradient[..., 2].abs().sum().item(), 0)
        self.assertEqual(binding_gradient(100.0)[..., 1].abs().sum().item(), 0.0,
                         'Saturated membership is expected to carry no gradient')

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

    def test_small_guidance_gradients_are_not_amplified(self):
        # A tiny gradient must remain a tiny update, rather than be normalized to lr.
        from types import SimpleNamespace
        store = self.code['AttentionStore'](attn_res=2)
        class ConditionalForward(torch.nn.Module):
            def forward(self, latents, timestep, **kwargs):
                weights = torch.tensor([0.1, 0.2, -0.1]).view(1, 1, 3)
                probabilities = (latents.flatten().view(1, 4, 1) * weights * 0.001).softmax(-1)
                store.forward(probabilities, True, 'down')
                return (torch.zeros_like(latents),)
        pipeline = object.__new__(self.code['FuzzyAttendExciteSDXLPipeline'])
        pipeline.unet = ConditionalForward()
        pipeline.scheduler = SimpleNamespace(scale_model_input=lambda value, timestep: value)
        config = SimpleNamespace(alpha=10., spatial_loss_weight=0., attend_excite_lr=0.2,
                                 n_inference_steps=50, token_groups=[[1]], text_span=None,
                                 membership_sharpness=100.0)
        latents = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
        updated = pipeline._update_latents_with_fuzzy_loss(latents, store, [1], config,
            timestep=1, prompt_embeds=torch.zeros(1, 3, 4), added_cond_kwargs={})
        diagnostics = store.guidance_diagnostics[0]
        self.assertLess(diagnostics['gradient_rms'], 0.01)
        self.assertLessEqual(diagnostics['update_rms'], config.attend_excite_lr * diagnostics['gradient_rms'] + 1e-7,
                             'Tiny gradients were amplified into large latent perturbations')
        self.assertFalse(torch.equal(latents, updated))

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

            def sample(updates, membership_mode='share'):
                config = code['RunConfig'](prompt='a cat', height=64, width=64, n_inference_steps=2,
                    max_iter_to_alter=updates, output_path=path, membership_mode=membership_mode)
                store = code['AttentionStore'](attn_res=4)
                pipeline.register_attention_control(store)
                output = pipeline(prompt='a cat', height=64, width=64, num_inference_steps=2,
                    guidance_scale=2, generator=torch.Generator().manual_seed(9), cfg=config,
                    attention_store=store, token_indices=indices, output_type='latent')
                return output.images, store

            baseline, _ = sample(0)
            unet.set_attn_processor(dict(pipeline._original_attention_processors))
            native = StableDiffusionXLPipeline(**pipeline.components, add_watermarker=False)
            native_output = native(prompt='a cat', negative_prompt='', height=64, width=64,
                num_inference_steps=2, guidance_scale=2,
                generator=torch.Generator().manual_seed(9), output_type='latent').images
            # Manual attention versus SDPA differs at fp32 rounding scale; the scheduler's
            # initial sigma (~157) magnifies this into ~2e-4 latent error in this fixture.
            self.assertTrue(torch.allclose(baseline, native_output, atol=5e-4, rtol=1e-5),
                            f'Guidance-off/native max latent difference: {(baseline - native_output).abs().max().item()}')
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
            relative, relative_store = sample(1, 'relative')
            self.assertTrue(torch.isfinite(relative).all())
            self.assertGreater(relative_store.guidance_diagnostics[0]['update_rms'], 0)
            self.assertNotAlmostEqual(relative_store.guidance_diagnostics[0]['loss'],
                                      store.guidance_diagnostics[0]['loss'], places=5,
                                      msg='SDXL ignored the selected membership mode')
            # Exercise inherited VAE decode/postprocessing as well as the latent path.
            config = code['RunConfig'](height=64, width=64, n_inference_steps=1, max_iter_to_alter=0, output_path=path)
            result = pipeline(prompt='a cat', height=64, width=64, num_inference_steps=1, guidance_scale=1,
                              generator=torch.Generator().manual_seed(9), cfg=config)
            self.assertEqual(result.images[0].size, (64, 64))

if __name__ == '__main__':
    unittest.main()
