import subprocess
import sys
import unittest
from types import SimpleNamespace


class ModelRegistryTests(unittest.TestCase):
    def test_fuzzy_session_forwards_bound_hedges_and_relations_to_notebook(self):
        import torch
        from fuzzydiff.models.sdxl import SDXLSession
        from contextlib import nullcontext
        seen = {}
        image = object()
        store = SimpleNamespace(guidance_diagnostics=[], hedge_targets={'mostly blue': {'target': .85}},
                                token_groups={'vase': [3]})
        def generate(*args, **kwargs):
            seen.update(kwargs)
            return image, None, store
        def refine(*args, **kwargs):
            store.refinement_weights = {'mostly blue': -.2}
            return image
        session = object.__new__(SDXLSession)
        session.method = 'fuzzydiff'
        session.base = SimpleNamespace(maybe_free_model_hooks=lambda: None)
        session.refiner = object()
        session.ns = {'generate': generate, 'refine_with_predicate': refine}
        session.options = dict(steps=50, guidance=7.5, size=512, updates=30, lr=5,
            sharpness=20, tnorm='product', membership_mode='share', negative='',
            bindings=[('mostly blue', 'vase')], relations=[('vase', 'above', 'table')],
            binding_weight=2, hedge_guidance=True, refine_strength=.4, refine_steps=30,
            predicate_strength=.5)
        session.last_details = {}
        session.options.update(prompt='mostly blue vase on table', words=['mostly blue', 'vase', 'table'])
        session.torch, session.device = torch, torch.device('cpu')
        session.generate(42)
        self.assertEqual(seen['relations'], [('mostly blue', 'bound_to', 'vase'), ('vase', 'above', 'table')])
        self.assertTrue(seen['hedge_guidance'])
        self.assertEqual(seen['binding_loss_weight'], 2)
        self.assertEqual(session.last_details['hedge_targets']['mostly blue']['target'], .85)
        self.assertEqual(session.last_details['refinement_weights'], {'mostly blue': -.2})

    def test_close_attempts_every_pipeline_when_a_hook_fails(self):
        from fuzzydiff.models.session import ModelSession
        calls = []
        def broken():
            calls.append('first')
            raise RuntimeError('hook failed')
        session = object.__new__(ModelSession)
        session.pipelines = [SimpleNamespace(maybe_free_model_hooks=broken),
                             SimpleNamespace(maybe_free_model_hooks=lambda: calls.append('second'))]
        session.original_processors = []
        with self.assertRaises(RuntimeError):
            session.close()
        self.assertEqual(calls, ['first', 'second'])
        self.assertEqual(session.pipelines, [])

    def test_aliases_remain_truthful_without_torch_imports(self):
        code = '''import sys
sys.modules['torch'] = None
from fuzzydiff.models.registry import resolve_model
for alias, expected in [('stable', 'sd14'), ('attend', 'attend_archived'), ('predicate', 'predicate_archived'), ('aesthetic', 'sdxl_base_refiner'), ('enhancing', 'sdxl_turbo'), ('fuzzydiff', 'fuzzydiff_full'), ('sd15', 'sd15')]:
    assert resolve_model(alias).id == expected
try:
    resolve_model('spo')
except ValueError:
    pass
else:
    raise AssertionError('stock SDXL must not be called SPO')
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_session_restores_processors_and_releases_references(self):
        from fuzzydiff.models.session import ModelSession
        captured = []
        unet = SimpleNamespace(set_attn_processor=lambda value: captured.append(value))
        pipeline = SimpleNamespace(unet=unet, maybe_free_model_hooks=lambda: captured.append('freed'))
        session = object.__new__(ModelSession)
        session.pipelines = [pipeline]
        session.original_processors = [{'a': 'original'}]
        session.base = pipeline
        session.ns = {'model': pipeline}
        session.close()
        session.close()
        self.assertEqual(captured, [{'a': 'original'}, 'freed'])
        self.assertEqual(session.pipelines, [])
        self.assertIsNone(session.base)
        self.assertEqual(session.ns, {})

    def test_factory_closes_partially_loaded_session_on_failure(self):
        from fuzzydiff.models import registry
        from unittest.mock import patch
        events = []
        class Broken:
            def __init__(self, method, options):
                events.append('allocated')
                raise ValueError('bad checkpoint')
            def close(self):
                events.append('closed')
                raise RuntimeError('secondary cleanup error')
        with patch.object(registry, 'adapter_class', return_value=Broken):
            with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                registry.create_session('enhancing', {})
        self.assertEqual(events, ['allocated', 'closed'])
