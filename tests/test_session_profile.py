from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class SessionProfileTests(unittest.TestCase):
    def session(self, fail=False):
        import torch
        from fuzzydiff.models.session import ModelSession

        class Tokenizer:
            def __call__(self, prompt):
                return prompt

        class Encoder(torch.nn.Module):
            def forward(self, value):
                return value

        class Session(ModelSession):
            def generate(self, seed, ledger=None):
                self.base.tokenizer('prompt')
                self.base.text_encoder('prompt')
                self.base.unet(torch.ones(1, 1))
                if fail:
                    raise RuntimeError('inference failed')
                return seed

        session = object.__new__(Session)
        session.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        session.ns = {}
        session.base = SimpleNamespace(tokenizer=Tokenizer(), text_encoder=Encoder(),
                                       unet=torch.nn.Linear(1, 1))
        session.pipelines = [session.base]
        return session

    def test_profile_returns_image_and_exclusive_measurements_and_restores_hooks(self):
        session = self.session()
        tokenizer_call = type(session.base.tokenizer).__call__
        result = session.profile(142)
        self.assertEqual(result.image, 142)
        self.assertEqual(result.measurements['unet_forward_calls'], 1)
        self.assertEqual(result.measurements['calls']['tokenize'], 1)
        self.assertGreaterEqual(result.measurements['text_encode_s'], 0)
        seconds = result.measurements['exclusive_seconds']
        self.assertEqual(result.measurements['profile_total_s'], sum(seconds.values()))
        self.assertEqual(result.measurements['generation_s'], seconds['generation'] + seconds.get('vae', 0))
        self.assertIs(type(session.base.tokenizer).__call__, tokenizer_call)
        self.assertNotIn('forward', vars(session.base.text_encoder))
        self.assertFalse(session.base.unet._forward_pre_hooks)

    def test_failed_profile_restores_all_instrumentation(self):
        session = self.session(fail=True)
        tokenizer_call = type(session.base.tokenizer).__call__
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            session.profile(42)
        self.assertIs(type(session.base.tokenizer).__call__, tokenizer_call)
        self.assertNotIn('forward', vars(session.base.text_encoder))
        self.assertFalse(session.base.unet._forward_pre_hooks)

    def test_cpu_session_profile_does_not_require_cuda(self):
        import torch
        session = self.session()
        session.torch, session.device = torch, torch.device('cpu')
        self.assertEqual(session.profile(42).image, 42)

    def test_generate_passes_one_context_to_adapter(self):
        import torch
        from fuzzydiff.models.session import ModelSession

        class Session(ModelSession):
            def _generate(self, context):
                return context

        session = object.__new__(Session)
        session.method = 'enhancing'
        session.options = dict(prompt='a ball', words=['ball'], steps=50, guidance=7.5, size=512)
        session.ns, session.torch, session.device = {}, torch, torch.device('cpu')
        context = session.generate(42)
        self.assertEqual(context.seed, 42)
        self.assertEqual(context.generator.initial_seed(), 42)
        self.assertEqual(context.common['num_inference_steps'], 4)
        self.assertEqual(context.common['guidance_scale'], 0)
        self.assertEqual(context.words, ['ball'])

    def test_benchmark_serializes_session_profile_without_pipeline_access(self):
        import torch
        from PIL import Image
        from fuzzydiff.experiments.benchmark import run_worker
        from fuzzydiff.models.session import ProfileResult

        measurements = dict(preprocess_s=1, tokenize_s=2, text_encode_s=3,
                            generation_s=4, profile_total_s=10,
                            exclusive_seconds={'generation': 4}, calls={}, unet_forward_calls=2)

        class Session:
            # No pipelines attribute: the benchmark must cross the session seam.
            last_details = {}
            def __init__(self, *args):
                pass
            def metadata(self):
                return {}
            def reset(self):
                pass
            def generate(self, seed):
                return Image.new('RGB', (2, 2), 'red')
            def profile(self, seed):
                return ProfileResult(self.generate(seed), measurements)
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            options = dict(output=folder, warmup=0, seeds=[42], repeats=1)
            with patch('fuzzydiff.models.registry.NotebookModel', Session), patch.multiple(
                    torch.cuda, is_available=lambda: True, set_device=lambda n: None,
                    init=lambda: None, get_device_name=lambda n: 'test',
                    get_device_properties=lambda n: SimpleNamespace(total_memory=0),
                    synchronize=lambda: None, empty_cache=lambda: None,
                    reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                    max_memory_reserved=lambda: 0):
                self.assertEqual(run_worker('stable', options), 0)
            rows = json.loads((Path(folder) / 'stable/runs.json').read_text())
            self.assertEqual(rows[0]['profile'], {**measurements, 'same_pixels_as_native': True})
            self.assertEqual(rows[0]['profile_status'], 'ok')
