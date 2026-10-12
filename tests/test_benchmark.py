"""Benchmark accounting and notebook loading; never download model weights."""
import importlib.util
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path


class BenchmarkTests(unittest.TestCase):
    def test_enhancing_is_turbo_and_uses_native_sampling_settings(self):
        models = self.module('benchmark_models')
        runner = self.module('run_benchmark')
        self.assertEqual(models.NOTEBOOKS['enhancing'].name, 'enhancing-f3 (1).ipynb')
        options = runner.parse_options(['--prompt', 'red ball beside blue ball',
            '--words', 'red ball,blue ball', '--methods', 'enhancing',
            '--steps', '4', '--guidance', '0', '--dry-run'])
        self.assertEqual(options['methods'], ['enhancing'])
        self.assertEqual(models.effective_sampling_settings('enhancing', options),
                         {'steps': 4, 'guidance': 0.0, 'size': 512})
        standard = {**options, 'steps': 50, 'guidance': 7.5}
        self.assertEqual(models.effective_sampling_settings('enhancing', standard)['steps'], 4)
        self.assertEqual(models.effective_sampling_settings('enhancing', standard)['guidance'], 0)
        self.assertEqual(models.effective_sampling_settings('stable', standard)['steps'], 50)

    def test_enhancing_generation_forwards_native_settings_without_refinement(self):
        import torch
        models = self.module('benchmark_models')
        model = models.NotebookModel.__new__(models.NotebookModel)
        model.method = 'enhancing'
        model.options = {'prompt': 'red ball', 'words': ['red ball'], 'size': 512,
                         'steps': 50, 'guidance': 7.5, 'negative': ''}
        model.ns, model.torch, model.device = {}, torch, torch.device('cpu')
        captured = {}
        sentinel = object()
        def base(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[sentinel])
        model.base = base
        self.assertIs(model.generate(142), sentinel)
        self.assertEqual(captured['num_inference_steps'], 4)
        self.assertEqual(captured['guidance_scale'], 0)
        self.assertEqual(captured['height'], 512)
        self.assertEqual(captured['generator'].initial_seed(), 142)

    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), f'{name} is not implemented')
        return __import__(name)

    def test_nested_timing_does_not_count_tokenizer_work_twice(self):
        timing = self.module('benchmark_timing')
        ticks = iter([0, 1, 2, 4, 6, 10])
        ledger = timing.TimeLedger(clock=lambda: next(ticks))
        with ledger.measure('generation'):
            with ledger.measure('tokenize'):
                with ledger.measure('tokenize'):
                    pass
        self.assertEqual(ledger.seconds, {'tokenize': 5, 'generation': 5})
        self.assertEqual(sum(ledger.seconds.values()), 10)

    def test_failed_timing_is_unwound_and_marked(self):
        timing = self.module('benchmark_timing')
        ticks = iter([0, 3, 4, 6])
        ledger = timing.TimeLedger(clock=lambda: next(ticks))
        with self.assertRaisesRegex(ValueError, 'bad'):
            with ledger.measure('tokenize'):
                raise ValueError('bad')
        with ledger.measure('generation'):
            pass
        self.assertEqual(ledger.seconds, {'tokenize': 3, 'generation': 2})
        self.assertEqual(ledger.failures, {'tokenize': 1})

    def test_definition_loader_does_not_run_notebook_batches_or_model_loads(self):
        models = self.module('benchmark_models')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'probe.ipynb'
            source = ('from dataclasses import dataclass\n'
                      'HEDGE_TARGETS = {"slightly": 0.15}\n'
                      '@dataclass\nclass Config:\n    value: int = 4\n'
                      'def operation(x):\n    return x + HEDGE_TARGETS["slightly"]\n'
                      'model = missing_download_function()\n'
                      'raise RuntimeError("Notebook batch must not run")\n')
            path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': source}]}))
            original = path.read_bytes()
            ns = models.load_definitions(path)
            self.assertEqual(ns['Config']().value, 4)
            self.assertEqual(ns['operation'](1), 1.15)
            self.assertNotIn('model', ns)
            self.assertEqual(path.read_bytes(), original)

    def test_tokenizer_instrumentation_preserves_results_and_restores_methods(self):
        timing = self.module('benchmark_timing')
        class Tokenizer:
            def __call__(self, text):
                return self.encode(text)
            def encode(self, text):
                return [ord(c) for c in text]
        target, other = Tokenizer(), Tokenizer()
        original = Tokenizer.__call__
        ticks = iter(range(20))
        ledger = timing.TimeLedger(clock=lambda: next(ticks))
        with timing.instrument_tokenizers([target], ledger):
            self.assertEqual(target('ab'), [97, 98])
            before = dict(ledger.seconds)
            self.assertEqual(other('c'), [99])
            self.assertEqual(ledger.seconds, before)
        self.assertIs(Tokenizer.__call__, original)
        self.assertEqual(ledger.seconds['tokenize'], 3)

    def test_summary_uses_native_whole_model_time_and_excludes_failed_runs(self):
        runner = self.module('run_benchmark')
        records = [
            {'method': 'fuzzydiff', 'status': 'ok', 'inference_s': 12.0,
             'peak_allocated_gib': 4.0, 'peak_reserved_gib': 5.0},
            {'method': 'fuzzydiff', 'status': 'ok', 'inference_s': 16.0,
             'peak_allocated_gib': 6.0, 'peak_reserved_gib': 7.0},
            {'method': 'fuzzydiff', 'status': 'error', 'inference_s': 999.0},
        ]
        summary = runner.summarize(records)['fuzzydiff']
        self.assertEqual(summary['successful_runs'], 2)
        self.assertEqual(summary['failed_runs'], 1)
        self.assertEqual(summary['inference_mean_s'], 14)
        self.assertEqual(summary['inference_median_s'], 14)
        self.assertEqual(summary['peak_allocated_gib'], 6)
        self.assertEqual(summary['peak_reserved_gib'], 7)

    def test_invalid_prompt_settings_fail_before_starting_workers(self):
        runner = self.module('run_benchmark')
        for arguments in [
            ['--prompt', 'a ball', '--words', 'missing'],
            ['--prompt', 'a ball', '--words', 'ball', '--repeats', '0'],
            ['--prompt', 'a ball', '--words', 'ball', '--seeds', '42,42'],
            ['--prompt', 'a ball', '--words', 'ball', '--size', '513'],
            ['--prompt', 'a ball', '--words', 'ball', '--size', '768'],
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                runner.parse_options(arguments)

    def test_nonfinite_diagnostic_keeps_prior_results_and_records_numerical_failure(self):
        runner = self.module('run_benchmark')
        self.assertTrue(hasattr(runner, 'append_record'))
        good = {'method': 'predicate', 'seed': 42, 'status': 'ok', 'inference_s': 5.0}
        rows = [good]
        bad = {'method': 'predicate', 'seed': 142, 'repeat': 0, 'status': 'ok',
               'inference_s': 7.0, 'details': {'loss_values': [float('nan')]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runs.json'
            runner.append_record(rows, bad, path)
            saved = json.loads(path.read_text())
        self.assertEqual(saved[0], good)
        self.assertEqual(saved[1]['seed'], 142)
        self.assertEqual(saved[1]['status'], 'error')
        self.assertEqual(saved[1]['phase'], 'recording_diagnostics')
        self.assertIn('non-finite', saved[1]['error'])

    def test_model_metadata_reads_transformers_config_commit_hash(self):
        models = self.module('benchmark_models')
        model = object.__new__(models.NotebookModel)
        model.method = 'stable'
        component = SimpleNamespace(config=SimpleNamespace(_commit_hash='revision-a'))
        model.base = SimpleNamespace(unet=SimpleNamespace(dtype='float16'))
        model.pipelines = [SimpleNamespace(
            scheduler=SimpleNamespace(__class__=SimpleNamespace(__name__='TestScheduler')),
            **{name: component for name in ('unet', 'vae', 'text_encoder', 'text_encoder_2')})]
        model.checkpoints = ['test/model']
        model.load_s = 3.5
        model.load_parts = {'base': 3.5}
        model.definition_setup_s = 0.1
        model.options = {'steps': 50, 'guidance': 7.5, 'size': 512}
        metadata = model.metadata()
        self.assertEqual(metadata['component_commit_hashes']['0.text_encoder'], 'revision-a')

    def test_all_actual_notebooks_load_definitions_without_pretrained_models(self):
        models = self.module('benchmark_models')
        expected = {'stable': 'generate', 'attend': 'AttendAndExcitePipeline',
                    'predicate': 'PredicatedDiffPipeline', 'aesthetic': 'generate_image_return',
                    'enhancing': 'StableDiffusionXLPipeline',
                    'fuzzydiff': 'FuzzyAttendExciteSDXLPipeline'}
        for method, symbol in expected.items():
            with self.subTest(method=method):
                ns = models.load_definitions(models.NOTEBOOKS[method])
                self.assertIn(symbol, ns)
                self.assertNotIn('model', ns)
                self.assertNotIn('stable', ns)
                self.assertNotIn('clip_model', ns)

    def test_preprocessing_profiler_restores_notebook_functions_after_failure(self):
        models = self.module('benchmark_models')
        timing = self.module('benchmark_timing')
        self.assertTrue(hasattr(models.NotebookModel, 'profile_preprocessing'))
        def groups(*args):
            raise ValueError('token alignment failed')
        model = object.__new__(models.NotebookModel)
        model.ns = {'get_token_groups': groups}
        ledger = timing.TimeLedger()
        with self.assertRaises(ValueError):
            with model.profile_preprocessing(ledger):
                model.ns['get_token_groups']()
        self.assertIs(model.ns['get_token_groups'], groups)
        self.assertEqual(ledger.failures['preprocess'], 1)

    def test_profiled_plain_adapter_preserves_seeded_pixels_with_real_cpu_modules(self):
        import torch
        from test_sd_fuzzy import tiny_pipeline
        from benchmark_models import NotebookModel
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            pipeline, _ = tiny_pipeline(Path(folder), None)
            model = object.__new__(NotebookModel)
            model.torch, model.device = torch, torch.device('cpu')
            model.method, model.ns = 'stable', {}
            model.options = dict(prompt='a red book', words=['red book'], size=64,
                                 steps=2, guidance=7.5, negative='')
            model.base, model.pipelines = pipeline, [pipeline]
            native = model.generate(42)
            original = pipeline.text_encoder.forward
            profiled = model.profile(42)
            self.assertEqual(native.tobytes(), profiled.image.tobytes())
            self.assertEqual(pipeline.text_encoder.forward, original)
            for category in ('preprocess', 'tokenize', 'text_encode', 'vae', 'generation'):
                self.assertGreater(profiled.measurements['exclusive_seconds'][category], 0)
            self.assertEqual(profiled.measurements['unet_forward_calls'], 2)

    def test_cpu_worker_failure_produces_report_without_fabricated_measurements(self):
        runner = self.module('run_benchmark')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results'
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONIOENCODING='utf-8')
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / 'run_benchmark.py'), '--prompt', 'a ball',
                '--words', 'ball', '--methods', 'stable', '--seeds', '42',
                '--repeats', '1', '--output', str(output)],
                env=environment, capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output / 'results.json').read_text())
            self.assertEqual(report['summary']['stable']['successful_runs'], 0)
            self.assertEqual(report['summary']['stable']['failed_runs'], 1)
            self.assertIsNone(report['summary']['stable']['inference_mean_s'])
            self.assertIn('requires a GPU', report['runs'][0]['error'])
            self.assertTrue((output / 'comparison.md').is_file())


if __name__ == '__main__':
    unittest.main()
