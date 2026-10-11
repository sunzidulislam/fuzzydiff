import gc
import json
import tempfile
import unittest
import weakref
from contextlib import nullcontext, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ReviewRegressionTests(unittest.TestCase):
    def test_closed_notebook_session_releases_registered_pipeline(self):
        from fuzzydiff.models.session import ModelSession
        from fuzzydiff.notebook_runtime import load_definitions
        class Pipeline:
            def maybe_free_model_hooks(self):
                pass
        with patch('fuzzydiff.notebook_runtime.definition_code', return_value=compile('', '<fixture>', 'exec')):
            ns = load_definitions(Path('owned.ipynb'))
        pipeline = Pipeline()
        reference = weakref.ref(pipeline)
        session = object.__new__(ModelSession)
        session.ns, session.base, session.pipelines = ns, pipeline, [pipeline]
        session.original_processors = []
        ns['model'] = pipeline
        del pipeline
        session.close()
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(ns, {})

    def test_legacy_loader_and_sampling_patches_control_notebook_model(self):
        import torch
        import benchmark_models as legacy
        class LoaderReached(Exception):
            pass
        seen = []
        def loader(path):
            seen.append(path)
            raise LoaderReached
        with patch.object(legacy, 'load_definitions', loader):
            with self.assertRaises(LoaderReached):
                legacy.NotebookModel('stable', {})
        self.assertEqual(len(seen), 1)
        model = object.__new__(legacy.NotebookModel)
        model.method, model.ns = 'stable', {}
        model.torch, model.device = torch, torch.device('cpu')
        model.options = dict(prompt='ball', words=[], size=512, steps=50, guidance=7.5, negative='')
        captured = {}
        def base(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=['image'])
        model.base = base
        with patch.object(legacy, 'effective_sampling_settings', return_value={'steps': 2, 'guidance': 3, 'size': 512}):
            self.assertEqual(model.generate(42), 'image')
        self.assertEqual(captured['num_inference_steps'], 2)
        self.assertEqual(captured['guidance_scale'], 3)

    def test_legacy_compiler_patch_reaches_actual_definition_loader(self):
        import benchmark_models as legacy
        with patch.object(legacy, 'definition_code', return_value=compile('answer = 7', '<fixture>', 'exec')):
            ns = legacy.load_definitions(Path('not-a-file.ipynb'))
        self.assertEqual(ns['answer'], 7)

    def test_benchmark_json_invalid_types_and_enums_fail_before_workers(self):
        from fuzzydiff.cli.benchmark import parse_options
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            base = {'prompt': 'ball', 'words': ['ball'], 'methods': ['enhancing'], 'guidance': 0}
            for invalid in ({'tnorm': 'typo'}, {'membership_mode': 'typo'},
                            {'warmup': 1.5}, {'steps': True}, {'repeats': 1.5},
                            {'size': 512.0}, {'prepare_cache': 'yes'}):
                path.write_text(json.dumps({**base, **invalid}))
                with self.subTest(invalid=invalid), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_options(['--config', str(path), '--dry-run'])

    def test_generation_method_options_nonobjects_get_clear_configuration_error(self):
        from fuzzydiff.config import merge_request_config
        for invalid in ([], None, 'wrong'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                merge_request_config({'model': 'stable', 'prompt': 'ball', 'method_options': invalid}, {})
