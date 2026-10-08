"""Exercise CLI-to-notebook settings without loading pretrained models."""
import builtins
import contextlib
import io
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import run_fullpipeline


class GenerationReached(Exception):
    pass


class RunnerOptionsTests(unittest.TestCase):
    def test_full_run_passes_requested_prompt_and_fuzzy_settings_to_generation(self):
        captured = {}

        def generate(prompt, words, **settings):
            captured.update(prompt=prompt, words=words, **settings)
            raise GenerationReached

        display_module = types.ModuleType('IPython.display')
        display_module.display = lambda *args: None

        def execute_without_models(code, namespace):
            # Execute the real configuration and entry cells, stopping at the GPU
            # boundary. The import/model-loading cells would download weights.
            if code.co_filename.endswith(':cell-4'):
                namespace.update(dataclass=dataclass, field=field, Path=Path,
                                 List=List, Optional=Optional)
                builtins.exec(code, namespace)
            elif code.co_filename.endswith(':cell-30'):
                namespace.update(generate=generate, model=object())
                builtins.exec(code, namespace)

        args = ['run_fullpipeline.py', '--full', '--prompt',
                'A slightly dusty red sports car parked on a road.',
                '--words', 'slightly dusty,red sports car,road', '--seed', '42',
                '--sharpness', '20', '--tnorm', 'product', '--lr', '5',
                '--membership-mode', 'relative',
                '--bind', 'slightly dusty>red sports car', '--binding-weight', '2']
        with patch.object(sys, 'argv', args), \
                patch.object(run_fullpipeline, 'exec', execute_without_models, create=True), \
                patch.dict(sys.modules, {'IPython.display': display_module}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(GenerationReached):
                run_fullpipeline.main()
        self.assertEqual(captured['prompt'], args[3])
        self.assertEqual(captured['words'], ['slightly dusty', 'red sports car', 'road'])
        self.assertEqual(captured['seed'], 42)
        self.assertEqual(captured['num_steps'], 50)
        self.assertEqual(captured['attend_excite_lr'], 5)
        self.assertEqual(captured['membership_sharpness'], 20)
        self.assertEqual(captured['membership_mode'], 'relative')
        self.assertEqual(captured['t_norm'], 'product')
        self.assertEqual(captured['relations'], [('slightly dusty', 'bound_to', 'red sports car')])
        self.assertEqual(captured['binding_loss_weight'], 2)

    def test_comparison_receives_bindings_instead_of_silently_dropping_them(self):
        captured = {}
        comparison = types.ModuleType('compare_fullpipeline')
        comparison.run_comparison = lambda namespace, **kwargs: captured.update(kwargs)
        args = ['run_fullpipeline.py', '--compare', '--prompt', 'dusty car',
                '--words', 'dusty,car', '--bind', 'dusty>car', '--binding-weight', '2']
        with patch.object(sys, 'argv', args), \
                patch.object(run_fullpipeline, 'exec', lambda *args: None, create=True), \
                patch.dict(sys.modules, {'compare_fullpipeline': comparison}), \
                patch.object(run_fullpipeline.importlib, 'reload', lambda module: module), \
                contextlib.redirect_stdout(io.StringIO()):
            run_fullpipeline.main()
        self.assertEqual(captured.get('bindings'), [('dusty', 'car')])
        self.assertEqual(captured.get('binding_weight'), 2)


if __name__ == '__main__':
    unittest.main()
