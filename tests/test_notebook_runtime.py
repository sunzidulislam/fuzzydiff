import json
import os
import tempfile
import unittest
from pathlib import Path


class NotebookRuntimeTests(unittest.TestCase):
    def test_runtime_keeps_cell_indices_and_loading_policies(self):
        from fuzzydiff.notebook_runtime import notebook_cells, compile_notebook, load_definitions, load_prefix_definitions
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.ipynb'
            path.write_text(json.dumps({'cells': [
                {'cell_type': 'markdown', 'source': 'note'},
                {'cell_type': 'code', 'source': ['%pip install forbidden\n', 'def selected():\n    return 7\n']},
                {'cell_type': 'code', 'source': 'FUZZY_ATTRIBUTE_TARGETS = {"mostly": .85}\ndef target():\n    return FUZZY_ATTRIBUTE_TARGETS["mostly"]\n'},
                {'cell_type': 'code', 'source': 'raise RuntimeError("batch must not execute")'},
            ]}), encoding='utf-8')
            original = path.read_bytes()
            self.assertEqual([i for i, _ in notebook_cells(path)], [1, 2, 3])
            self.assertTrue(compile_notebook(path)[0][1].co_filename.endswith(':cell-1'))
            ns = load_definitions(path)
            self.assertEqual(ns['target'](), .85)
            scope = {'sentinel': 1}
            self.assertIs(load_prefix_definitions(path, ('def selected',), scope), scope)
            self.assertEqual(scope['selected'](), 7)
            self.assertNotIn('target', scope)
            self.assertEqual(original, path.read_bytes())

    def test_paths_are_independent_of_cwd_and_hashes_require_files(self):
        from fuzzydiff.paths import repository_root, canonical_notebook
        from fuzzydiff.notebook_runtime import source_hashes
        root = Path(__file__).resolve().parents[1]
        old = Path.cwd()
        try:
            os.chdir(root.parent)
            self.assertEqual(repository_root(), root)
            self.assertEqual(canonical_notebook(), root / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb')
            hashes = source_hashes([canonical_notebook()])
            self.assertEqual(len(hashes['pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb']), 64)
            with self.assertRaises(FileNotFoundError):
                source_hashes([root / 'missing-required-source.py'])
        finally:
            os.chdir(old)
