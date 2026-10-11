import subprocess
import sys
import unittest
from types import SimpleNamespace


class ModelRegistryTests(unittest.TestCase):
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
        with patch.object(registry, 'adapter_class', return_value=Broken):
            with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                registry.create_session('enhancing', {})
        self.assertEqual(events, ['allocated', 'closed'])
