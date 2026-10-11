import importlib
import unittest
from unittest.mock import patch


class PackageSDTests(unittest.TestCase):
    def test_legacy_patch_reaches_generation_and_reload(self):
        import run_sd
        from fuzzydiff.models import sd15
        self.assertIs(run_sd, sd15)
        sentinel = object()
        with patch.object(run_sd, 'resolve_relations', return_value=sentinel):
            self.assertIs(sd15.resolve_bindings({}, [('a', 'b')]), sentinel)
        importlib.reload(run_sd)
        self.assertIs(run_sd.generate_plain, sd15.generate_plain)

    def test_cli_parses_help_without_importing_model_runtime(self):
        import subprocess
        import sys
        code = ('import sys; sys.modules["torch"] = None; '
                'from fuzzydiff.cli.sd15 import main; main(["--help"])')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--fuzzy', result.stdout)
