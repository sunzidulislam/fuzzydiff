import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageCommandTests(unittest.TestCase):
    def test_full_cli_is_lightweight_from_external_directory(self):
        result = subprocess.run([sys.executable, str(ROOT / 'run_fullpipeline.py'), '--check'],
                                cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Compiled 26 code cells', result.stdout)
        code = '''import sys
sys.modules['torch'] = None
from fuzzydiff.cli.full_pipeline import main
main(['--check'])
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_benchmark_workers_resolve_repo_and_record_implementation_sources(self):
        # The real worker runs on CPU and must record a GPU refusal without fabricated times.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'pilot'
            result = subprocess.run([sys.executable, str(ROOT / 'run_benchmark.py'),
                '--prompt', 'a ball', '--words', 'ball', '--methods', 'stable',
                '--seeds', '42', '--repeats', '1', '--output', str(output)],
                cwd=directory, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output / 'results.json').read_text())
            self.assertIn('requires a GPU', report['runs'][0]['error'])
            self.assertIsNone(report['summary']['stable']['inference_mean_s'])
            hashes = json.loads((output / 'source_hashes.json').read_text())
            self.assertIn('fuzzydiff/models/sdxl.py', hashes)
            self.assertIn('fuzzydiff/notebook_runtime.py', hashes)

    def test_legacy_imports_share_patch_state_and_reload_implementation(self):
        import importlib
        import run_benchmark
        from fuzzydiff.experiments import benchmark
        import run_fullpipeline
        from fuzzydiff.cli import full_pipeline
        import benchmark_timing
        from fuzzydiff.reporting import timing
        self.assertIs(run_benchmark, benchmark)
        self.assertIs(run_fullpipeline, full_pipeline)
        self.assertIs(benchmark_timing, timing)
        importlib.reload(run_benchmark)
        self.assertIs(run_benchmark.run_worker, benchmark.run_worker)

    def test_dry_run_package_entry_without_torch(self):
        code = '''import sys
sys.modules['torch'] = None
from fuzzydiff.cli.benchmark import main
assert main(['--prompt','a ball','--words','ball','--methods','enhancing','--steps','4','--guidance','0','--dry-run']) == 0
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
