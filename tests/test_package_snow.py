import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path


class PackageSnowTests(unittest.TestCase):
    def test_package_plans_and_actual_implementation_fingerprints(self):
        from fuzzydiff.experiments.snow.study import make_design
        from fuzzydiff.cli.snow import main
        import snow_experiment
        from fuzzydiff.experiments.snow import study
        self.assertIs(snow_experiment, study)
        hashes = make_design()['code_sha256']
        self.assertIn('fuzzydiff/experiments/snow/study.py', hashes)
        self.assertIn('fuzzydiff/experiments/snow/gpu.py', hashes)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-written'
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(['--plan', '--output', str(output)]), 0)
            self.assertEqual(json.loads(stream.getvalue())['images'], 15)
            self.assertFalse(output.exists())

    def test_prior_code_manifest_cannot_resume_after_restructuring(self):
        from fuzzydiff.experiments.snow.study import make_design, run_study
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = make_design()
            old = {**design, 'code_sha256': {'snow_experiment.py': 'old'}}
            (root / 'manifest.json').write_text(json.dumps({'design': old, 'records': {}}))
            with self.assertRaises(ValueError):
                run_study(root, design, object())
