import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from snow_experiment import make_design, jobs_for, run_study


class FakeBackend:
    def __init__(self, fail=None):
        self.calls = []
        self.phases = []
        self.fail = fail

    def prepare(self, method):
        self.phases.append(method)

    def generate(self, job, root):
        self.calls.append(job['id'])
        if job['id'] == self.fail:
            raise RuntimeError('generation interrupted')
        if job['method'] == 'fuzzy':
            (root / job['attention']).parent.mkdir(exist_ok=True)
            (root / job['attention']).write_bytes(b'attention fixture')
        return Image.new('RGB', (48, 48), (100, 140, 180)), {'clip': 0.25}

    def close(self):
        pass


class StudyTests(unittest.TestCase):
    def test_original_metadata_failure_can_resume_without_losing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = make_design()
            previous = json.loads(json.dumps(current))
            previous['code_sha256'] = {
                'snow_gpu.py': '7d48b27b7f334bf5cf8011881cdc62b45c76073a7862c25ec608bcc9c0e99f95',
                'snow_experiment.py': '14561514686917a6078271b2801afc12f3b68779cb529c9f5970ef0fc91840e7',
                'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb': 'dc0be5b2c7b0905f4aa3aa82fd85501fc13c4ec154c76d50a4cd587adc0dac91',
            }
            job = jobs_for(previous)[0]
            manifest = {'design': previous, 'scope': 'pilot', 'runs': {
                job['id']: {'job': job, 'status': 'failed', 'attempts': [
                    {'status': 'failed', 'error': "AttributeError: 'CLIPTextConfig' object has no attribute 'get'"}]}},
                'phase_times': []}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'review_key.json').write_text(json.dumps({'design': previous, 'images': {}}))
            result = run_study(root, current, FakeBackend())
            self.assertEqual(len(result['runs'][job['id']]['attempts']), 2)
            self.assertEqual(result['runs'][job['id']]['status'], 'complete')
            self.assertEqual(result['design'], current)
            self.assertEqual(result['code_upgrades'][0]['from'], previous['code_sha256'])
            self.assertEqual(json.loads((root / 'review_key.json').read_text())['design'], current)

    def test_design_counts_and_matched_prompts(self):
        design = make_design()
        pilot = jobs_for(design, False)
        full = jobs_for(design, True)
        self.assertEqual(len(pilot), 15)
        self.assertEqual(len(full), 225)
        self.assertEqual({j['seed'] for j in pilot}, {142})
        self.assertEqual({j['scene'] for j in pilot}, {'alpine'})
        self.assertEqual({j['seed'] for j in full}, {42, 84, 142, 202, 314})
        for level in range(5):
            group = [j for j in pilot if j['level'] == level]
            self.assertEqual(len({j['prompt'] for j in group}), 1)
            self.assertEqual({j['method'] for j in group}, {'native', 'fuzzy', 'refined'})

    def test_resume_and_missing_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = FakeBackend()
            manifest = run_study(root, make_design(), first)
            self.assertEqual(len(first.calls), 15)
            self.assertEqual(first.phases, ['native', 'fuzzy', 'refined'])
            second = FakeBackend()
            run_study(root, make_design(), second)
            self.assertEqual(second.calls, [])
            fuzzy = next(j for j in jobs_for(make_design(), False) if j['method'] == 'fuzzy')
            (root / fuzzy['attention']).unlink()
            third = FakeBackend()
            run_study(root, make_design(), third)
            self.assertEqual(third.calls, [fuzzy['id'], fuzzy['base_id'] + '_refined'])
            self.assertEqual(manifest['design']['precision']['custom_latents'], 'float32')

    def test_failure_persisted_and_only_failed_job_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = make_design()
            failed = next(j['id'] for j in jobs_for(design, False) if j['method'] == 'refined')
            manifest = run_study(root, design, FakeBackend(fail=failed))
            self.assertEqual(manifest['runs'][failed]['status'], 'failed')
            self.assertIn('generation interrupted', manifest['runs'][failed]['attempts'][0]['error'])
            self.assertEqual(sum(r['status'] == 'complete' for r in manifest['runs'].values()), 14)
            backend = FakeBackend()
            result = run_study(root, design, backend)
            self.assertEqual(backend.calls, [failed])
            self.assertEqual(len(result['runs'][failed]['attempts']), 2)

    def test_changed_settings_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_study(root, make_design(), FakeBackend())
            before = (root / 'manifest.json').read_bytes()
            changed = make_design()
            changed['settings']['base_steps'] = 4
            with self.assertRaisesRegex(ValueError, 'different'):
                run_study(root, changed, FakeBackend())
            self.assertEqual((root / 'manifest.json').read_bytes(), before)

    def test_full_expansion_keeps_pilot_and_rejects_scope_shrink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = make_design()
            run_study(root, design, FakeBackend())
            backend = FakeBackend()
            result = run_study(root, design, backend, full=True)
            self.assertEqual(len(backend.calls), 210)
            self.assertEqual(len(result['runs']), 225)
            with self.assertRaisesRegex(ValueError, '--full'):
                run_study(root, design, FakeBackend())

    def test_nonfinite_metadata_is_recorded_as_failure(self):
        class InvalidBackend(FakeBackend):
            def generate(self, job, root):
                image, metadata = super().generate(job, root)
                metadata['clip'] = float('nan')
                return image, metadata
        with tempfile.TemporaryDirectory() as directory:
            result = run_study(Path(directory), make_design(), InvalidBackend())
            self.assertTrue(all(record['status'] != 'complete' for record in result['runs'].values()))
            self.assertTrue(any('JSON' in record['attempts'][0]['error'] for record in result['runs'].values()
                                if record['attempts']))


if __name__ == '__main__':
    unittest.main()
