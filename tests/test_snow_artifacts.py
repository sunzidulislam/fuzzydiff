import csv
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from snow_experiment import make_design, jobs_for, run_study
from test_snow_experiment import FakeBackend
from snow_artifacts import export_artifacts
from analyze_snow_ratings import analyze


class ArtifactTests(unittest.TestCase):
    def test_shared_sheets_remain_blank_after_entering_private_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_study(root, make_design(), FakeBackend())
            export_artifacts(root)
            sheet = root / 'review_pack/reviewer_1.csv'
            with sheet.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row.update(snow_coverage='3', prompt_adherence='4', scene_preservation='3',
                           notes='Reviewer real name: Example Person')
            with sheet.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            completed = root / 'completed.csv'
            completed.write_bytes(sheet.read_bytes())
            export_artifacts(root)
            with zipfile.ZipFile(root / 'review_pack.zip') as archive:
                shared = archive.read('reviewer_1.csv').decode()
                self.assertNotIn('Example Person', shared)
                shared_rows = list(csv.DictReader(shared.splitlines()))
                self.assertTrue(all(row['snow_coverage'] == '' for row in shared_rows))

    def test_reference_change_rejects_old_preservation_ratings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_study(root, make_design(), FakeBackend())
            export_artifacts(root)
            with (root / 'review_pack/reviewer_1.csv').open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row.update(snow_coverage='3', prompt_adherence='4', scene_preservation='3')
            completed = root / 'completed.csv'
            manifest = json.loads((root / 'manifest.json').read_text())
            reference = next(record for record in manifest['runs'].values()
                             if record['job']['method'] == 'native' and record['job']['level'] == 2)
            image = root / reference['job']['image']
            from PIL import Image
            Image.new('RGB', (48, 48), 'white').save(image)
            reference['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(manifest))
            # Exclude the changed candidate itself; the other candidates' old reference must be rejected.
            unchanged_rows = [r for r in rows if r['image_sha256'] == manifest['runs'][
                json.loads((root / 'review_key.json').read_text())['images'][r['image_id']]]['image_sha256']]
            with completed.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(unchanged_rows)
            with self.assertRaisesRegex(ValueError, 'reference changed'):
                analyze(root, [completed])

    def test_artifact_cli_returns_failure_for_missing_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = run_study(root, make_design(), FakeBackend())
            image = next(iter(manifest['runs'].values()))['job']['image']
            (root / image).unlink()
            repo = Path(__file__).resolve().parents[1]
            result = subprocess.run([sys.executable, str(repo / 'run_snow_experiment.py'),
                                     '--artifacts', '--output', str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('14/15 images complete', result.stdout)

    def test_anonymous_pack_and_stable_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_study(root, make_design(), FakeBackend())
            export_artifacts(root)
            key = json.loads((root / 'review_key.json').read_text())
            # Allocate IDs for the entire design up front, keeping pilot IDs stable on expansion.
            self.assertEqual(len(key['images']), 225)
            self.assertTrue((root / 'grids/alpine_seed_142.pdf').exists())
            self.assertTrue((root / 'grids/alpine_seed_142.png').exists())
            with zipfile.ZipFile(root / 'review_pack.zip') as pack:
                self.assertFalse(any('key' in name or 'manifest' in name for name in pack.namelist()))
                self.assertFalse(any('fuzzy' in name or 'native' in name or 'level_' in name for name in pack.namelist()))
                coverage = pack.read('coverage_reviewer_1.html').decode()
                self.assertNotIn('very heavy', coverage)
                self.assertNotIn('FuzzyDiff', coverage)
                self.assertNotIn('realistic landscape', coverage)
            sheet = root / 'review_pack/reviewer_1.csv'
            with sheet.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(row['snow_coverage'] == '' for row in rows))
            before = sheet.read_bytes()
            export_artifacts(root)
            self.assertEqual(key, json.loads((root / 'review_key.json').read_text()))
            self.assertEqual(sheet.read_bytes(), before)

    def test_incomplete_pack_excludes_failed_image_and_reference_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = jobs_for(make_design())
            reference = next(j['id'] for j in jobs if j['method'] == 'native' and j['level'] == 2)
            run_study(root, make_design(), FakeBackend(fail=reference))
            export_artifacts(root)
            status = json.loads((root / 'artifact_status.json').read_text())
            self.assertEqual(status['complete'], 14)
            self.assertEqual(status['missing'], [reference])
            page = (root / 'review_pack/alignment_reviewer_1.html').read_text(encoding='utf-8')
            self.assertIn('Reference unavailable', page)

    def test_ratings_two_reviewers_disagreement_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_study(root, make_design(), FakeBackend())
            export_artifacts(root)
            paths = []
            for reviewer in (1, 2):
                source = root / f'review_pack/reviewer_{reviewer}.csv'
                with source.open(newline='', encoding='utf-8') as stream:
                    reader = csv.DictReader(stream)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    row.update(snow_coverage=str(reviewer), prompt_adherence='4', scene_preservation='3')
                target = root / f'completed_{reviewer}.csv'
                with target.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                paths.append(target)
            partial = analyze(root, paths[:1])
            self.assertFalse(partial['all_images_have_two_complete_reviews'])
            result = analyze(root, paths)
            self.assertTrue(result['all_images_have_two_complete_reviews'])
            self.assertEqual(result['image_summaries'][0]['snow_coverage']['range'], 1)
            self.assertEqual(result['image_summaries'][0]['snow_coverage']['mean'], 1.5)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                analyze(root, [paths[0], paths[0]])
            bad = paths[0].read_text().replace(',4,3,', ',6,3,')
            paths[0].write_text(bad)
            with self.assertRaisesRegex(ValueError, '1.*5'):
                analyze(root, paths)


if __name__ == '__main__':
    unittest.main()
