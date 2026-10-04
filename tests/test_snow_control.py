import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from snow_control import SnowRenderer, control_design, run_control
from snow_control_figure import export_control_grid


def fixture():
    y, x = np.mgrid[:64, :64]
    rgb = np.stack([70 + (x * 3 + y) % 40, 65 + (x + y) % 40, 60 + (x + y * 2) % 40], axis=-1).astype('uint8')
    mask = y > 16 + np.abs(x - 32) // 2
    rgb[~mask] = [90, 150, 205]
    return Image.fromarray(rgb), mask


class SnowControlTests(unittest.TestCase):
    def test_control_counts_and_plan_cpu_only(self):
        design = control_design()
        self.assertEqual(len(design['coverage']), 11)
        self.assertEqual(len(design['thickness']), 9)
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            result = subprocess.run([sys.executable, str(repo / 'run_snow_control.py'), '--plan',
                                     '--output', str(output)], text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)['rendered_images'], 99)
            self.assertFalse(output.exists())

    def test_nested_coverage_fixed_geometry_and_thickness_progression(self):
        image, terrain = fixture()
        renderer = SnowRenderer(image, terrain, seed=142)
        previous = np.zeros(terrain.shape, dtype=bool)
        for coverage in control_design()['coverage']:
            mask = renderer.coverage_mask(coverage)
            self.assertFalse(np.any(previous & ~mask))
            self.assertLessEqual(abs(mask.sum() / terrain.sum() - coverage), 1 / terrain.sum())
            previous = mask
            means = []
            for thickness in control_design()['thickness']:
                output, metrics = renderer.render(coverage, thickness)
                rgb = np.asarray(output)
                np.testing.assert_array_equal(rgb[~terrain], np.asarray(image)[~terrain])
                self.assertAlmostEqual(metrics['selected_mask_coverage'], mask.sum() / terrain.sum())
                means.append(rgb[terrain].mean())
            self.assertTrue(all(a <= b for a, b in zip(means, means[1:])))
        clean, _ = renderer.render(0, 1)
        np.testing.assert_array_equal(np.asarray(clean), np.asarray(image))
        no_layer, _ = renderer.render(1, 0)
        np.testing.assert_array_equal(np.asarray(no_layer), np.asarray(image))
        first, _ = renderer.render(0.6, 0.7)
        second, _ = SnowRenderer(image, terrain, seed=142).render(0.6, 0.7)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_invalid_mask_and_parameters(self):
        image, terrain = fixture()
        for mask in (np.zeros_like(terrain), np.ones((5, 5)), np.full(terrain.shape, np.nan)):
            with self.assertRaises(ValueError):
                SnowRenderer(image, mask)
        renderer = SnowRenderer(image, terrain)
        for coverage, thickness in ((-1, 0.5), (0.5, 2), (float('nan'), 0.5)):
            with self.assertRaises(ValueError):
                renderer.render(coverage, thickness)

    def test_resume_preserves_source_and_recovers_missing_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, terrain = fixture()
            source, mask = root / 'input.png', root / 'mask.png'
            image.save(source)
            Image.fromarray(terrain.astype('uint8') * 255).save(mask)
            output = root / 'experiment'
            design = control_design(source=source, mask=mask)
            manifest = run_control(output, design, source=source, mask=mask)
            self.assertEqual(len(manifest['runs']), 99)
            self.assertTrue(all(r['status'] == 'complete' for r in manifest['runs'].values()))
            saved_source = (output / 'source.png').read_bytes()
            attempts = {k: len(r['attempts']) for k, r in manifest['runs'].items()}
            first = next(iter(manifest['runs']))
            (output / manifest['runs'][first]['image']).unlink()
            resumed = run_control(output, design)
            self.assertEqual((output / 'source.png').read_bytes(), saved_source)
            self.assertEqual(len(resumed['runs'][first]['attempts']), attempts[first] + 1)
            self.assertTrue(all(len(r['attempts']) == attempts[k] for k, r in resumed['runs'].items() if k != first))
            changed = json.loads(json.dumps(design))
            changed['seed'] = 42
            with self.assertRaisesRegex(ValueError, 'different'):
                run_control(output, changed)
            (output / 'mountain_mask.png').unlink()
            with self.assertRaisesRegex(ValueError, 'mask'):
                run_control(output, design)

    def test_figure_layout_and_missing_cell_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, terrain = fixture()
            source, mask = root / 'input.png', root / 'mask.png'
            image.save(source)
            Image.fromarray(terrain.astype('uint8') * 255).save(mask)
            output = root / 'experiment'
            design = control_design(source=source, mask=mask)
            manifest = run_control(output, design, source=source, mask=mask)
            first = next(iter(manifest['runs']))
            (output / manifest['runs'][first]['image']).unlink()
            status = export_control_grid(output)
            self.assertEqual((status['rows'], status['columns']), (11, 9))
            self.assertEqual(status['complete'], 98)
            self.assertEqual(status['missing'], [first])
            self.assertTrue((output / 'snow_control_grid.pdf').exists())
            self.assertTrue((output / 'snow_control_grid.png').exists())



if __name__ == '__main__':
    unittest.main()
