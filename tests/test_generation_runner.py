import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class GenerationTests(unittest.TestCase):
    def test_malformed_config_and_example_requests_are_validated_without_models(self):
        from fuzzydiff.config import load_request_config, merge_request_config
        root = Path(__file__).resolve().parents[1]
        for name in ('generate_fuzzydiff', 'generate_sdxl_turbo'):
            request = merge_request_config(load_request_config(root / f'configs/{name}.json'), {})
            self.assertEqual(request.seed, 142)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for text in ('[1]', '{broken'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_request_config(path)

    def test_single_generation_saves_final_image_and_cleans_up(self):
        from PIL import Image
        from fuzzydiff.generation.requests import GenerationRequest
        from fuzzydiff.generation.runner import generate_image, save_result
        events = []
        class Session:
            last_details = {'truth': .99}
            def generate(self, seed):
                events.append(seed)
                return Image.new('RGB', (8, 8), 'red')
            def metadata(self):
                return {'effective_sampling_settings': {'steps': 4, 'guidance': 0}}
            def close(self):
                events.append('closed')
        with patch('fuzzydiff.generation.runner.create_session', return_value=Session()):
            result = generate_image(GenerationRequest('enhancing', 'a red ball', seed=42))
        self.assertEqual(events, [42, 'closed'])
        self.assertEqual(result.image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result.metadata['model'], 'sdxl_turbo')
        with tempfile.TemporaryDirectory() as directory:
            files = save_result(result, Path(directory), 42)
            self.assertEqual(files['image'].name, 'seed_42.png')
            metadata = json.loads(files['metadata'].read_text())
            self.assertEqual(metadata['diagnostics'], {'truth': .99})

    def test_generation_failure_closes_session(self):
        from fuzzydiff.generation.requests import GenerationRequest
        from fuzzydiff.generation.runner import generate_image
        events = []
        class Session:
            def generate(self, seed):
                raise ValueError('sampling failed')
            def close(self):
                events.append('closed')
        with patch('fuzzydiff.generation.runner.create_session', return_value=Session()):
            with self.assertRaisesRegex(ValueError, 'sampling failed'):
                generate_image(GenerationRequest('stable', 'ball'))
        self.assertEqual(events, ['closed'])

    def test_config_flags_override_and_invalid_settings_fail_before_loading(self):
        from fuzzydiff.config import merge_request_config
        request = merge_request_config({'model': 'fuzzydiff', 'prompt': 'mostly blue vase',
            'words': ['mostly blue', 'vase'], 'method_options': {'sharpness': 20, 'lr': 5}},
            {'seed': 999, 'method_options': {'lr': .2}})
        self.assertEqual(request.seed, 999)
        self.assertEqual(request.method_options, {'sharpness': 20, 'lr': .2})
        for bad in ({'unknown': 1}, {'seed': -1}, {'size': 513},
                    {'method_options': {'made_up': 1}}, {'words': ['missing']}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                merge_request_config({'model': 'stable', 'prompt': 'a ball'}, bad)

    def test_cli_displays_only_final_image_once(self):
        from PIL import Image
        from fuzzydiff.generation.requests import GenerationResult
        from fuzzydiff.cli.generate import main
        result = GenerationResult(Image.new('RGB', (8, 8)), {}, {})
        seen = []
        with patch('fuzzydiff.cli.generate.generate_image', return_value=result), \
                patch('fuzzydiff.cli.generate.display_image', side_effect=seen.append):
            self.assertEqual(main(['--model', 'stable', '--prompt', 'ball', '--display']), 0)
        self.assertEqual(seen, [result.image])
