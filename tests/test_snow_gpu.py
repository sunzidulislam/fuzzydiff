import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from snow_gpu import SnowBackend, load_attention, save_attention, notebook_cells
from snow_experiment import make_design, jobs_for


class StoreFixture:
    def __init__(self, attn_res=16):
        self.attn_res = attn_res
        self.token_groups = {'snow': [5, 6], 'mountains': [8]}
        self.guidance_diagnostics = [{'step': 0, 'gradient_rms': 0.00001}]

    def get_average_attention(self):
        return {'down_cross': torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)}


class GPUAdapterTests(unittest.TestCase):
    def test_native_generation_metadata_handles_real_transformers_config(self):
        from transformers import CLIPTextConfig
        from diffusers import DPMSolverMultistepScheduler
        from PIL import Image
        class NativeFixture:
            scheduler = DPMSolverMultistepScheduler()
            unet = SimpleNamespace(config={'_commit_hash': 'unet-revision'})
            vae = SimpleNamespace(config={})
            text_encoder = SimpleNamespace(config=CLIPTextConfig(_commit_hash='text-revision'))
            text_encoder_2 = SimpleNamespace(config=CLIPTextConfig())
            def __call__(self, **kwargs):
                return SimpleNamespace(images=[Image.new('RGB', (8, 8))])
        backend = SnowBackend(make_design())
        backend.native = NativeFixture()
        backend.namespace = {'clip_prompt_similarities': lambda image, prompt: {'full_text': 0.3}}
        job = next(j for j in jobs_for(backend.design) if j['method'] == 'native')
        with patch.multiple(torch.cuda, reset_peak_memory_stats=lambda: None,
                            get_device_name=lambda index: 'CPU fixture', synchronize=lambda: None,
                            max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0):
            _, metadata = backend.generate(job, Path('.'))
        self.assertEqual(metadata['checkpoint_commits']['text_encoder'], 'text-revision')
        self.assertIsNone(metadata['checkpoint_commits']['text_encoder_2'])
        self.assertEqual(metadata['checkpoint_commits']['unet'], 'unet-revision')
        self.assertEqual(metadata['scheduler']['lambda_min_clipped'], '-inf')
        json.dumps(metadata, allow_nan=False)

    def test_attention_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'attention' / 'fixture.pt'
            store = StoreFixture()
            save_attention(file, store)
            recovered = load_attention(file, StoreFixture)
            self.assertEqual(recovered.token_groups, store.token_groups)
            self.assertEqual(recovered.guidance_diagnostics, store.guidance_diagnostics)
            self.assertEqual(recovered.counts, {'down_cross': 1})
            torch.testing.assert_close(recovered.attention_store['down_cross'], store.get_average_attention()['down_cross'])
            self.assertFalse(recovered.active)

    def test_plan_cli_without_gpu_or_files(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'never-created'
            for flag, expected in (([], 15), (['--full'], 225)):
                result = subprocess.run([sys.executable, str(repo / 'run_snow_experiment.py'),
                                         '--plan', '--output', str(output), *flag],
                                        capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout)['images'], expected)
            self.assertFalse(output.exists())
        self.assertEqual(len(list(notebook_cells())), 26)


if __name__ == '__main__':
    unittest.main()
