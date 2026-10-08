"""Visualize actual DINO final-layer CLS-to-patch self-attention.

This is model self-attention, not segmentation or class-specific attribution.
Reference: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from map import read_image, save_png


def render(values, image):
    low, high = float(values.min()), float(values.max())
    normalized = (values - low) / max(high - low, 1e-12)
    scaled = cv2.resize(normalized, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((scaled * 255).round().astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
    return heatmap, overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--out', type=Path, default=Path('attention_results/vision_attention'))
    parser.add_argument('--model', default='facebook/dino-vits16')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    print(f'Loading pretrained model: {args.model}', flush=True)
    processor = ViTImageProcessor.from_pretrained(args.model)
    model = ViTModel.from_pretrained(args.model, attn_implementation='eager').eval()
    photo = Image.open(args.image).convert('RGB')
    inputs = processor(images=photo, return_tensors='pt')
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True)
    if not outputs.attentions or outputs.attentions[-1] is None:
        raise RuntimeError('Model did not return attention weights.')
    attention = outputs.attentions[-1][0].float().cpu().numpy()
    if not np.isfinite(attention).all() or np.any(attention < 0):
        raise RuntimeError('Invalid model attention probabilities.')
    np.testing.assert_allclose(attention.sum(-1), 1, atol=1e-5)
    height, width = inputs['pixel_values'].shape[-2:]
    patch = model.config.patch_size
    grid = (height // patch, width // patch)
    heads = attention[:, 0, 1:].reshape(attention.shape[0], *grid)
    np.savez_compressed(args.out / 'raw_attention.npz',
                        final_layer_attention=attention, cls_to_patch=heads)
    image = read_image(args.image)
    mean = heads.mean(0)
    heatmap, overlay = render(mean, image)
    save_png(args.out / 'attention_heatmap.png', heatmap)
    save_png(args.out / 'attention_overlay.png', overlay)
    panels = []
    for index, head in enumerate(heads):
        heat, over = render(head, image)
        save_png(args.out / f'head_{index}_heatmap.png', heat)
        save_png(args.out / f'head_{index}_overlay.png', over)
        panel = cv2.copyMakeBorder(over, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, f'Head {index}', (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    save_png(args.out / 'attention_heads.png', np.hstack(panels))
    metadata = {
        'model': args.model,
        'revision': getattr(model.config, '_commit_hash', None),
        'input_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'method': 'Final transformer layer, CLS query to image patch keys, mean over all heads',
        'layer_index': len(outputs.attentions) - 1,
        'heads': attention.shape[0],
        'patch_grid': grid,
        'model_input_size': [int(width), int(height)],
        'display': 'Min-max scaling of mean patch weights; bilinear resize; JET colormap',
        'overlay_heatmap_alpha': 0.6,
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'limitation': 'Self-attention is not a hand mask, class-specific attribution, or SDXL cross-attention.',
    }
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)
    print(f'Saved real attention maps to {args.out}', flush=True)


if __name__ == '__main__':
    main()
