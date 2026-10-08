"""Render saved DINO attention with a more visible source photograph."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from map import read_image, save_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('attention', type=Path)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs='+', help='Average separately normalized selected heads.')
    parser.add_argument('--alpha', type=float, default=0.40)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--out', type=Path, default=Path('attention_results/vision_attention/attention_girl_clear.png'))
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1 or args.scale < 1:
        parser.error('Use alpha between 0 and 1 and a positive scale.')
    with np.load(args.attention) as data:
        heads = data['cls_to_patch']
        selected_heads = args.heads if args.heads is not None else [args.head]
        if any(index < 0 or index >= len(heads) for index in selected_heads):
            parser.error('Head index is outside the saved attention array.')
        selected = heads[selected_heads].copy()
        if args.heads is not None:
            low = selected.min(axis=(1, 2), keepdims=True)
            span = selected.max(axis=(1, 2), keepdims=True) - low
            selected = (selected - low) / np.maximum(span, 1e-12)
        values = selected.mean(0)
    if not np.isfinite(values).all():
        parser.error('Attention values must be finite.')
    source = read_image(args.image)
    height, width = source.shape[:2]
    image = cv2.resize(source, (width * args.scale, height * args.scale),
                       interpolation=cv2.INTER_CUBIC)
    normalized = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
    displayed = cv2.resize(normalized, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((displayed * 255).round().astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - args.alpha, heatmap, args.alpha, 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_png(args.out, overlay)
    metadata = {'source_attention': str(args.attention), 'heads': selected_heads,
                'layer_index': 11, 'heatmap_alpha': args.alpha,
                'scale': args.scale, 'normalization': 'min-max',
                'attention_interpolation': 'bilinear',
                'aggregation': 'Mean of per-head min-max-normalized weights' if args.heads is not None else 'Single head',
                'note': 'Selected heads, not the all-head average. Upscaling adds no model resolution.'}
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {args.out} ({image.shape[1]} x {image.shape[0]}); heads {selected_heads}.')


if __name__ == '__main__':
    main()
