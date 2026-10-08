#!/usr/bin/env python3
"""Extract foreground with GrabCut and save mask-based visualizations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def save_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def foreground_mask(image, rect=None, select=False, iterations=6, foreground_points=None):
    """Rectangle-guided GrabCut; all requested objects must be inside the rectangle."""
    h, w = image.shape[:2]
    if min(h, w) < 16:
        raise ValueError('Use an image at least 16 x 16 pixels.')
    if select:
        # Resize only the selection window; map coordinates back to original pixels.
        scale = min(1.0, 1100 / w, 750 / h)
        display = cv2.resize(image, (round(w * scale), round(h * scale)))
        chosen = cv2.selectROI('Select ALL wanted objects, then press Enter', display,
                               showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        if chosen[2] == 0 or chosen[3] == 0:
            raise ValueError('Selection cancelled.')
        rect = tuple(round(v / scale) for v in chosen)
    if rect is None:
        margin = max(1, round(min(h, w) * 0.025))
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    x, y, rw, rh = map(int, rect)
    if x < 0 or y < 0 or rw < 2 or rh < 2 or x + rw > w or y + rh > h:
        raise ValueError('Rectangle must fit inside the image and have positive size.')
    if rw * rh >= w * h:
        raise ValueError('Leave some background outside the rectangle for GrabCut.')
    labels = np.zeros((h, w), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(7)
    cv2.grabCut(image, labels, (x, y, rw, rh), bg, fg, iterations,
                cv2.GC_INIT_WITH_RECT)
    if foreground_points:
        # Reconsider rejected pixels inside the rectangle using user guidance.
        region = labels[y:y + rh, x:x + rw]
        region[region == cv2.GC_BGD] = cv2.GC_PR_BGD
        for px, py in foreground_points:
            if not (x <= px < x + rw and y <= py < y + rh):
                raise ValueError('Foreground points must lie inside the rectangle.')
            cv2.circle(labels, (px, py), 3, cv2.GC_FGD, -1)
        cv2.grabCut(image, labels, None, bg, fg, iterations, cv2.GC_INIT_WITH_MASK)
    return np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255


def make_visualizations(mask, blur=0.018, image=None):
    """Construct a cyan glow illustration from a mask; it is not learned attention."""
    h, w = mask.shape
    m = (mask > 127).astype(np.float32)
    sigma = max(1.0, min(h, w) * blur)
    soft = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma)
    edge = cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(np.float32) / 255
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=max(0.7, sigma * 0.75))
    if glow.max() > 0:
        glow /= glow.max()
    strength = np.clip(0.30 * soft + 0.70 * glow, 0, 1)
    strength = strength[..., None]
    # Colors expressed in BGR for OpenCV.
    dark = np.array([48, 43, 12], np.float32)
    cyan = np.array([236, 237, 184], np.float32)
    attention_style = np.clip(dark + strength * (cyan - dark), 0, 255).astype(np.uint8)
    if image is not None:
        # Preserve faint scene texture and subject detail beneath a diffuse glow.
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        luminance = cv2.GaussianBlur(luminance, (0, 0), sigmaX=max(1.0, sigma * 0.75))
        background_detail = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma)
        teal_dark = np.array([48, 48, 8], np.float32)
        teal_light = np.array([142, 145, 55], np.float32)
        detail = background_detail * (1 - soft) + luminance * soft
        base = teal_dark + detail[..., None] * (teal_light - teal_dark)
        base += soft[..., None] * np.array([14, 18, 5], np.float32)
        # Outer mask edges cannot describe a hand resting over a dress. Retain
        # photo edges within the foreground to show those interior boundaries.
        detail_image = cv2.GaussianBlur(image, (3, 3), 0.7).astype(np.float32)
        blue, green, red = cv2.split(detail_image)
        skin = ((red - blue > 20) & (red > green * 1.07)).astype(np.uint8) * 255
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        skin[mask == 0] = 0
        skin[:h // 2] = 0
        count, components, stats, _ = cv2.connectedComponentsWithStats(skin)
        arms = np.zeros_like(skin)
        for component in range(1, count):
            _, _, cw, ch, _ = stats[component]
            if ch > h * 0.20 and ch > cw * 1.5:
                arms[components == component] = 255
        interior_edges = cv2.morphologyEx(arms, cv2.MORPH_GRADIENT,
                                         np.ones((3, 3), np.uint8)).astype(np.float32) / 255
        interior_edges[:h // 2] = 0
        interior = cv2.erode((m * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
        interior_edges *= interior.astype(np.float32) / 255
        interior_glow = cv2.GaussianBlur(interior_edges, (0, 0), sigmaX=1.0)
        if interior_glow.max() > 0:
            interior_glow /= interior_glow.max()
        luminous = np.array([245, 255, 194], np.float32)
        halo = np.clip(glow * 0.95 + interior_glow * 0.60, 0, 1)[..., None]
        attention_style = np.clip(base * (1 - halo) + luminous * halo, 0, 255)
        attention_style = cv2.GaussianBlur(attention_style, (0, 0), sigmaX=0.8).astype(np.uint8)
    beige = np.array([130, 146, 159], np.float32)
    cream = np.array([229, 236, 242], np.float32)
    styled_mask = (beige + m[..., None] * (cream - beige)).astype(np.uint8)
    return attention_style, styled_mask, (soft * 255).round().astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', type=Path)
    p.add_argument('--out', type=Path, default=Path('attention_results'))
    p.add_argument('--fg-point', type=int, nargs=2, action='append', metavar=('X', 'Y'),
                   help='Mark a known foreground point; repeat for missed areas.')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--rect', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    group.add_argument('--select', action='store_true', help='Draw a box in a desktop window.')
    group.add_argument('--mask', type=Path, help='Existing same-size mask; white = foreground.')
    args = p.parse_args()
    try:
        image = read_image(args.image)
        if args.mask is not None:
            mask = read_image(args.mask, cv2.IMREAD_GRAYSCALE)
            if mask.shape != image.shape[:2]:
                raise ValueError('Mask dimensions must match the input image.')
            mask = (mask > 127).astype(np.uint8) * 255
        else:
            mask = foreground_mask(image, rect=args.rect, select=args.select,
                                   foreground_points=args.fg_point)
        if not np.any(mask):
            raise ValueError('No foreground found; try --rect or supply --mask.')
        attention, styled, soft = make_visualizations(mask, image=image)
        args.out.mkdir(parents=True, exist_ok=True)
        outputs = {
            'foreground.png': np.dstack((image, mask)),
            'foreground_mask.png': mask,
            'soft_mask.png': soft,
            'attention_style.png': attention,
            'styled_mask.png': styled,
        }
        for name, result in outputs.items():
            save_png(args.out / name, result)
        print(json.dumps({
            'input': str(args.image),
            'width': image.shape[1],
            'height': image.shape[0],
            'foreground_pixels': int(np.count_nonzero(mask)),
            'outputs': [str(args.out / name) for name in outputs],
        }, indent=2))
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
