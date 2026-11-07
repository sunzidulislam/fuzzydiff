"""Rebuild a run folder from a downloaded rating page and its key, when the run itself is lost.

The page embeds every panel as a 512 px JPEG (quality 90), so restored images are not the 1024 px PNGs
the run wrote. Similarity is measured at CLIP's 224 px and barely moves; CLIP-IQA runs at the image's
own size, so its absolute values differ from a full-resolution run (every arm is restored the same way,
so paired comparisons stay fair). Each meta.json says the image was restored.

  python -I code/restore_run.py --page fmas.html --key fmas.key.json --out outputs/fmas
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

FIGURE = re.compile(r'<figure data-panel="([A-Z])" data-group="([0-9a-f]+)">.*?'
                    r'src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)"', re.S)


def restore(page: Path, key: Path, out: Path) -> int:
    """Write out/<arm>/<group>/image.png and meta.json for every panel of the page; returns the count."""
    with open(key) as f:
        groups = json.load(f)
    n = 0
    for letter, group, data in FIGURE.findall(Path(page).read_text(encoding="utf-8")):
        entry = groups[group]
        arm = entry["panels"][letter]
        d = Path(out) / arm / group
        d.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB").save(d / "image.png")
        (d / "meta.json").write_text(json.dumps({"prompt": entry["prompt"], "arm": arm,
                                                 "restored_from": Path(page).name}))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(f"restored {restore(args.page, args.key, args.out)} images into {args.out}")


if __name__ == "__main__":
    main()
