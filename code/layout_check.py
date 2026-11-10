"""Does Stage 1 change the picture? One prompt under several guidance settings, read from the run folders.

For each setting it reports how far the latent was moved, what the size score did, how many pixels
changed against SD3, and whether the image is still a photo. A verdict is a first reading of numbers,
not a result: look at the sheet. Nothing here is paper evidence.

Folders: <root>/<setting>/stage1/<job>/ and one <root>/<setting>/unguided/<job>/ (SD3, same seed).
Usage: python -I layout_check.py --root /kaggle/working/outputs/exp3_one
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

UNCHANGED_BELOW = 8.0       # mean pixel difference to SD3 (0-255); the runs that looked the same were 2.5 to 6
BROKEN_ABOVE = 0.2          # share of saturated pixels; the 30 photos of Exp. iii were below 0.07, the two flat images above 0.47


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def step_rows(records):
    """One row per guided step: what was tried, what was accepted, and the size reading at that step."""
    size = {r["step"]: r for r in records if r["type"] == "constraint" and r["family"] == "F8" and r["component"] == "main"}
    rows = []
    for r in sorted((r for r in records if r["type"] == "step"), key=lambda r: r["step"]):
        accepted = bool(r["accepted"])
        # every rejection halves the step, and direction has RMS 1, so this is the RMS change of the latent
        moved = r["eta"] / 2 ** len(r["rejections"]) if accepted and r.get("eta") else 0.0
        f8 = size.get(r["step"], {})
        rows.append({"step": r["step"], "sigma": r["sigma"], "accepted": accepted, "moved": moved,
                     "loss_before": sum(r["loss_before"].values()), "loss_after": sum(r["loss_after"].values()),
                     "size_active": bool(f8.get("active")), "size_score": f8.get("raw"), "areas": f8.get("areas"),
                     "rejections": r["rejections"]})
    return rows


def summarize(rows, final, sd3_final):
    """Totals of one run. final and sd3_final are the F8 records of the finished images (or None)."""
    active = [r for r in rows if r["size_active"]]
    return {"guided_steps": len(rows), "accepted": sum(r["accepted"] for r in rows),
            "size_steps": len(active), "moved_total": sum(r["moved"] for r in rows),
            "score_sd3": sd3_final and sd3_final["s"], "score": final and final["s"],
            "mu_sd3": sd3_final and sd3_final["mu"], "mu": final and final["mu"],
            "low": final and final["low"], "high": final and final["high"]}


def pixel_change(image, reference):
    """Mean absolute difference per channel value, 0 to 255."""
    from PIL import ImageChops, ImageStat
    a, b = image.convert("RGB"), reference.convert("RGB").resize(image.size)
    return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3


def saturated_share(image):
    """Share of pixels with a channel at black or white: large flat saturated areas mean the image left the photo range."""
    from PIL import ImageChops
    r, g, b = image.convert("RGB").split()
    top = ImageChops.lighter(ImageChops.lighter(r, g), b).point(lambda v: 255 if v >= 253 else 0)
    bottom = ImageChops.darker(ImageChops.darker(r, g), b).point(lambda v: 255 if v <= 2 else 0)
    return ImageChops.lighter(top, bottom).histogram()[255] / (image.size[0] * image.size[1])


def verdict(change, saturated, s):
    if saturated > BROKEN_ABOVE:
        return "broken: flat saturated image, guidance too strong"
    if change < UNCHANGED_BELOW:
        if s["score"] is not None and s["score_sd3"] is not None and abs(s["score"] - s["score_sd3"]) > 0.2:
            return "same picture, but the score moved: the estimator changed, not the image"
        return "same picture as SD3: guidance too weak"
    return "picture changed: look at the sheet"


def _size_final(folder):
    meta = json.loads((folder / "meta.json").read_text())
    f8 = [f for f in meta.get("finals", []) if f["family"] == "F8" and f["component"] == "main"]
    return meta, (f8[0] if f8 else None)


def _fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def check(root, out=None):
    """Print the table of every setting under root, write steps.csv and sheet.jpg. Returns the summaries."""
    from PIL import Image, ImageDraw
    root = Path(root)
    out = Path(out) if out else root
    sd3_dirs = sorted(root.glob("*/unguided/*/meta.json"))
    if not sd3_dirs:
        raise SystemExit(f"no unguided image under {root}")
    sd3_dir = sd3_dirs[0].parent
    job = sd3_dir.name
    sd3_meta, sd3_final = _size_final(sd3_dir)
    sd3_image = Image.open(sd3_dir / "image.png")
    print("prompt:", sd3_meta["prompt"])
    if sd3_final:
        print(f"size score = log(area ratio), target {_fmt(sd3_final['low'])} to {_fmt(sd3_final['high'])};"
              f" SD3 scores {_fmt(sd3_final['s'])} (mu {_fmt(sd3_final['mu'])})")
    panels, results, csv = [("SD3", sd3_image)], {}, ["setting,step,sigma,accepted,moved,loss_before,loss_after,size_active,size_score,area_1,area_2,rejections"]
    print(f"\n{'setting':16s} {'steps':>5s} {'acc.':>4s} {'size':>4s} {'moved':>6s} {'score':>6s} {'mu':>5s} {'pixels':>6s} {'sat.':>5s}  verdict")
    for folder in sorted(p.parent for p in root.glob(f"*/stage1/{job}/meta.json")):
        name = folder.parent.parent.name
        _, final = _size_final(folder)
        rows = step_rows(read_jsonl(folder / "log.jsonl"))
        s = summarize(rows, final, sd3_final)
        image = Image.open(folder / "image.png")
        change, sat = pixel_change(image, sd3_image), saturated_share(image)
        s.update(pixel_change=change, saturated=sat, verdict=verdict(change, sat, s))
        results[name] = s
        panels.append((f"{name}  score {_fmt(s['score'])}", image))
        print(f"{name:16s} {s['guided_steps']:5d} {s['accepted']:4d} {s['size_steps']:4d} {s['moved_total']:6.3f} "
              f"{_fmt(s['score']):>6s} {_fmt(s['mu']):>5s} {change:6.1f} {sat:5.2f}  {s['verdict']}")
        for r in rows:
            areas = r["areas"] or [None, None]
            csv.append(",".join(str(v) for v in (name, r["step"], f"{r['sigma']:.3f}", int(r["accepted"]), f"{r['moved']:.4f}",
                                                 f"{r['loss_before']:.3f}", f"{r['loss_after']:.3f}", int(r["size_active"]),
                                                 _fmt(r["size_score"], 3), _fmt(areas[0], 3), _fmt(areas[1], 3),
                                                 "|".join(r["rejections"]))))
    print("\nsteps: guided steps; acc.: accepted; size: steps where the size constraint was active; moved: summed RMS change"
          "\nof the latent (its own RMS is about 1); pixels: mean difference to SD3, 0-255; sat.: share of saturated pixels")
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps.csv").write_text("\n".join(csv) + "\n")
    side = 512
    sheet = Image.new("RGB", ((side + 12) * len(panels), side + 48), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), sd3_meta["prompt"], fill="black")
    for i, (label, image) in enumerate(panels):
        draw.text((i * (side + 12) + 8, 26), label, fill="black")
        sheet.paste(image.convert("RGB").resize((side, side)), (i * (side + 12), 44))
    sheet.save(out / "sheet.jpg", quality=90)
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    print("wrote", out / "sheet.jpg", "and", out / "steps.csv")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check(args.root, args.out)
