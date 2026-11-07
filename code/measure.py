"""Measure every estimator the current program needs, on one clean prediction.

Gradients reach the latent only through x0 and through the attention readout. Constraints read the
results from Evidence, so a candidate is scored with exactly the quantities its step was frozen on.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_stats import mixture_stats, srgb_to_oklab, trimmed_mean  # noqa: E402
from constraints import Evidence, mix_key  # noqa: E402
from estimators import area, centroid, exclusive_core, extent, object_photometric, region_crop, scene_lightness  # noqa: E402
from grounding import aggregate_maps, binding, binding_keys, overlap  # noqa: E402

MEASURE_ABLATIONS = ("attention_only", "overlap")


def measure_scene(backend, x0, out, program, bin):
    """Scene-level evidence only (M1): no segmentation or attention."""
    return Evidence(x0=x0, scene={"scene_lightness": scene_lightness(x0)})


@dataclass
class EstimatorDeps:
    segmenter: object
    siglip: object
    calib: object
    settings: object


class Measure:
    """Full evidence for one prompt: supports, presence, colour, size, and attention statistics."""

    def __init__(self, deps: EstimatorDeps, prompt: str, cond, ablation: str = ""):
        """ablation "attention_only": supports are the phrase attention maps instead of CLIPSeg;
        "overlap": binding is the plain overlap b(a, q) instead of the contrastive beta(a, q)."""
        if ablation and ablation not in MEASURE_ABLATIONS:
            raise ValueError(f"unknown measurement ablation {ablation!r}")
        self.deps, self.prompt, self.cond, self.ablation = deps, prompt, cond, ablation
        s = deps.settings
        self.box_k, self.erode = s.get("cfgBoxK"), s.get("cfgErode", int)
        self.tau, self.s_p = s.get("cfgSmaxTemp"), s.get("cfgSigmaP")
        self.core_threshold, self.core_temp = s.get("core_threshold"), s.get("core_temp")

    def __call__(self, backend, x0, out, program, bin) -> Evidence:
        x = x0[0] if x0.dim() == 4 else x0
        ev = Evidence(x0=x0, scene={"scene_lightness": scene_lightness(x)})
        objects = list(dict.fromkeys(program.objects))
        if objects:
            maps = None
            if self.ablation == "attention_only":
                maps = self._grounded(out) if out is not None and out.maps else {}
            self._objects(x, objects, ev, bin, maps)
        needs_colour = any(c.family in ("F6", "leakage") for c in program.constraints)
        lab = srgb_to_oklab(x) if needs_colour else None
        for c in program.constraints:
            if c.family == "F2a" and c.quantity != "scene_lightness" and c.args[0] in ev.objects:
                obj = ev.objects[c.args[0]]
                obj[c.quantity] = object_photometric(x, obj["core"], c.quantity)
            elif c.family == "F6" and c.args[0] in ev.objects:
                self._mixture(lab, c, ev)
            elif c.family == "leakage" and c.args[2] in ev.objects:
                self._leakage(lab, c, ev)
            elif c.family == "F8":
                self._size(c, ev)
        if out is not None and out.maps:
            self._attention(out, program, ev)
        return ev

    def _objects(self, x, objects, ev, bin, maps=None):
        d = self.deps
        supports = d.segmenter(x, objects) if maps is None else self._attention_supports(x, objects, maps)
        for q in objects:
            s = supports[q]
            region = region_crop(x, s, self.box_k)
            g, r = d.siglip.logits(x, region, q)
            h = d.calib.presence_logit(bin, g, r)
            others = [supports[r] for r in objects if r != q]
            core = exclusive_core(s, others, self.erode, self.core_threshold, self.core_temp)
            ev.objects[q] = {"S": s, "core": core, "A": area(s), "c": centroid(s), "h": h,
                             "g": g.detach(), "r": r.detach(),
                             "p": torch.sigmoid(h.detach()), "psi": d.calib.psi(q, {})}

    def _weights(self, core, lab):
        w = F.interpolate(core.detach()[None, None], size=lab.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return w.to(lab.dtype)

    def _prototype(self, term, lab):
        return self.deps.calib.prototype(term).to(lab.device, lab.dtype)

    def _mixture(self, lab, c, ev):
        obj = ev.objects[c.args[0]]
        base, tint = c.quantity
        c2 = self._prototype(tint, lab)
        if base == "achromatic":
            c1 = torch.stack([c2[0], torch.zeros_like(c2[0]), torch.zeros_like(c2[0])])
        else:
            c1 = self._prototype(base, lab)
        sigma_p = self.s_p * lab.shape[-1]
        obj[mix_key(c.quantity)] = mixture_stats(lab, self._weights(obj["core"], lab), c1, c2, sigma_p=sigma_p)

    def _leakage(self, lab, c, ev):
        """Colour test on the other object: how close its core colour is to this colour term."""
        term, _, other = c.args
        mean = trimmed_mean(lab, self._weights(ev.objects[other]["core"], lab))
        target = self._prototype(term, lab)
        scale = self.deps.calib.leakage_scale()
        ev.pairs[("leakage", term, other)] = torch.exp(-((mean[1:] - target[1:]) ** 2).sum() / (2 * scale ** 2))

    def _size(self, c, ev):
        q, r = c.args
        if q not in ev.objects or r not in ev.objects:
            return
        a, b = ev.objects[q], ev.objects[r]
        if c.quantity == "area":
            g_q, g_r = a["A"], b["A"]
        else:
            axis = "y" if c.quantity == "height" else "x"
            g_q, g_r = extent(a["S"], axis), extent(b["S"], axis)
        ev.pairs[("F8", q, r, c.quantity)] = (g_q, g_r)

    def _grounded(self, out):
        blocks = sorted(next(iter(out.maps.values())))
        return aggregate_maps(out, self.cond, blocks, self.deps.calib.attention_thresholds("objects"))

    def _attention_supports(self, x, objects, grounded):
        """Attention-only ablation: each object's normalized phrase map on the segmenter's grid.
        An object with no map gets an empty support, so its support gate closes."""
        grid = 64
        out = {}
        for q in objects:
            g = grounded.get(q) or grounded.get(f"{q}#0")
            if g is None or g.P.dim() != 2:
                out[q] = torch.zeros(grid, grid, dtype=x.dtype, device=x.device)
                continue
            p = F.interpolate(g.P[None, None].to(x.device, x.dtype), size=(grid, grid), mode="bilinear",
                              align_corners=False)[0, 0]
            out[q] = p.clamp(0, 1)
        return out

    def _attention(self, out, program, ev):
        grounded = self._grounded(out)
        for key, g in grounded.items():
            ev.attention[key] = {"P": g.P, "informative": g.informative, "peak": g.peak, "share": g.share,
                                 "mass": g.mass}
        heads = set(program.objects)
        for (a, q), (a_key, q_key) in binding_keys(self.prompt, program).items():
            others = [g.P for k, g in grounded.items() if k.split("#")[0] in heads and k != q_key]
            if a_key in grounded and q_key in grounded and others:
                if self.ablation == "overlap":
                    ev.pairs[("binding", a, q)] = overlap(grounded[a_key].P, grounded[q_key].P)
                else:
                    ev.pairs[("binding", a, q)] = binding(grounded[a_key].P, grounded[q_key].P, others, self.tau)
                ev.raw_ok[("binding", a, q)] = (
                    grounded[a_key].informative and grounded[q_key].informative)
