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
from estimators import area, centroid, erode_core, extent, object_photometric, region_crop, scene_lightness  # noqa: E402
from grounding import aggregate_maps, binding, binding_keys  # noqa: E402


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

    def __init__(self, deps: EstimatorDeps, prompt: str, cond):
        self.deps, self.prompt, self.cond = deps, prompt, cond
        s = deps.settings
        self.box_k, self.erode = s.get("cfgBoxK"), s.get("cfgErode", int)
        self.tau, self.s_p = s.get("cfgSmaxTemp"), s.get("cfgSigmaP")

    def __call__(self, backend, x0, out, program, bin) -> Evidence:
        x = x0[0] if x0.dim() == 4 else x0
        ev = Evidence(x0=x0, scene={"scene_lightness": scene_lightness(x)})
        objects = list(dict.fromkeys(program.objects))
        if objects:
            self._objects(x, objects, ev, bin)
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

    def _objects(self, x, objects, ev, bin):
        d = self.deps
        supports = d.segmenter(x, objects)
        for q in objects:
            s = supports[q]
            region = region_crop(x, s, self.box_k)
            g, r = d.siglip.logits(x, region, q)
            h = d.calib.presence_logit(bin, g, r)
            ev.objects[q] = {"S": s, "core": erode_core(s, self.erode), "A": area(s), "c": centroid(s), "h": h,
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

    def _attention(self, out, program, ev):
        calib = self.deps.calib
        blocks = sorted(next(iter(out.maps.values())))
        grounded = aggregate_maps(out, self.cond, blocks, calib.attention_thresholds("objects"))
        for key, g in grounded.items():
            ev.attention[key] = {"P": g.P, "informative": g.informative, "peak": g.peak, "share": g.share,
                                 "mass": g.mass}
        heads = set(program.objects)
        for (a, q), (a_key, q_key) in binding_keys(self.prompt, program).items():
            others = [g.P for k, g in grounded.items() if k.split("#")[0] in heads and k != q_key]
            if a_key in grounded and q_key in grounded and others:
                ev.pairs[("binding", a, q)] = binding(grounded[a_key].P, grounded[q_key].P, others, self.tau)
                ev.raw_ok[("binding", a, q)] = (
                    grounded[a_key].informative and grounded[q_key].informative)
