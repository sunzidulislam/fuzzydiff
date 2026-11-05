"""Frozen visual estimators on the clean prediction (paper: Tab. estimators).

Analytic colour statistics take images in [0, 1] as [3, H, W] or [1, 3, H, W]. Support weights enter
photometric statistics with stop-gradient, so guidance changes pixel colours inside the object and
not the mask. Area, centroid, and extent keep the gradient through the support, because those
predicates are about the mask. Preprocessing runs inside the autograd graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS_C = 1e-6
GRAY_FILL = 0.5
MIN_HALF_EXTENT = 0.05
_RGB_TO_XYZ = [[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]]
_WHITE_D65 = (0.95047, 1.0, 1.08883)
_PREFIXES = ("a photo of", "a close-up photo of", "an image of")
_AN_EXCEPTIONS = {"hour", "honest", "heir"}
_A_EXCEPTIONS = {"uniform", "unicorn", "university", "one", "useful", "user"}


def _chw(x):
    return x[0] if x.dim() == 4 else x


def pool_grid(x, size=64):
    return F.adaptive_avg_pool2d(_chw(x).unsqueeze(0), size)[0]


def srgb_to_cielab(x):
    """sRGB -> CIELAB (D65), differentiable everywhere (both power laws have linear toes).

    No clamp here: Backend.decode already clamps with a straight-through gradient, and a second
    clamp would zero the gradient of saturated pixels."""
    x = _chw(x)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4)
    m = torch.as_tensor(_RGB_TO_XYZ, dtype=x.dtype, device=x.device)
    xyz = torch.einsum("ij,jhw->ihw", m, lin)
    white = torch.as_tensor(_WHITE_D65, dtype=x.dtype, device=x.device).view(3, 1, 1)
    t = xyz / white
    d = 6 / 29
    f = torch.where(t > d ** 3, t.clamp_min(d ** 3) ** (1 / 3), t / (3 * d ** 2) + 4 / 29)
    return torch.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def scene_lightness(x):
    """Mean L* of the whole image on the 64^2 grid (F2a scene lightness)."""
    return srgb_to_cielab(pool_grid(x))[0].mean()


def object_photometric(x, s_core, quantity):
    """Core-weighted mean L* ("lightness") or C*_ab ("chroma") on the 64^2 grid."""
    w = s_core.detach().to(x.device, x.dtype)
    lab = srgb_to_cielab(pool_grid(x, w.shape[-1]))
    if quantity == "lightness":
        value = lab[0]
    elif quantity == "chroma":
        value = torch.sqrt(lab[1] ** 2 + lab[2] ** 2 + 1e-8)
    else:
        raise ValueError(f"unknown photometric quantity {quantity}")
    return (value * w).sum() / (w.sum() + EPS_C)


def erode_core(s, k=3):
    """S_core = -maxpool_k(-S): excludes boundary and background bleed."""
    return -F.max_pool2d(-s[None, None], k, stride=1, padding=k // 2)[0, 0]


# ---------------------------------------------------------------------------- support geometry
def _coords(s):
    h, w = s.shape
    ys = (torch.arange(h, dtype=s.dtype, device=s.device) + 0.5) / h
    xs = (torch.arange(w, dtype=s.dtype, device=s.device) + 0.5) / w
    return ys.view(-1, 1), xs.view(1, -1)


def area(s):
    return s.mean()


def centroid(s, eps=EPS_C):
    """Support-weighted centre as (x, y) in [0, 1]^2."""
    ys, xs = _coords(s)
    mass = s.sum() + eps
    return torch.stack([(s * xs).sum() / mass, (s * ys).sum() / mass])


def extent(s, axis, eps=EPS_C):
    """sqrt(12 Var) of the normalized coordinate: exact for a uniformly filled axis-aligned box."""
    ys, xs = _coords(s)
    u = ys if axis == "y" else xs
    mass = s.sum() + eps
    mean = (s * u).sum() / mass
    var = (s * (u - mean) ** 2).sum() / mass
    return torch.sqrt(12 * var + eps)


def _gaussian_blur(x, sigma):
    r = max(int(3 * sigma), 1)
    grid = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-grid ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def region_crop(x, s, box_k, size=384):
    """Eq. region_sampling: soft mask with gray fill, sampled in an affine frame held by stop-gradient."""
    x = _chw(x)
    s = s.to(x.device, x.dtype)
    upsampled = F.interpolate(s[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)
    soft = _gaussian_blur(upsampled, max(x.shape[-1] / 64, 1.0))[0]
    masked = soft * x + (1 - soft) * GRAY_FILL
    with torch.no_grad():
        frame = s.detach()
        if frame.sum() <= EPS_C:
            cx, cy, half = 0.5, 0.5, 0.5
        else:
            ys, xs = _coords(frame)
            mass = frame.sum()
            cx, cy = float((frame * xs).sum() / mass), float((frame * ys).sum() / mass)
            std_x = float(((frame * (xs - cx) ** 2).sum() / mass).sqrt())
            std_y = float(((frame * (ys - cy) ** 2).sum() / mass).sqrt())
            half = min(max(box_k * max(std_x, std_y), MIN_HALF_EXTENT), 1.0)
        theta = torch.tensor([[[2 * half, 0.0, 2 * cx - 1], [0.0, 2 * half, 2 * cy - 1]]], dtype=x.dtype, device=x.device)
        grid = F.affine_grid(theta, (1, 3, size, size), align_corners=False)
    return F.grid_sample(masked[None], grid, mode="bilinear", padding_mode="border", align_corners=False)


# ---------------------------------------------------------------------------- CLIPSeg and SigLIP
def _frozen(model):
    model.eval()
    model.requires_grad_(False)
    return model


def _checkpoint_vision(model, vision):
    """Recompute vision layers in the backward pass. These layers have no dropout, so setting their
    training flag (which the library checks before checkpointing) changes nothing else."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vision.encoder.training = True
    for layer in vision.encoder.layers:
        layer.training = True


class ClipSegSupport:
    """Support S_q: sigmoid of pre-threshold CLIPSeg logits, area-averaged to the grid.

    Runs the vision encoder itself, because CLIPSegForImageSegmentation.forward wraps it in no_grad,
    which would cut the gradient to the pixels."""

    def __init__(self, model, encode_text, mean, std, size=352, grid=64):
        self.model, self.encode_text, self.size, self.grid = _frozen(model), encode_text, size, grid
        _checkpoint_vision(self.model, self.model.clip.vision_model)
        self.mean, self.std = mean, std
        self._texts = {}

    @classmethod
    def from_pretrained(cls, name, device, dtype=torch.float32):
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
        processor = AutoProcessor.from_pretrained(name)
        model = CLIPSegForImageSegmentation.from_pretrained(name, torch_dtype=dtype).to(device)

        def encode_text(phrase):
            tokens = processor.tokenizer([phrase], padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                return model.clip.get_text_features(**tokens)

        ip = processor.image_processor
        return cls(model, encode_text, tuple(ip.image_mean), tuple(ip.image_std),
                   size=processor_size(ip, model.config.vision_config.image_size))

    def _text(self, phrase):
        if phrase not in self._texts:
            self._texts[phrase] = self.encode_text(phrase)
        return self._texts[phrase]

    def __call__(self, x, phrases):
        x = _chw(x)
        param = next(self.model.parameters())
        img = F.interpolate(x[None].to(param.device, param.dtype), size=(self.size, self.size), mode="bilinear",
                            align_corners=False, antialias=True)
        mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        vision = self.model.clip.vision_model(pixel_values=(img - mean) / std, output_hidden_states=True, return_dict=True)
        activations = [vision.hidden_states[i + 1] for i in self.model.extract_layers]
        out = {}
        for phrase in phrases:
            cond = self._text(phrase).to(img.device, img.dtype)
            logits = self.model.decoder(activations, cond, return_dict=True).logits
            logits = logits.reshape(1, 1, *logits.shape[-2:])
            out[phrase] = F.adaptive_avg_pool2d(torch.sigmoid(logits), self.grid)[0, 0].to(x.device, x.dtype)
        return out


def _article(word):
    w = word.split()[0].lower()
    if w in _AN_EXCEPTIONS:
        return "an"
    if w in _A_EXCEPTIONS:
        return "a"
    return "an" if w[0] in "aeiou" else "a"


def text_templates(phrase):
    """Reference templates T^0_k(q) of App. estimators."""
    return [f"{prefix} {_article(phrase)} {phrase}" for prefix in _PREFIXES]


class SigLIPEvidence:
    """Global and region logits for one object phrase, averaged over the three templates."""

    def __init__(self, model, encode_text, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), size=384):
        self.model, self.encode_text, self.size = _frozen(model), encode_text, size
        _checkpoint_vision(self.model, self.model.vision_model)
        self.mean, self.std = mean, std
        self._texts = {}

    @classmethod
    def from_pretrained(cls, name, device, dtype=torch.float32):
        from transformers import AutoProcessor, SiglipModel
        processor = AutoProcessor.from_pretrained(name)
        model = SiglipModel.from_pretrained(name, torch_dtype=dtype).to(device)
        model.text_model.to("cpu")          # about 40% of the weights; embeddings are computed once per phrase

        def encode_text(text):
            tokens = processor.tokenizer([text], padding="max_length", return_tensors="pt")
            with torch.no_grad():
                return model.get_text_features(**tokens)

        ip = processor.image_processor
        return cls(model, encode_text, tuple(ip.image_mean), tuple(ip.image_std),
                   size=processor_size(ip, model.config.vision_config.image_size))

    def _text(self, phrase):
        if phrase not in self._texts:
            self._texts[phrase] = torch.cat([self.encode_text(t) for t in text_templates(phrase)])
        return self._texts[phrase]

    def _logit(self, image, phrase):
        param = next(self.model.parameters())
        img = F.interpolate(image.to(param.device, param.dtype), size=(self.size, self.size), mode="bilinear",
                            align_corners=False, antialias=True)
        mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
        image_emb = F.normalize(self.model.get_image_features(pixel_values=(img - mean) / std), dim=-1)
        text_emb = F.normalize(self._text(phrase).to(img.device, img.dtype), dim=-1)
        logits = (image_emb @ text_emb.T) * self.model.logit_scale.exp() + self.model.logit_bias
        return logits.mean()

    def logits(self, x, region, phrase):
        return self._logit(_chw(x)[None], phrase), self._logit(region, phrase)


class FakeSupport:
    """Test double for ClipSegSupport: fixed masks, or a callable mask(phrase, x)."""

    def __init__(self, masks):
        self.masks = masks

    def __call__(self, x, phrases):
        if callable(self.masks):
            return {p: self.masks(p, _chw(x)) for p in phrases}
        return {p: self.masks[p].to(_chw(x).device, _chw(x).dtype) for p in phrases}


class FakeSigLIP:
    """Test double for SigLIPEvidence: fn(phrase, image) gives the logit."""

    def __init__(self, fn):
        self.fn = fn

    def logits(self, x, region, phrase):
        return (torch.as_tensor(self.fn(phrase, _chw(x)[None]), dtype=_chw(x).dtype),
                torch.as_tensor(self.fn(phrase, region), dtype=_chw(x).dtype))


def processor_size(image_processor, default):
    """Input size from the preprocessor config, which can differ from the vision config."""
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        return size.get("height", size.get("shortest_edge", default))
    return size if isinstance(size, int) else default
