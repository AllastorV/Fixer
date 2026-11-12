from dataclasses import dataclass

import numpy as np

from . import grain, spatial
from .lut import apply, sample, slot_weights
from .spaces import di_decode, di_encode, from_working, sanitize, to_working


@dataclass(frozen=True)
class Params:
    in_space: str = "dwg_di"
    out_space: str = "dwg_di"
    exposure_ev: float = 0.0
    push_pull: float = 0.0
    impact: float = 1.0
    premultiplied: bool = False
    grain_amount: float = 0.0
    grain_sigma_px: float = 1.0
    grain_seed: int = 0
    frame: int = 0
    origin_x: int = 0
    origin_y: int = 0
    halation_amount: float = 0.0
    halation_radius_px: float = 6.0
    bloom_amount: float = 0.0
    bloom_radius_px: float = 15.0
    print_log_offset: tuple = (0.0, 0.0, 0.0)


def _film(rgba, luts, params, quantize_weights, region):
    ex1, ey1, ex2, ey2 = region
    rgb = sanitize(rgba[ey1:ey2, ex1:ex2, :3]) if rgba.ndim == 3 else sanitize(rgba[..., :3])
    alpha = (rgba[ey1:ey2, ex1:ex2, 3:4] if rgba.ndim == 3 else rgba[..., 3:4]).astype(np.float64)
    if params.premultiplied:
        rgb = np.where(alpha > 0, rgb / np.where(alpha > 0, alpha, 1.0), rgb)
    working = sanitize(to_working(rgb, params.in_space))
    lin = working * np.exp2(params.exposure_ev)
    if params.halation_amount > 0.0:
        lin = spatial.halation(lin, params.halation_amount, params.halation_radius_px)
    film = np.zeros_like(lin)
    two_stage = luts.get("print") is not None
    noise = None
    if params.grain_amount > 0.0:
        h, w = lin.shape[:2]
        noise = grain.noise_field(params.grain_seed, params.frame, params.origin_x + ex1, params.origin_y + ey1, w, h,
                                  params.grain_sigma_px)
    coords = di_encode(lin)
    for slot, w in slot_weights(params.push_pull).items():
        if w:
            film += w * (sample(luts[slot], coords, quantize_weights) if two_stage else apply(luts[slot], lin, quantize_weights))
    if two_stage:
        lo, hi = luts["print_domain"]
        if noise is not None:
            film = film - grain.offset(noise, params.grain_amount, lin, luts.get("grain"), bool(luts.get("grain_mono")))
        q = (film + np.asarray(params.print_log_offset, np.float64) - lo) / (hi - lo)
        film = di_decode(sample(luts["print"], q, quantize_weights))
    if noise is not None and not two_stage:
        film = grain.apply(film, noise, params.grain_amount, lin, luts.get("grain"), bool(luts.get("grain_mono")))
    if params.bloom_amount > 0.0:
        film = spatial.bloom(film, params.bloom_amount, params.bloom_radius_px)
    return working, film, alpha


def render(rgba, luts, params, quantize_weights=False, window=None):
    rgba = np.asarray(rgba, dtype=np.float32)
    if window is not None:
        x1, y1, x2, y2 = (window[0] - params.origin_x, window[1] - params.origin_y, window[2] - params.origin_x,
                          window[3] - params.origin_y)
    if params.impact == 0.0:
        return rgba.copy() if window is None else rgba[y1:y2, x1:x2].copy()
    if any(luts.get(k) is None for k in ("under", "normal", "over")):
        raise ValueError("profile not ready: missing slot")
    if rgba.ndim != 3:
        working, film, alpha = _film(rgba, luts, params, quantize_weights, (0, 0, 0, 0))
        crop = (slice(None),)
    else:
        h, w = rgba.shape[:2]
        if window is None:
            x1, y1, x2, y2 = 0, 0, w, h
        m = 0
        if params.halation_amount > 0.0:
            m += spatial.margin(params.halation_radius_px)
        if params.bloom_amount > 0.0:
            m += spatial.margin(params.bloom_radius_px)
        e = (max(x1 - m, 0), max(y1 - m, 0), min(x2 + m, w), min(y2 + m, h))
        working, film, alpha = _film(rgba, luts, params, quantize_weights, e)
        crop = (slice(y1 - e[1], y2 - e[1]), slice(x1 - e[0], x2 - e[0]))
    working, film, alpha = working[crop], film[crop], alpha[crop]
    ref = sanitize(from_working(working, params.out_space))
    out = ref + params.impact * (from_working(film, params.out_space) - ref)
    if params.premultiplied:
        out = out * alpha
    return np.concatenate([out, alpha], axis=-1).astype(np.float32)
