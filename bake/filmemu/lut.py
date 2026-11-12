import numpy as np

from .spaces import di_decode, di_encode

LUT_SIZE = 65


def bake(fn, n=LUT_SIZE):
    axis = np.linspace(0.0, 1.0, n)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.stack([r, g, b], axis=-1)
    return di_encode(fn(di_decode(coords))).astype(np.float32)


def sample(lut, coords, quantize_weights=False):
    n = lut.shape[0]
    c = np.clip(np.asarray(coords, dtype=np.float64), 0.0, 1.0)
    pos = c * (n - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), n - 2)
    f = pos - i0
    if quantize_weights:
        f = np.round(f * 256.0) / 256.0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    lut = lut.astype(np.float64)

    def at(db, dg, dr):
        return lut[b0 + db, g0 + dg, r0 + dr]

    c00 = at(0, 0, 0) * (1 - fr) + at(0, 0, 1) * fr
    c01 = at(0, 1, 0) * (1 - fr) + at(0, 1, 1) * fr
    c10 = at(1, 0, 0) * (1 - fr) + at(1, 0, 1) * fr
    c11 = at(1, 1, 0) * (1 - fr) + at(1, 1, 1) * fr
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    return c0 * (1 - fb) + c1 * fb


def apply(lut, lin, quantize_weights=False):
    return di_decode(sample(lut, di_encode(lin), quantize_weights))


def slot_weights(p):
    t = min(abs(float(p)), 2.0) / 2.0
    s = t * t * (3.0 - 2.0 * t)
    if p >= 0:
        return {"under": 0.0, "normal": 1.0 - s, "over": s}
    return {"under": s, "normal": 1.0 - s, "over": 0.0}
