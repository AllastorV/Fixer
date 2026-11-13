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


def sample_tetrahedral(lut, coords):
    n = lut.shape[0]
    position = np.clip(np.asarray(coords, np.float64), 0.0, 1.0) * (n - 1)
    corner = np.minimum(np.floor(position).astype(np.int64), n - 2)
    fraction = position - corner
    order = np.argsort(-fraction, axis=-1, kind="stable")
    previous = lut[corner[..., 2], corner[..., 1], corner[..., 0]].astype(np.float64)
    result = previous.copy()
    for i in range(3):
        axis = order[..., i:i + 1]
        corner += np.eye(3, dtype=np.int64)[axis[..., 0]]
        following = lut[corner[..., 2], corner[..., 1], corner[..., 0]].astype(np.float64)
        weight = np.take_along_axis(fraction, axis, axis=-1)
        result += weight * (following - previous)
        previous = following
    return result


def apply(lut, lin, quantize_weights=False, tetrahedral=False):
    coords = di_encode(lin)
    encoded = sample_tetrahedral(lut, coords) if tetrahedral else sample(lut, coords, quantize_weights)
    return di_decode(encoded)


def slot_weights(p):
    t = min(abs(float(p)), 2.0) / 2.0
    s = t * t * (3.0 - 2.0 * t)
    if p >= 0:
        return {"under": 0.0, "normal": 1.0 - s, "over": s}
    return {"under": s, "normal": 1.0 - s, "over": 0.0}
