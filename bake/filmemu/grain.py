import numpy as np

BASE_SIGMA_LOG10 = 0.04
TONAL_FLOOR = 0.25
TONAL_WIDTH_DECADES = 1.0
REF_HEIGHT = 1080.0
MIN_BLUR_PX = 0.5
MAX_BLUR_PX = 20.0
DWG_Y = np.array([0.27411851, 0.8736319, -0.14775041])

M32 = np.uint64(0xFFFFFFFF)


def hash32(x):
    x = np.asarray(x, dtype=np.uint64) & M32
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7FEB352D)) & M32
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846CA68B)) & M32
    x ^= x >> np.uint64(16)
    return x


def blur_sigma_px(size, frame_height, render_scale=1.0):
    return min(max(MIN_BLUR_PX, size * (frame_height / REF_HEIGHT) * render_scale), MAX_BLUR_PX)


CLOUD_RADIUS = 2.5
LAYER_SCALE = (1.0, 0.9, 1.2)
DENSITY = 1.0
MAX_PER_CELL = 4
CLOUD_MEAN = DENSITY * np.pi / 3.0
CLOUD_STD = np.sqrt(DENSITY * (4.0 / 3.0) * np.pi / 5.0)
CLOUD_CDF = np.cumsum([np.exp(-DENSITY) * DENSITY ** k / np.prod(np.arange(1, k + 1)) for k in range(MAX_PER_CELL)])


def _unit(h):
    return ((h >> np.uint64(8)).astype(np.float64) + 0.5) * 2.0 ** -24


def _next(h):
    return hash32((h + np.uint64(0x9E3779B9)) & M32)


def cloud_base(seed, frame, channel):
    h = hash32(np.uint64(seed) & M32)
    h = hash32(h ^ (np.uint64(np.int64(frame) & 0xFFFFFFFF)))
    return hash32(h ^ np.uint64(channel))


def cell_hash(base, cx, cy):
    mx = (cx.astype(np.uint64) * np.uint64(0x8DA6B343)) & M32
    my = (cy.astype(np.uint64) * np.uint64(0xD8163841)) & M32
    return hash32(base ^ mx ^ my)


def clouds(seed, frame, px, py, channel, radius):
    qx, qy = px / radius, py / radius
    cx0, cy0 = np.floor(qx).astype(np.int64), np.floor(qy).astype(np.int64)
    lx, ly = qx - cx0, qy - cy0
    base = cloud_base(seed, frame, channel)
    out = np.zeros(px.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            h = cell_hash(base, cx0 + dx, cy0 + dy)
            u = _unit(h)
            count = sum((u > c).astype(np.int64) for c in CLOUD_CDF)
            for j in range(MAX_PER_CELL):
                h = _next(h)
                ex = lx - (dx + ((h & np.uint64(2047)).astype(np.float64) + 0.5) / 2048.0)
                ey = ly - (dy + (((h >> np.uint64(11)) & np.uint64(2047)).astype(np.float64) + 0.5) / 2048.0)
                a = ((h >> np.uint64(22)).astype(np.float64) + 0.5) * (2.0 / 1024.0)
                b = np.maximum(1.0 - (ex * ex + ey * ey), 0.0)
                out += np.where(j < count, a * b * b, 0.0)
    return (out - CLOUD_MEAN) / CLOUD_STD


def noise_field(seed, frame, x0, y0, w, h, sigma):
    sigma = float(np.float32(min(max(sigma, MIN_BLUR_PX), MAX_BLUR_PX)))
    ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w]
    px, py = xs + 0.5, ys + 0.5
    return np.stack([clouds(seed, frame, px, py, c, CLOUD_RADIUS * sigma * LAYER_SCALE[c]) for c in range(3)], axis=-1)


TABLE_SIZE = 64
TABLE_LO = -3.0
TABLE_HI = 3.0


def table_axis():
    return np.linspace(TABLE_LO, TABLE_HI, TABLE_SIZE)


def default_table():
    u = table_axis() / TONAL_WIDTH_DECADES
    return np.tile(TONAL_FLOOR + (1.0 - TONAL_FLOOR) * np.exp(-0.5 * u * u), (3, 1))


def lookup(table, scene):
    t = np.asarray(table, np.float32).astype(np.float64)
    u = np.log10(np.maximum(scene, 2.0 ** -20) / 0.18)
    f = (np.clip(u, TABLE_LO, TABLE_HI) - TABLE_LO) * ((TABLE_SIZE - 1) / (TABLE_HI - TABLE_LO))
    i = np.minimum(np.floor(f).astype(int), TABLE_SIZE - 2)
    a = f - i
    out = np.empty_like(u)
    for c in range(3):
        out[..., c] = t[c][i[..., c]] + a[..., c] * (t[c][i[..., c] + 1] - t[c][i[..., c]])
    return out


def offset(noise, amount, scene, table=None, mono=False):
    sigma = lookup(default_table() if table is None else table, scene)
    if mono:
        noise = np.repeat(noise[..., 0:1], 3, axis=-1)
        sigma = np.repeat(lookup(default_table() if table is None else table, np.repeat(np.asarray(scene)[..., 1:2], 3, axis=-1))[..., 1:2], 3, axis=-1)
    return amount * BASE_SIGMA_LOG10 * sigma * noise


def apply(film, noise, amount, scene, table=None, mono=False):
    return film * np.power(10.0, -offset(noise, amount, scene, table, mono))
