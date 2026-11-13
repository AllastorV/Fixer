import colour
import numpy as np

from . import lut, spaces
from .stocks import SCAN_SHADOW_STOPS, SCAN_WHITE_STOPS

PCTS = (2, 10, 25, 50, 75, 90, 98)
SLOPE_MIN, SLOPE_MAX = 0.5, 2.0
SAT_MIN, SAT_MAX = 0.85, 1.25
TINT_MIN, TINT_MAX = 0.93, 1.07
MIN_SHARE = 0.005
NEUTRAL_C = 15.0
KNOT_GAP = 1e-3
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
REC709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]
Y_W = REC709.matrix_RGB_to_XYZ[1]
DWG_TO_709 = spaces._matrix(spaces.WORKING, "ITU-R BT.709")
REC709_TO_DWG = spaces._matrix("ITU-R BT.709", spaces.WORKING)
CUBE_HEADER = "FIXER reference match only; film, grain, halation, bloom not included"
PCTS3 = (0.5, 2, 10, 25, 50, 75, 90, 98, 99.5)
SLOPE3_MIN, SLOPE3_MAX = 0.2, 5.0
SAT3_MIN, SAT3_MAX = 0.05, 2.5
# Ratios to green, normalized to unit luminance by gain_vec; not exposure gains.
GAIN_MIN, GAIN_MAX = 1.0 / 64.0, 64.0
LIFT_MAX = 4.0
LIFT_EPS = 1e-6
MERGE = 0.5
NEWTON_STEPS = 8
MONO_STEPS = 20
MONO_GRID = 401


def finite(rgb):
    return np.nan_to_num(np.asarray(rgb, np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def downsample(rgb, width=256):
    rgb = finite(rgb)
    h, w = rgb.shape[:2]
    if w <= width:
        return rgb
    height = max(1, int(round(h * width / w)))

    def weights(n_src, n_dst):
        edges = np.arange(n_dst + 1) * (n_src / n_dst)
        lo, hi = edges[:-1, None], edges[1:, None]
        src = np.arange(n_src)[None, :]
        return np.clip(np.minimum(hi, src + 1) - np.maximum(lo, src), 0.0, None) / (n_src / n_dst)

    return np.einsum("xw,ywc->yxc", weights(w, width), np.einsum("yh,hwc->ywc", weights(h, height), rgb, optimize=True), optimize=True)


def decode(rgb, encoding):
    rgb = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0)
    return spaces.g24_decode(rgb) if encoding == "rec709" else colour.models.eotf_sRGB(rgb)


def lab_709(lin):
    return colour.XYZ_to_Lab(colour.RGB_to_XYZ(lin, REC709, apply_cctf_decoding=False), D65)


def L_of_Y(y):
    return colour.colorimetry.lightness_CIE1976(np.asarray(y, np.float64) * 100.0)


def Y_of_L(L):
    return colour.colorimetry.luminance_CIE1976(np.asarray(L, np.float64)) / 100.0


def stats(rgb, encoding="rec709"):
    enc = np.clip(finite(rgb)[..., :3].reshape(-1, 3), 0.0, 1.0)
    ok = ((enc < 254.0 / 255.0) & (enc > 1.0 / 255.0)).all(-1)
    lab = lab_709(decode(enc, encoding))
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.hypot(A, B)
    n = len(L)
    p = np.percentile(L, PCTS3)
    zones = {"sh": L <= p[3], "mid": (L > p[3]) & (L < p[5]), "hl": (L >= p[5]) & (L > p[3])}

    def zone(m):
        m = m & ok
        return [float(L[m].mean()), float(A[m].mean()), float(B[m].mean())] if m.sum() >= MIN_SHARE * n else None

    mid = zones["mid"] & ok
    return {"L": [float(v) for v in p],
            "sat": float(np.median(C[mid] / (L[mid] + 16.0))) if mid.sum() >= MIN_SHARE * n else None,
            "tint": {k: zone(m) for k, m in zones.items()}}


def fit_curve3(src, dst):
    x = np.clip(np.asarray(src, np.float64), 0.0, 100.0)
    y = np.clip(np.asarray(dst, np.float64), 0.0, 100.0)
    k, i = [], 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[j] - x[i] < MERGE:
            j += 1
        k.append([float(x[i:j].mean()), float(y[i:j].mean())])
        i = j
    if len(k) == 1:
        k.append([k[0][0] + MERGE, k[0][1] + MERGE])
    out = [k[0]]
    for xi, yi in k[1:]:
        dx, py = xi - out[-1][0], out[-1][1]
        out.append([xi, float(min(max(yi, py + SLOPE3_MIN * dx), py + SLOPE3_MAX * dx))])
    return out


def curve_eval3(curve, L):
    k = np.asarray(curve, np.float64)
    s0 = (k[1, 1] - k[0, 1]) / (k[1, 0] - k[0, 0])
    L = np.asarray(L, np.float64)
    lo = np.maximum(k[0, 1] + (L - k[0, 0]) * s0, 0.0)
    hi = k[-1, 1] + (L - k[-1, 0]) * SLOPE3_MIN
    return np.where(L < k[0, 0], lo, np.where(L > k[-1, 0], hi, np.interp(L, k[:, 0], k[:, 1])))


def gain_vec(r, b):
    g = np.array([r, 1.0, b])
    return g / (g @ Y_W)


def solve_gains(L, da, db):
    y = float(Y_of_L(L))
    grey = np.array([y, y, y])
    target = lab_709(grey)[1:] + [da, db]
    v = np.zeros(2)
    h = 1e-4
    for _ in range(NEWTON_STEPS):
        f = lab_709(grey * gain_vec(*np.exp(v)))[1:] - target
        jr = (lab_709(grey * gain_vec(*np.exp(v + [h, 0.0])))[1:] - target - f) / h
        jb = (lab_709(grey * gain_vec(*np.exp(v + [0.0, h])))[1:] - target - f) / h
        det = jr[0] * jb[1] - jb[0] * jr[1]
        if not abs(det) > 1e-12:
            break
        step = np.array([(f[0] * jb[1] - jb[0] * f[1]) / det, (jr[0] * f[1] - f[0] * jr[1]) / det])
        v = np.clip(v - step, np.log(GAIN_MIN), np.log(GAIN_MAX))
    return [float(np.exp(v[0])), float(np.exp(v[1]))]


def _knots(v):
    v = np.clip(np.asarray(v, np.float64), KNOT_GAP, 100.0 - KNOT_GAP * (len(v) + 1))
    for i in range(1, len(v)):
        v[i] = max(v[i], v[i - 1] + KNOT_GAP)
    return v


def fit_curve(src, dst):
    x, y = _knots(src), _knots(dst)
    out, prev_x, prev_y = [], 0.0, 0.0
    for xi, yi in zip(x, y):
        dx = xi - prev_x
        lo = max(prev_y + SLOPE_MIN * dx, 100.0 - SLOPE_MAX * (100.0 - xi))
        hi = min(prev_y + SLOPE_MAX * dx, 100.0 - SLOPE_MIN * (100.0 - xi))
        prev_x, prev_y = xi, float(np.clip(yi, lo, hi))
        out.append([float(xi), prev_y])
    return out


def tint_gains(L, da, db):
    y = float(Y_of_L(L))
    grey = np.array([y, y, y])
    base = lab_709(grey)[1:]
    h = 1e-3
    jr = (lab_709(grey * [1.0 + h, 1.0, 1.0])[1:] - base) / h
    jb = (lab_709(grey * [1.0, 1.0, 1.0 + h])[1:] - base) / h
    d = np.linalg.solve(np.stack([jr, jb], -1), [da, db])
    return [float(np.clip(1.0 + d[0], TINT_MIN, TINT_MAX)), float(np.clip(1.0 + d[1], TINT_MIN, TINT_MAX))]


def zone_L(frame_stats):
    p, t = frame_stats["L"], frame_stats["tint"]
    sh = t["sh"][0] if t["sh"] else min(p[1], 40.0)
    hl = t["hl"][0] if t["hl"] else max(p[5], 55.0)
    mid = t["mid"][0] if t["mid"] else p[3]
    return [float(sh), float(min(max(mid, sh + 1.0), hl - 1.0)), float(hl)]


def zone_L3(frame_stats):
    p, t = frame_stats["L"], frame_stats["tint"]
    sh = t["sh"][0] if t["sh"] else p[2]
    hl = t["hl"][0] if t["hl"] else p[6]
    mid = t["mid"][0] if t["mid"] else p[4]
    if hl - sh < 2.0:
        sh = min(max(sh - 1.0, 0.0), 98.0)
        hl = sh + 2.0
    return [float(sh), float(min(max(mid, sh + 1.0), hl - 1.0)), float(hl)]


def fit(frame_stats, ref_stats, source="film"):
    curve = fit_curve3(frame_stats["L"], ref_stats["L"])
    fs, rs = frame_stats["sat"], ref_stats["sat"]
    sat = float(np.clip(rs / fs, SAT3_MIN, SAT3_MAX)) if fs and rs is not None and fs > 1e-6 else 1.0
    tint = {}
    for k in ("sh", "mid", "hl"):
        f, r = frame_stats["tint"][k], ref_stats["tint"][k]
        if not (f and r):
            tint[k] = [1.0, 1.0]
            continue
        L2 = float(curve_eval3(curve, f[0]))
        keep = sat * (L2 + 16.0) / (f[0] + 16.0)
        tint[k] = solve_gains(L2, r[1] - keep * f[1], r[2] - keep * f[2])
    st = {"version": 3, "mode": "simple", "source": source, "curve": curve, "sat": sat, "tint": tint,
          "zone_L": zone_L3(frame_stats), "ref": ref_stats, "frame": frame_stats}
    st["tint"] = monotone_tint(st)
    return st


def grey_monotone(state):
    y = Y_of_L(np.linspace(0.0, 100.0, MONO_GRID))
    grey = (y[:, None] * np.ones(3)) @ REC709_TO_DWG.T
    out = transform3(state)(grey)
    return bool(np.all(np.diff(out, axis=0) >= 0.0) and np.all(np.diff(out @ DWG_TO_709.T, axis=0) >= -1e-12))


def monotone_tint(state):
    base = {k: list(v) for k, v in state["tint"].items()}
    m = np.log(base["mid"])
    for i in range(MONO_STEPS + 1):
        t = 1.0 - i / MONO_STEPS
        tint = {k: (base[k] if k == "mid" else [float(v) for v in np.exp(m + t * (np.log(base[k]) - m))]) for k in base}
        if grey_monotone(dict(state, tint=tint)):
            return tint
    return tint


def curve_eval(curve, L):
    k = np.array([[0.0, 0.0], *curve, [100.0, 100.0]])
    slope = (k[-1, 1] - k[-2, 1]) / (k[-1, 0] - k[-2, 0])
    L = np.asarray(L, np.float64)
    return np.where(L > 100.0, 100.0 + (L - 100.0) * slope, np.interp(L, k[:, 0], k[:, 1]))


def transform3(state):
    curve, sat, zl = state["curve"], state["sat"], state["zone_L"]
    if curve == [[0.0, 0.0], [100.0, 100.0]] and sat == 1.0 and all(v == [1.0, 1.0] for v in state["tint"].values()):
        return lambda c: np.asarray(c, np.float64)

    g = {k: gain_vec(*v) for k, v in state["tint"].items()}

    def apply(c):
        rgb = np.asarray(c, np.float64) @ DWG_TO_709.T
        if state["source"] == "only":
            rgb = np.clip(rgb, 0.0, 1.0)
        y = rgb @ Y_W
        ye = np.maximum(y, LIFT_EPS)
        L0 = L_of_Y(ye)
        y2 = Y_of_L(curve_eval3(curve, L0))
        k = np.minimum(y2 / ye, LIFT_MAX)
        rgb = y2[..., None] + k[..., None] * (rgb - ye[..., None])
        y2 = y2 + k * (y - ye)
        rgb = y2[..., None] + sat * (rgb - y2[..., None])
        ws = np.clip((zl[1] - L0) / (zl[1] - zl[0]), 0.0, 1.0)[..., None]
        wh = np.clip((L0 - zl[1]) / (zl[2] - zl[1]), 0.0, 1.0)[..., None]
        rgb = rgb * ((1.0 - ws - wh) * g["mid"] + ws * g["sh"] + wh * g["hl"])
        return np.maximum(rgb @ REC709_TO_DWG.T, 0.0)

    return apply



def transform(state):
    if state.get("version") == 3:
        return transform3(state)
    curve, sat = state["curve"], state["sat"]
    gains = {k: np.array([v[0], 1.0, v[1]]) for k, v in state["tint"].items()}
    zl = state["zone_L"] if state.get("version") == 2 else None

    def apply(c):
        rgb = np.asarray(c, np.float64) @ DWG_TO_709.T
        y = rgb @ Y_W
        L0 = L_of_Y(y)
        live = y > 1e-6
        ys = np.where(live, y, 1.0)
        y2 = np.where(live, Y_of_L(curve_eval(curve, L_of_Y(ys))), ys)
        rgb = rgb * np.where(live, y2 / ys, 1.0)[..., None]
        y = rgb @ Y_W
        rgb = y[..., None] + sat * (rgb - y[..., None])
        if zl:
            ws = np.clip((zl[1] - L0) / (zl[1] - zl[0]), 0.0, 1.0)[..., None]
            wh = np.clip((L0 - zl[1]) / (zl[2] - zl[1]), 0.0, 1.0)[..., None]
        else:
            lg = np.log2(np.maximum(y, 1e-9) / 0.18)
            ws = np.clip(-lg / SCAN_SHADOW_STOPS, 0.0, 1.0)[..., None]
            wh = np.clip(lg / SCAN_WHITE_STOPS, 0.0, 1.0)[..., None]
        rgb = rgb * ((1.0 - ws - wh) * gains["mid"] + ws * gains["sh"] + wh * gains["hl"])
        return np.maximum(rgb @ REC709_TO_DWG.T, 0.0)

    return apply


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def skin(c):
    lab = lab_709(np.asarray(c, np.float64) @ DWG_TO_709.T)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(A, B)
    h = np.degrees(np.arctan2(B, A)) % 360.0
    wh = _smoothstep(10.0, 20.0, h) * (1.0 - _smoothstep(75.0, 85.0, h))
    wc = _smoothstep(2.0, 8.0, C) * (1.0 - _smoothstep(45.0, 51.0, C))
    return np.where((L >= 15.0) & (L <= 95.0), wh * wc, 0.0)


def blended(state, strength, protect):
    m = transform(state)

    def apply(c):
        c = np.asarray(c, np.float64)
        w = strength * (1.0 - protect * skin(c))
        return c + w[..., None] * (m(c) - c)

    return apply


def refine(state, encoded, reference_encoded=None):
    import copy
    state = copy.deepcopy(state)
    frame, ref = state["frame"], state["ref"]
    close = max(abs(a - b) for a, b in zip(frame["L"], ref["L"])) < 0.5
    close = close and (frame["sat"] is None or ref["sat"] is None or abs(frame["sat"] - ref["sat"]) < 0.005)
    for zone in ("sh", "mid", "hl"):
        f, r = frame["tint"][zone], ref["tint"][zone]
        if f and r:
            close = close and np.linalg.norm(np.array(f[1:]) - r[1:]) < 0.5
    # Flat patches can cross percentile zone boundaries after 8-bit quantization.
    # A spatially matching reference still represents an identity grade.
    if reference_encoded is not None and encoded.shape[:2] == reference_encoded.shape[:2]:
        a = lab_709(decode(finite(encoded)[..., :3], "rec709"))
        b = lab_709(decode(finite(reference_encoded)[..., :3], "srgb"))
        distance = np.linalg.norm(a - b, axis=-1)
        close = close or (float(distance.mean()) < 0.5 and float(np.mean(distance < 1.0)) >= 0.95)
    if close:
        state.update(curve=[[0.0, 0.0], [100.0, 100.0]], sat=1.0,
                     tint={k: [1.0, 1.0] for k in ("sh", "mid", "hl")})
        return state
    samples = finite(encoded)[..., :3].reshape(-1, 3)[::16]
    lin = spaces.g24_decode(samples) @ REC709_TO_DWG.T
    ref = state["ref"]

    def cost(candidate):
        out = transform(candidate)(lin) @ DWG_TO_709.T
        measured = stats(spaces.g24_encode(out))
        error = np.mean(np.abs(np.array(measured["L"]) - ref["L"]))
        if measured["sat"] is not None and ref["sat"] is not None:
            error += 66.0 * abs(measured["sat"] - ref["sat"])
        zones = [k for k in ("sh", "mid", "hl") if ref["tint"][k]]
        if zones:
            error += sum(np.linalg.norm(np.array(measured["tint"][k][1:]) - ref["tint"][k][1:])
                         if measured["tint"][k] else 100.0 for k in zones) / len(zones)
        return float(error)

    # A cast contributes to the chroma statistic. Fit saturation and tint jointly,
    # including a low-saturation start so a strong cast cannot trap the solve.
    winner, winner_error = None, float("inf")
    for scale in (1.0, 0.15):
        candidate = copy.deepcopy(state)
        candidate["sat"] = float(np.clip(state["sat"] * scale, SAT3_MIN, SAT3_MAX))
        for zone in ("sh", "mid", "hl"):
            f, r = frame["tint"][zone], ref["tint"][zone]
            if f and r:
                L2 = float(curve_eval3(candidate["curve"], f[0]))
                keep = candidate["sat"] * (L2 + 16.0) / (f[0] + 16.0)
                candidate["tint"][zone] = solve_gains(L2, r[1] - keep * f[1], r[2] - keep * f[2])
        candidate["tint"] = monotone_tint(candidate)
        best = cost(candidate)
        for step in (0.5, 0.25, 0.12, 0.06, 0.03):
            for repeat in range(3):
                # Saturation, six zone gains, then two whole-image gain moves.
                for index in range(9):
                    base = copy.deepcopy(candidate)
                    for sign in (-1, 1):
                        trial = copy.deepcopy(base)
                        if index == 0:
                            trial["sat"] = float(np.clip(base["sat"] * np.exp(sign * step), SAT3_MIN, SAT3_MAX))
                        else:
                            channel = (index - 1) % 2
                            zones = ("sh", "mid", "hl") if index > 6 else (("sh", "mid", "hl")[(index - 1) // 2],)
                            for zone in zones:
                                trial["tint"][zone][channel] = float(np.clip(base["tint"][zone][channel] * np.exp(sign * step), GAIN_MIN, GAIN_MAX))
                        # Refinement must retain the monotonicity established by fit().
                        if not grey_monotone(trial):
                            continue
                        error = cost(trial)
                        if error < best - 0.01:
                            candidate, best = trial, error
        if best < winner_error:
            winner, winner_error = candidate, best
    return winner



def bake_lut(state, strength, protect, size=33):
    return lut.bake(blended(state, strength, protect), size)


def write_cube(path, state, strength, protect, size=33):
    axis = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    dwg = spaces.g24_decode(np.stack([r, g, b], -1)) @ REC709_TO_DWG.T
    out = spaces.g24_encode(blended(state, strength, protect)(dwg) @ DWG_TO_709.T).reshape(-1, 3)
    lines = ["TITLE \"FIXER reference match\"", f"# {CUBE_HEADER}", f"LUT_3D_SIZE {size}",
             "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in out]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
