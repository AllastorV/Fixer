import json
from dataclasses import dataclass

import numpy as np
import colour

from . import spectral
from .spectral import DATA, WL, SHAPE, FilmSystem, Stock

STOCKS = DATA / "stocks"
READOUT_SIGMA = 12.0
PRINT_AIM_DENSITY = 1.0
PRINTER_CCT = 3200.0
HEAT_GLASS_EDGE = 750.0
HEAT_GLASS_WIDTH = 15.0
DYES = ("cyan", "magenta", "yellow")


@dataclass(frozen=True)
class TabulatedCurve:
    x: np.ndarray
    d: np.ndarray
    x0: float
    gain: float = 1.0

    def density(self, log_h):
        return self.gain * np.interp(log_h, self.x, self.d)


def load(stock_id):
    return json.loads((STOCKS / f"{stock_id}.json").read_text(encoding="utf-8"))


def _spectrum(curve, extrapolate="hold"):
    w = np.asarray(curve["wavelength"], float)
    v = np.asarray(curve["value"], float)
    out = np.interp(WL, w, v)
    if extrapolate == "decline":
        hi = WL > w[-1]
        slope = min(0.0, (v[-1] - v[-11]) / (w[-1] - w[-11]))
        out[hi] = np.clip(v[-1] + slope * (WL[hi] - w[-1]), 0.0, v[-1])
    return out


def _sensitivity(curve):
    w = np.asarray(curve["wavelength"], float)
    lin = np.power(10.0, np.asarray(curve["value"], float))
    return np.where((WL >= w[0]) & (WL <= w[-1]), np.interp(WL, w, lin), 0.0)


def readout_matrix(dyes):
    peaks = WL[np.argmax(dyes, axis=1)]
    bands = np.exp(-0.5 * ((WL[None, :] - peaks[:, None]) / READOUT_SIGMA) ** 2)
    bands /= bands.sum(axis=1, keepdims=True)
    return bands @ dyes.T


def effective_dyes(dyes):
    return np.linalg.inv(readout_matrix(dyes)).T @ dyes


DYE_REFERENCE_STOCK = "kodak-vision3-250d"
DYE_SHIFTS = np.arange(-40, 41, 2)
_DERIVED = {}


def derived_dyes(data):
    key = data["id"]
    if key not in _DERIVED:
        ref = load(DYE_REFERENCE_STOCK)["dye_density"]["curves"]
        curves = data["dye_density"]["curves"]
        target = _spectrum(curves["midscale_neutral"]) - _spectrum(curves["minimum"])
        m = (WL >= 400) & (WL <= 700)
        shifted = {k: [np.interp(WL - s, ref[k]["wavelength"], ref[k]["value"]) for s in DYE_SHIFTS] for k in DYES}
        best = None
        for i, a in enumerate(shifted["cyan"]):
            for j, b in enumerate(shifted["magenta"]):
                for k, c in enumerate(shifted["yellow"]):
                    A = np.stack([a, b, c], axis=1)[m]
                    coef, *_ = np.linalg.lstsq(A, target[m], rcond=None)
                    if np.any(coef <= 0):
                        continue
                    rms = float(np.sqrt(np.mean((A @ coef - target[m]) ** 2)))
                    if best is None or rms < best[0]:
                        best = (rms, (i, j, k), coef)
        rms0, idx, coef = best
        dyes0 = np.stack([shifted[n][s] for n, s in zip(DYES, idx)])
        dyes, rms, params = _refine_dyes(dyes0, coef, target, m)
        _DERIVED[key] = (dyes, {"rms": rms, "rms_shifted_reference": rms0, "shift_nm": [int(DYE_SHIFTS[s]) for s in idx],
                                "dye_params": params})
    return _DERIVED[key]


def _skewed(peak, left, right):
    w = np.where(WL < peak, left, right)
    return np.exp(-0.5 * ((WL - peak) / w) ** 2)


def _refine_dyes(dyes0, coef0, target, m):
    from scipy.optimize import least_squares

    init = []
    for d, a in zip(dyes0, coef0):
        p = float(WL[np.argmax(d)])
        half = d >= d.max() / 2
        lw = max(p - float(WL[half][0]), 15.0) / 1.1774
        rw = max(float(WL[half][-1]) - p, 15.0) / 1.1774
        init += [p, lw, rw, float(a), 0.08]
    init = np.array(init)
    lo = np.array([v for i in range(3) for v in (init[5 * i] - 30, 15, 15, 0.05, 0.0)])
    hi = np.array([v for i in range(3) for v in (init[5 * i] + 30, 80, 120, 3.0, 0.4)])
    tails = np.stack([np.exp(-0.5 * ((WL - 420.0) / 40.0) ** 2), np.exp(-0.5 * ((WL - 430.0) / 40.0) ** 2), np.zeros_like(WL)])

    def model(x):
        shapes = np.stack([_skewed(*x[5 * i:5 * i + 3]) + x[5 * i + 4] * tails[i] for i in range(3)])
        return shapes, x[3::5]

    def resid(x):
        shapes, amounts = model(x)
        return (amounts @ shapes - target)[m]

    sol = least_squares(resid, np.clip(init, lo, hi), bounds=(lo, hi))
    shapes, amounts = model(sol.x)
    shapes = shapes / shapes.max(axis=1, keepdims=True)
    rms = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    params = [{"peak_nm": round(float(sol.x[5 * i]), 1), "left_sigma": round(float(sol.x[5 * i + 1]), 1),
               "right_sigma": round(float(sol.x[5 * i + 2]), 1), "blue_tail": round(float(sol.x[5 * i + 4]), 3)} for i in range(3)]
    return shapes, rms, params


def _stock(data, extrapolate):
    curves = data["dye_density"]["curves"]
    if all(k in curves for k in DYES):
        dyes = np.stack([_spectrum(curves[k], extrapolate) for k in DYES])
    else:
        dyes = derived_dyes(data)[0]
    sens = np.stack([_sensitivity(data["spectral_sensitivity"]["channels"][c]) for c in "rgb"])
    return dyes, sens


def negative(stock_id, contrast=1.0):
    data = load(stock_id)
    dyes, sens = _stock(data, "hold")
    base = _spectrum(data["dye_density"]["curves"]["minimum"])
    curves = []
    for c in "rgb":
        ch = data["sensitometry"]["channels"][c]
        x = np.asarray(ch["x"], float) * np.log10(2.0)
        d = np.asarray(ch["density"], float)
        curves.append(TabulatedCurve(x, d - d[0], 0.0, contrast))
    return Stock(sens, effective_dyes(dyes), base, tuple(curves), data["label"])


def print_stock(stock_id):
    data = load(stock_id)
    dyes, sens = _stock(data, "decline")
    grey_aim = data.get("grey_aim_above_dmin")
    aim = readout_matrix(dyes).sum(axis=1) * PRINT_AIM_DENSITY if grey_aim is None else np.full(3, float(grey_aim))
    curves = []
    for k, c in enumerate("rgb"):
        ch = data["sensitometry"]["channels"][c]
        x, d = np.asarray(ch["x"], float), np.asarray(ch["density"], float)
        curves.append(TabulatedCurve(x, d, float(np.interp(aim[k] + d[0], d, x))))
    return Stock(sens, effective_dyes(dyes), np.zeros_like(WL, dtype=np.float64), tuple(curves), data["label"])


def printer_light():
    planck = colour.sd_blackbody(PRINTER_CCT, SHAPE).values
    glass = 1.0 / (1.0 + np.exp((WL - HEAT_GLASS_EDGE) / HEAT_GLASS_WIDTH))
    return planck / planck.max() * glass


def projector():
    sd = colour.SDS_LIGHT_SOURCES["Kinoton 75P"].copy().align(SHAPE, extrapolator_kwargs={"method": "Constant"})
    return sd.values / sd.values.max()


NEUTRAL_STOPS = np.arange(-8.0, 8.0001, 0.05)
NEUTRAL_MIN_CONDITION = 0.05


def _neutral_output(chain, dens):
    t = chain.neg_transmittance(dens)
    e = np.einsum("nl,kl->nk", t * chain.s.printer_light[None, :], chain.s.print_.sensitivity) / chain.e_grey
    log_e = np.log10(np.maximum(e, 2.0 ** -30)) + np.array([c.x0 for c in chain.s.print_.curves])
    return chain.display_linear(log_e)


def _residual(chain, d):
    out = _neutral_output(chain, d[None, :])[0]
    return np.array([out[0] - out[1], out[2] - out[1]]) / max(out[1], 1e-9)


def _jacobian(chain, d, r):
    jac = np.zeros((2, 2))
    for k, ch in enumerate((0, 2)):
        dd = d.copy()
        dd[ch] += 1e-4
        jac[:, k] = (_residual(chain, dd) - r) / 1e-4
    return jac


def neutralize(fs):
    chain = spectral.Chain(fs)
    x = NEUTRAL_STOPS * np.log10(2.0)
    dens = np.stack([c.density(x) for c in fs.negative.curves], axis=-1)
    start = int(np.argmin(np.abs(x)))
    det0 = abs(np.linalg.det(_jacobian(chain, dens[start], _residual(chain, dens[start]))))
    fixed = dens.copy()
    for order in (range(start, len(x)), range(start, -1, -1)):
        offset = np.zeros(2)
        solving = True
        for i in order:
            d = dens[i].copy()
            d[[0, 2]] += offset
            for _ in range(8 if solving else 0):
                r = _residual(chain, d)
                if np.abs(r).max() < 1e-7:
                    break
                jac = _jacobian(chain, d, r)
                if abs(np.linalg.det(jac)) < NEUTRAL_MIN_CONDITION * det0:
                    solving = False
                    d[[0, 2]] = dens[i, [0, 2]] + offset
                    break
                d[[0, 2]] -= np.clip(np.linalg.solve(jac, r), -0.2, 0.2)
            offset = d[[0, 2]] - dens[i, [0, 2]]
            fixed[i] = d
    fixed = np.maximum.accumulate(fixed, axis=0)
    curves = tuple(TabulatedCurve(x, fixed[:, k], 0.0) for k in range(3))
    return FilmSystem(Stock(fs.negative.sensitivity, fs.negative.dyes, fs.negative.base, curves, fs.negative.label),
                      fs.print_, fs.printer_light, fs.projector)


GRAIN_DELTA = 1e-3
GRAIN_REFERENCE_SIGMA = 0.0062


GRAIN_FALLBACK_STOCK = "kodak-vision3-250d"


def _sigma(channel, u):
    return np.power(10.0, np.interp(u, channel["x"], channel["log10_sigma"]))


def _granularity(gran, ch, u):
    if ch in gran["channels"]:
        return _sigma(gran["channels"][ch], u)
    ref = load(GRAIN_FALLBACK_STOCK)["granularity"]["channels"]
    return _sigma(gran["channels"]["g"], u) * _sigma(ref[ch], u) / _sigma(ref["g"], u)


def grain_table(fs, negative_id):
    from . import grain

    if is_bw(negative_id):
        from . import bw

        return bw.grain_table(fs, negative_id)
    if is_reversal(negative_id):
        from . import reversal

        return reversal.grain_table(fs, negative_id)

    data = load(negative_id)
    gran = data.get("granularity") or load(GRAIN_FALLBACK_STOCK)["granularity"]
    level = 1.0 if "granularity" in data or "rms_granularity" not in data else data["rms_granularity"] / 1000.0 / GRAIN_REFERENCE_SIGMA
    chain = spectral.Chain(fs)
    u = grain.table_axis()
    dens = np.stack([c.density(u) for c in fs.negative.curves], axis=-1)
    base = np.log10(np.maximum(_neutral_output(chain, dens), 1e-12))
    out = np.zeros((3, len(u)))
    gammas = np.zeros((3, len(u)))
    for c, ch in enumerate("rgb"):
        d = dens.copy()
        d[:, c] += GRAIN_DELTA
        gamma = np.abs(np.log10(np.maximum(_neutral_output(chain, d), 1e-12))[:, c] - base[:, c]) / GRAIN_DELTA
        sigma = _granularity(gran, ch, u)
        out[c] = sigma
        gammas[c] = gamma
    return out * level / (np.interp(0.0, u, gammas[1]) * GRAIN_REFERENCE_SIGMA)


LIGHTS = ("daylight", "tungsten")
TUNGSTEN_CCT = 3200.0


def light(name):
    if name == "daylight":
        sd = colour.SDS_ILLUMINANTS["D55"].copy().align(SHAPE, extrapolator_kwargs={"method": "Constant"}).values
    elif name == "tungsten":
        sd = colour.sd_blackbody(TUNGSTEN_CCT, SHAPE).values
    else:
        raise ValueError(name)
    return sd / sd.max()


def balance(negative_id):
    return load(negative_id).get("balance")


def opposite(name):
    return LIGHTS[1 - LIGHTS.index(name)]


GREY_BALANCE_STOPS = np.arange(-12.0, 12.0001, 0.05)


def grey_balance(chain):
    lin = (0.18 * 2.0 ** GREY_BALANCE_STOPS)[:, None].repeat(3, 1)
    out = np.maximum.accumulate(chain.display_linear(chain.print_log_exposure(lin)), axis=0)
    target = out[:, 1]

    def apply(rgb):
        rgb = np.asarray(rgb, np.float64).copy()
        for c in (0, 2):
            x, i = np.unique(out[:, c], return_index=True)
            rgb[..., c] = np.interp(rgb[..., c], x, target[i])
        return rgb

    return apply


PRINT_BALANCE_RANGE = np.arange(-3.0, 3.0001, 0.01)


def print_balance(chain):
    x0 = np.array([c.x0 for c in chain.s.print_.curves])
    out = np.maximum.accumulate(chain.display_linear(x0[None, :] + PRINT_BALANCE_RANGE[::-1, None]), axis=0)
    target = out[:, 1]

    def apply(rgb):
        rgb = np.asarray(rgb, np.float64).copy()
        for c in (0, 2):
            x, i = np.unique(out[:, c], return_index=True)
            rgb[..., c] = np.interp(rgb[..., c], x, target[i])
        return rgb

    return apply


def is_paper(print_id):
    return load(print_id).get("role") == "paper"


def viewing(print_id):
    v = load(print_id).get("viewing")
    if v is None:
        return projector()
    sd = colour.SDS_ILLUMINANTS[v].copy().align(SHAPE, extrapolator_kwargs={"method": "Constant"}).values
    return sd / sd.max()


def is_bw(negative_id):
    return load(negative_id).get("role") == "negative_bw"


def is_reversal(stock_id):
    return load(stock_id).get("role") == "reversal"


def make_chain(fs, scene_light=None, design_light=None):
    from . import bw, reversal

    if isinstance(fs, bw.BWSystem):
        return bw.BWChain(fs)
    if isinstance(fs, reversal.ReversalSystem):
        return reversal.ReversalChain(fs, scene_light, design_light)
    return spectral.Chain(fs, scene_light, design_light)


_FAMILY_FIX = {}


def _raw(negative_id, print_id, contrast=1.0):
    return FilmSystem(negative(negative_id, contrast), print_stock(print_id), printer_light(), viewing(print_id))


def family(print_id):
    out = []
    for p in sorted(STOCKS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("role") == "negative" and (d.get("print") or ("kodak-2383" if d["id"].startswith("kodak-vision3") else "kodak-endura")) == print_id:
            out.append(d["id"])
    return out


def family_fix(print_id):
    if print_id not in _FAMILY_FIX:
        x = NEUTRAL_STOPS * np.log10(2.0)
        fixes = []
        for neg in family(print_id):
            raw = _raw(neg, print_id)
            fixed = neutralize(raw)
            fixes.append(np.stack([fc.density(x) - rc.density(x) for fc, rc in zip(fixed.negative.curves, raw.negative.curves)], axis=-1))
        _FAMILY_FIX[print_id] = (x, np.mean(fixes, axis=0))
    return _FAMILY_FIX[print_id]


_FAMILY_BAL = {}


def family_balance(print_id):
    if print_id not in _FAMILY_BAL:
        lin = (0.18 * 2.0 ** GREY_BALANCE_STOPS)[:, None].repeat(3, 1)
        logs = []
        for neg in family(print_id):
            ch = spectral.Chain(characteristic(_raw(neg, print_id), print_id))
            logs.append(np.log10(np.maximum(np.maximum.accumulate(ch.display_linear(ch.print_log_exposure(lin)), axis=0), 1e-9)))
        out = np.power(10.0, np.mean(logs, axis=0))
        _FAMILY_BAL[print_id] = out

    out = _FAMILY_BAL[print_id]

    def apply(rgb):
        rgb = np.asarray(rgb, np.float64).copy()
        for c in (0, 2):
            x, i = np.unique(out[:, c], return_index=True)
            rgb[..., c] = np.interp(rgb[..., c], x, out[i, 1])
        return rgb

    return apply


def characteristic(fs, print_id):
    x, fix = family_fix(print_id)
    curves = tuple(TabulatedCurve(x, c.density(x) + fix[:, k], 0.0) for k, c in enumerate(fs.negative.curves))
    return FilmSystem(Stock(fs.negative.sensitivity, fs.negative.dyes, fs.negative.base, curves, fs.negative.label),
                      fs.print_, fs.printer_light, fs.projector)


def system(negative_id, print_id, contrast=1.0, balanced=True):
    if is_bw(negative_id):
        from . import bw

        return bw.system(negative_id, print_id, contrast)
    if is_reversal(negative_id):
        from . import reversal

        return reversal.system(negative_id, contrast)
    fs = _raw(negative_id, print_id, contrast)
    return neutralize(fs) if balanced else characteristic(fs, print_id)


SCAN_WHITE_STOPS = np.log2(1.0 / 0.18)
SCAN_SHADOW_STOPS = 4.0


def scan_look(negative_id):
    look = load(negative_id).get("scan")
    return scan_function(look) if look else None


def scan_function(look):
    sat, con = look["saturation"], look["contrast"]
    ts, th = np.asarray(look["shadow_tint"]), np.asarray(look["highlight_tint"])
    tm = np.asarray(look.get("mid_tint", [1.0, 1.0, 1.0]))
    y_w = np.array([0.27411851, 0.8736319, -0.14775041])

    def apply(rgb):
        rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
        y = np.maximum(rgb @ y_w, 1e-9)
        lg = np.log2(y / 0.18)
        top = np.clip(lg, 0.0, SCAN_WHITE_STOPS)
        curved = np.where(lg < 0.0, lg * con, lg + (con - 1.0) * top * (1.0 - top / SCAN_WHITE_STOPS))
        rgb = rgb * np.power(2.0, curved - lg)[..., None]
        y = rgb @ y_w
        rgb = y[..., None] + sat * (rgb - y[..., None])
        ws = np.clip(-lg / SCAN_SHADOW_STOPS, 0.0, 1.0)[..., None]
        wh = np.clip(lg / SCAN_WHITE_STOPS, 0.0, 1.0)[..., None]
        return np.maximum(rgb * tm * (1.0 + ws * (ts - 1.0) + wh * (th - 1.0)), 0.0)

    return apply
