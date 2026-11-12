import json
import pathlib
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import colour
import numpy as np
from scipy.optimize import least_squares

from filmemu import spaces, stocks
from filmemu.pipeline import Params, render
from resolve_check import luts

KEYS = ("chroma", "spread", "hl_a", "hl_b", "mid_a", "mid_b", "sh_a", "sh_b")
SHRINK = 0.6
SCALE = {"chroma": 1.5, "spread": 3.0, "hl_a": 0.7, "hl_b": 0.7, "mid_a": 0.7, "mid_b": 0.7, "sh_a": 0.7, "sh_b": 0.7}
SCENES = ("city", "courtyard", "forest", "interior", "sunrise", "sunset")
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
REC709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]
DWG = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]


def lab_stats(display, ok):
    lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(display, REC709, apply_cctf_decoding=False), D65)
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.hypot(A, B)
    p5, p10, p90, p95 = np.quantile(L, [0.05, 0.1, 0.9, 0.95])

    def sel(m):
        m = m & (C < 15.0)
        return (float(A[m].mean()), float(B[m].mean())) if m.sum() >= 0.005 * len(L) else (np.nan, np.nan)

    hl = sel(ok & (L >= max(p90, 55)))
    sh = sel((L <= min(p10, 40)) & (L > 2))
    mid = sel((L > 25) & (L < 75))
    m = (L > 15) & (L < 90) & ok
    return {"chroma": float(C[m].mean()) if m.any() else 0.0, "spread": float(p95 - p5),
            "hl_a": hl[0], "hl_b": hl[1], "mid_a": mid[0], "mid_b": mid[1], "sh_a": sh[0], "sh_b": sh[1]}


def scenes(hdri_dir):
    out = []
    for name in SCENES:
        rgb = np.load(pathlib.Path(hdri_dir) / f"{name}.npy")[::2, ::2].reshape(-1, 3).astype(np.float64)
        y = rgb @ REC709.matrix_RGB_to_XYZ[1]
        rgb = rgb * (0.18 / np.median(np.maximum(y, 1e-6)))
        dwg = colour.RGB_to_RGB(np.maximum(rgb, 0.0), REC709, DWG)
        out.append(np.concatenate([spaces.di_encode(dwg), np.ones((len(dwg), 1))], -1).astype(np.float32))
    return out


def display_linear(pdir, scene_list):
    lt = luts(pdir)
    return [spaces.di_decode(render(s, lt, Params(out_space="dwg_di"))[:, :3]) for s in scene_list]


def look(theta):
    sat, con, tsr, tsb, thr, thb, tmr, tmb = theta
    return {"saturation": sat, "contrast": con, "shadow_tint": [tsr, 1.0, tsb], "highlight_tint": [thr, 1.0, thb],
            "mid_tint": [tmr, 1.0, tmb]}


def model_stats(lin_list, theta):
    fn = stocks.scan_function(look(theta)) if theta is not None else (lambda x: x)
    acc = {k: [] for k in KEYS}
    for lin in lin_list:
        disp = np.clip(colour.RGB_to_RGB(fn(lin), DWG, REC709), 0.0, None)
        enc = np.clip(spaces.g24_encode(disp), 0.0, 1.0)
        s = lab_stats(spaces.g24_decode(enc), (enc < 254.0 / 255.0).all(-1))
        for k in KEYS:
            acc[k].append(s[k])
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def fit(lin_list, target):
    theta0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    lo = [0.88, 0.92, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]
    hi = [1.15, 1.1, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07]
    keys = [k for k in KEYS if k in target]

    def res(t):
        s = model_stats(lin_list, t)
        return np.concatenate([[(s[k] - target[k]) / SCALE[k] for k in keys], 2.0 * (t - theta0)])

    r = least_squares(res, theta0, bounds=(lo, hi), diff_step=1e-3, max_nfev=60)
    return r.x, model_stats(lin_list, r.x)


def main(stats_json, profiles_root, hdri_dir, out_json):
    lomo = json.loads(pathlib.Path(stats_json).read_text(encoding="utf-8"))["films"]
    root = pathlib.Path(profiles_root)
    scene_list = scenes(hdri_dir)
    dirs = {neg: next(p for p in sorted(root.iterdir()) if p.name == neg or p.name.startswith(neg + "-")) for neg in lomo}
    lin = {neg: display_linear(d, scene_list) for neg, d in dirs.items()}
    base = {neg: model_stats(lin[neg], None) for neg in lomo}
    groups = {"bw": [n for n in lomo if stocks.is_bw(n)], "colour": [n for n in lomo if not stocks.is_bw(n)]}
    result = {}
    for gname, members in groups.items():
        usable = lambda n, k: k not in lomo[n].get("unreliable", ())
        lmean = {k: np.mean([lomo[n][k] for n in members if usable(n, k)]) for k in KEYS}
        bmean = {k: np.mean([base[n][k] for n in members if usable(n, k)]) for k in KEYS}
        for n in members:
            if gname == "bw":
                target = {k: lomo[n][k] for k in KEYS if k not in ("chroma", "spread")}
                target["spread"] = bmean["spread"] + SHRINK * (lomo[n]["spread"] - lmean["spread"])
            else:
                target = {k: bmean[k] + SHRINK * (lomo[n][k] - lmean[k]) if usable(n, k) else base[n][k] for k in KEYS}
            theta, got = fit(lin[n], target)
            result[n] = {"look": {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in look(theta).items()},
                         "target": {k: round(float(v), 2) for k, v in target.items()},
                         "base": {k: round(v, 2) for k, v in base[n].items()}, "fitted": {k: round(v, 2) for k, v in got.items()},
                         "photos": lomo[n]["kept"]}
            print(n, json.dumps(result[n]["look"]), flush=True)
    pathlib.Path(out_json).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("usage: scan_fit.py LOMO_STATS.json PROFILES_DIR HDRI_NPY_DIR OUT.json")
    main(*sys.argv[1:])
