import argparse
import json
import pathlib
import sys
import warnings

warnings.simplefilter("ignore")

import colour
import numpy as np
import tifffile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ab_blind import find_pairs

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
CHROMA_MIN = 8.0
SKIN_HUE = (20.0, 70.0)


def load(path, exclude):
    a = tifffile.imread(path)[..., :3].astype(np.float64) / 65535.0
    keep = np.ones(a.shape[:2], bool)
    h, w = keep.shape
    for x0, y0, x1, y1 in exclude:
        keep[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    lin = np.power(np.clip(a, 0, 1), 2.4)
    xyz = colour.RGB_to_XYZ(lin, colour.RGB_COLOURSPACES["ITU-R BT.709"])
    lab = colour.XYZ_to_Lab(xyz, colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    return lab[keep], lin[keep]


def stats(lab, lin):
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    c = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    return {"L_q": np.quantile(L, QUANTILES), "C_mean": c.mean(), "rgb_mean": lin.mean(0), "hue": hue, "chroma": c, "L": L}


def hue_diff(h1, h2):
    return (h2 - h1 + 180) % 360 - 180


def compare(src, exclude):
    pairs, _ = find_pairs(src)
    rows, curve_d, curve_o = [], [], []
    for scene in sorted(pairs):
        ld, rd = load(pairs[scene]["dehancer"], exclude)
        lo, ro = load(pairs[scene]["ours"], exclude)
        sd, so = stats(ld, rd), stats(lo, ro)
        m = (sd["chroma"] > CHROMA_MIN) & (so["chroma"] > CHROMA_MIN)
        dh = hue_diff(sd["hue"][m], so["hue"][m]) if m.any() else np.array([0.0])
        skin = m & (sd["hue"] > SKIN_HUE[0]) & (sd["hue"] < SKIN_HUE[1]) & (sd["L"] > 25) & (sd["L"] < 85)
        rows.append({
            "scene": scene,
            "L_dehancer": sd["L_q"].round(1).tolist(), "L_ours": so["L_q"].round(1).tolist(),
            "chroma_ratio": round(float(so["C_mean"] / max(sd["C_mean"], 1e-6)), 3),
            "hue_shift_median_deg": round(float(np.median(dh)), 1),
            "skin_px": int(skin.sum()),
            "skin_hue_shift_deg": round(float(np.median(hue_diff(sd["hue"][skin], so["hue"][skin]))), 1) if skin.sum() > 500 else None,
            "wb_ours_minus_dehancer": (np.log2(so["rgb_mean"] / so["rgb_mean"].mean()) - np.log2(sd["rgb_mean"] / sd["rgb_mean"].mean())).round(3).tolist(),
        })
        curve_d.append(np.quantile(sd["L"], np.linspace(0.01, 0.99, 50)))
        curve_o.append(np.quantile(so["L"], np.linspace(0.01, 0.99, 50)))
    cd, co = np.concatenate(curve_d), np.concatenate(curve_o)
    bins = np.arange(0, 101, 10)
    tone = []
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        sel = (cd >= lo_) & (cd < hi_)
        if sel.sum() >= 5:
            tone.append({"L_dehancer": f"{lo_}-{hi_}", "L_ours_minus_dehancer": round(float(np.median(co[sel] - cd[sel])), 1), "n": int(sel.sum())})
    return rows, tone


def main(argv):
    ap = argparse.ArgumentParser(prog="ab_compare.py")
    ap.add_argument("src")
    ap.add_argument("--exclude", type=float, nargs=4, action="append", default=[], metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows, tone = compare(a.src, a.exclude)
    print("sahne | L* q05/q25/q50/q75/q95 Dehancer -> bizimki | kroma oranı | ton kayması° | ten kayması° (px) | WB log2 R,G,B")
    for r in rows:
        skin = f"{r['skin_hue_shift_deg']} ({r['skin_px']})" if r["skin_hue_shift_deg"] is not None else f"- ({r['skin_px']})"
        print(f"{r['scene']} | {r['L_dehancer']} -> {r['L_ours']} | {r['chroma_ratio']} | {r['hue_shift_median_deg']} | {skin} | {r['wb_ours_minus_dehancer']}")
    print("\nTon eğrisi (Dehancer L* aralığına göre, bizimki − Dehancer, medyan ΔL*):")
    for t in tone:
        print(f"  L* {t['L_dehancer']:>7}: {t['L_ours_minus_dehancer']:+.1f}  (n={t['n']})")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"scenes": rows, "tone": tone}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
