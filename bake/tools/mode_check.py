import json
import pathlib
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import colour
import numpy as np
import pymupdf

from filmemu import spaces
from filmemu.pipeline import Params, render
from resolve_check import luts

REC, DWG = colour.RGB_COLOURSPACES["ITU-R BT.709"], colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
SH, SW, TW, WH = 256, 512, 256, 48
STOPS = np.arange(-6, 7)
MODES = (("datasheet", False, False), ("lab scan", False, True), ("grey balance", True, False))
RATIO = (0.75, 1.0 / 0.75)
GRAIN = dict(grain_amount=0.5, grain_sigma_px=1.0, grain_seed=7)
HAL = dict(halation_amount=0.3, halation_radius_px=6.0)


def scene(hdri_dir):
    tiles = []
    for name in ("courtyard", "sunset"):
        a = np.load(pathlib.Path(hdri_dir) / f"{name}.npy")[:SH, 200:200 + SW].astype(np.float64)
        tiles.append(a * (0.18 / np.median(a.reshape(-1, 3) @ REC.matrix_RGB_to_XYZ[1])))
    test = np.zeros((SH, TW, 3))
    test[:SH // 2] = 0.18
    yy, xx = np.mgrid[SH // 2:SH, 0:TW]
    test[SH // 2:][(yy - 3 * SH // 4) ** 2 + (xx - TW // 2) ** 2 <= 36] = 16.0
    top = np.concatenate(tiles + [test], 1)
    wedge = np.repeat(np.repeat((0.18 * 2.0 ** STOPS)[None, :, None], top.shape[1] // len(STOPS) + 1, 1)[:, :top.shape[1]], WH, 0)
    img = np.concatenate([top, np.repeat(wedge, 3, -1) if wedge.shape[-1] == 1 else wedge], 0)
    dwg = colour.RGB_to_RGB(np.maximum(img, 0.0), REC, DWG)
    return np.concatenate([spaces.di_encode(dwg), np.ones(img.shape[:2] + (1,))], -1).astype(np.float32)


def profile_luts(pdir, bal, scan):
    lt = luts(pdir, bal, scan)
    meta = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
    if "grain" in meta:
        lt["grain"] = np.fromfile(pdir / "grain.bin", np.float32).reshape(3, -1)
    lt["grain_mono"] = meta.get("grain_mono") == "true"
    return lt, meta


def lab(display):
    lin = np.clip(display, 0.0, 1.0) ** 2.4
    return colour.XYZ_to_Lab(colour.RGB_to_XYZ(lin, REC, apply_cctf_decoding=False), D65)


def measure(img, lt):
    shown = render(img, lt, Params(out_space="rec709_g24", **GRAIN, **HAL))[..., :3]
    hal = render(img, lt, Params(out_space="rec709_g24", **HAL))[..., :3]
    none = render(img, lt, Params(out_space="rec709_g24"))[..., :3]
    x0 = 2 * SW
    grain = float(np.std((shown - hal)[16:SH // 2 - 16, x0 + 16:x0 + TW - 16]))
    yy, xx = np.mgrid[SH // 2:SH, 0:TW]
    r = np.hypot(yy - 3 * SH // 4, xx - TW // 2)
    ring = (r > 9) & (r < 40)
    d = (np.clip(hal, 0, 1) ** 2.4 - np.clip(none, 0, 1) ** 2.4)[SH // 2:SH, x0:x0 + TW]
    halation = float(d[ring].sum(0)[0])
    red_share = float(d[ring].sum(0)[0] / max(d[ring].sum(), 1e-9))
    step = img.shape[1] // len(STOPS) + 1
    cols = (np.arange(len(STOPS)) * step + step // 2).clip(0, img.shape[1] - 1)
    wl = lab(none[SH + WH // 2, cols])
    return shown, none, {"grain_rms": grain, "halation_red": halation, "halation_red_share": red_share,
                         "wedge_L": wl[:, 0].round(2).tolist(), "wedge_C": np.hypot(wl[:, 1], wl[:, 2]).round(2).tolist()}


def judge(m, meta, is_bw):
    base = m["datasheet"]
    fails = []
    for name in ("lab scan", "grey balance"):
        for k in ("grain_rms", "halation_red"):
            if base[k] > 1e-6:
                q = m[name][k] / base[k]
                m[name][k + "_ratio"] = round(q, 3)
                if not RATIO[0] <= q <= RATIO[1]:
                    fails.append(f"{name}: {k} ratio {q:.2f}")
    for name, v in m.items():
        if min(np.diff(v["wedge_L"])) < -0.2:
            fails.append(f"{name}: grey wedge not monotone")
    has_bal = "balanced_slots" in meta or "balanced_print" in meta
    if has_bal:
        c = np.array(m["grey balance"]["wedge_C"])[(STOPS >= -2) & (STOPS <= 3)]
        if c.max() > 1.0:
            fails.append(f"grey balance: wedge C* max {c.max():.2f} > 1.0")
    return fails


def main(profiles_root, hdri_dir, out_dir):
    root, out = pathlib.Path(profiles_root), pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = scene(hdri_dir)
    pdf = pymupdf.open()
    report = {}
    dirs = sorted((p for p in root.iterdir() if (p / "profile.json").exists()), key=lambda p: json.loads((p / "profile.json").read_text(encoding="utf-8"))["label"])
    for pdir in dirs:
        meta = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
        if meta.get("test") in (True, "true"):
            continue
        is_bw = meta.get("grain_mono") == "true" and "balanced_slots" not in meta and "balanced_print" not in meta
        rows, m, plain = [], {}, {}
        for name, bal, scan in MODES:
            lt, _ = profile_luts(pdir, bal, scan)
            shown, none, m[name] = measure(img, lt)
            rows.append(shown)
            plain[name] = none
        de = colour.delta_E(lab(plain["lab scan"][:SH, :2 * SW]), lab(plain["datasheet"][:SH, :2 * SW]), method="CIE 2000")
        m["lab scan"]["mean_dE2000_vs_datasheet"] = round(float(np.mean(de)), 2)
        fails = judge(m, meta, is_bw)
        report[meta["profile_id"]] = {"label": meta["label"], "pass": not fails, "fails": fails, "modes": m}
        q = (np.clip(np.concatenate([np.concatenate([r, np.ones((8,) + r.shape[1:])], 0) for r in rows], 0), 0, 1) * 255 + 0.5).astype(np.uint8)
        png = out / f"{meta['profile_id']}.png"
        pix = pymupdf.Pixmap(pymupdf.csRGB, q.shape[1], q.shape[0], q.tobytes(), False)
        pix.save(png)
        page = pdf.new_page(width=q.shape[1] / 2 + 40, height=q.shape[0] / 2 + 150)
        page.insert_text((20, 28), f"{meta['label']}  -  {'PASS' if not fails else 'FAIL'}", fontsize=16, color=(0, 0.5, 0) if not fails else (0.8, 0, 0))
        page.insert_image(pymupdf.Rect(20, 40, 20 + q.shape[1] / 2, 40 + q.shape[0] / 2), stream=pix.tobytes("jpeg", jpg_quality=85))
        y = 40 + q.shape[0] / 2 + 14
        for i, (name, _, _) in enumerate(MODES):
            page.insert_text((24, 40 + (i + 0.08) * (q.shape[0] / 2) / 3 + 10), name, fontsize=10, color=(1, 1, 1))
            v = m[name]
            extra = "".join(f"  {k}={v[k]}" for k in ("grain_rms_ratio", "halation_red_ratio", "mean_dE2000_vs_datasheet") if k in v)
            page.insert_text((20, y), f"{name}: grain rms {v['grain_rms']:.4f}  halation {v['halation_red']:.2f} (red {v['halation_red_share']:.0%}){extra}", fontsize=8)
            y += 12
        page.insert_text((20, y), "fails: " + ("; ".join(fails) if fails else "none"), fontsize=8, color=(0.8, 0, 0) if fails else (0, 0.5, 0))
        print(f"{meta['profile_id']:44s} {'PASS' if not fails else 'FAIL ' + '; '.join(fails)}", flush=True)
    pdf.save(out / "mode_check.pdf")
    (out / "mode_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    n = sum(r["pass"] for r in report.values())
    print(f"{n}/{len(report)} PASS")
    return 0 if n == len(report) else 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: mode_check.py PROFILES_DIR HDRI_NPY_DIR OUT_DIR")
    sys.exit(main(*sys.argv[1:]))
