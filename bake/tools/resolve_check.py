import json
import pathlib
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import colour
import numpy as np
import pymupdf
import tifffile

from filmemu import spaces
from filmemu.pipeline import Params, render

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILES = ROOT / "build" / "release" / "Fixer.ofx.bundle" / "Contents" / "Resources" / "profiles"
PROFILE = "kodak-vision3-250d-2383"
W, H = 1920, 1080
MAX_TOL, P99_TOL = 1e-2, 2e-3


def chart():
    d65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    dwg = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    xyz = colour.xyY_to_XYZ(np.array(list(cc.data.values())))
    xyz = colour.chromatic_adaptation(xyz, colour.xy_to_XYZ(cc.illuminant), colour.xy_to_XYZ(d65))
    lin_cc = colour.XYZ_to_RGB(xyz, dwg, illuminant=d65)
    img = np.zeros((H, W, 3))
    pw, ph = W // 6, 120
    for i, c in enumerate(lin_cc):
        r, k = divmod(i, 6)
        img[r * ph:(r + 1) * ph, k * pw:(k + 1) * pw] = c
    y = 4 * ph
    img[y:y + 120] = (0.18 * 2.0 ** np.linspace(-8, 6, W))[None, :, None]
    y += 120
    img[y:y + 120] = (0.18 * 2.0 ** np.repeat(np.arange(-7, 6), W // 13 + 1)[:W])[None, :, None]
    y += 120
    h = np.linspace(0, 1, W, endpoint=False)
    rows = H - y
    for j, s in enumerate((1.0, 0.6, 0.3)):
        band = colour.HSV_to_RGB(np.stack([h, np.full(W, s), np.ones(W)], -1)) * 0.5
        band = colour.RGB_to_RGB(band, colour.RGB_COLOURSPACES["ITU-R BT.709"], dwg)
        img[y + j * rows // 3:y + (j + 1) * rows // 3] = band[None]
    return img


def luts(pdir=None, bal=False, scan=False):
    pdir = pathlib.Path(pdir) if pdir else PROFILES / PROFILE
    meta = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
    n = meta["lut_size"]
    two = meta.get("kind") == "two_stage"
    bal = bal and ("balanced_slots" in meta or "balanced_print" in meta)
    scan = scan and "scan_print" in meta
    names = {s: s for s in ("under", "normal", "over", "print")}
    if bal:
        for s in meta.get("balanced_slots", {}):
            names[s] = f"{s}_bal"
        if "balanced_print" in meta:
            names["print"] = "print_bal"
    if scan:
        names["print"] = "print_bal_scan" if bal else "print_scan"
    load = lambda f: np.fromfile(pdir / f"{f}.bin", np.float32).reshape(n, n, n, 3)
    out = {s: load(names[s]) for s in ("under", "normal", "over")}
    if two:
        out["print"] = load(names["print"])
        out["print_domain"] = (meta["print_lo"], meta["print_hi"])
    return out


def variants(root):
    for pdir in sorted(p for p in pathlib.Path(root).iterdir() if (p / "profile.json").exists()):
        meta = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
        if meta.get("test") in (True, "true"):
            continue
        yield pdir, meta, "base", False, False
        if "balanced_slots" in meta or "balanced_print" in meta:
            yield pdir, meta, "bal", True, False
        if "scan_print" in meta:
            yield pdir, meta, "scan", False, True


def make_all(out_dir, root):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = to16(spaces.di_encode(chart().reshape(-1, 3))).reshape(H, W, 3)
    tifffile.imwrite(out / "w10_input_dwg_di.tif", q, photometric="rgb")
    rgba = np.concatenate([q.reshape(-1, 3) / 65535.0, np.ones((W * H, 1))], -1).astype(np.float32)
    rows = []
    for pdir, meta, tag, bal, scan in variants(root):
        exp = render(rgba, luts(pdir, bal, scan), Params(out_space="rec709_g24"))[:, :3].reshape(H, W, 3)
        tifffile.imwrite(out / f"{meta['profile_id']}__{tag}.tif", to16(exp), photometric="rgb")
        rows.append({"profile_id": meta["profile_id"], "label": meta["label"], "variant": tag, "grey_balance": bal, "lab_scan": scan})
        print(meta["profile_id"], tag, flush=True)
    (out / "variants.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def compare_all(export_dir, expected_dir):
    rows = json.loads((pathlib.Path(expected_dir) / "variants.json").read_text(encoding="utf-8"))
    fails = 0
    for r in rows:
        name = f"{r['profile_id']}__{r['variant']}"
        hits = sorted(pathlib.Path(export_dir).glob(f"{name}*.tif"))
        if not hits:
            print(f"MISSING {name}")
            fails += 1
            continue
        print(name, end=" ", flush=True)
        fails += compare(hits[0], pathlib.Path(expected_dir) / f"{name}.tif", str(pathlib.Path(export_dir) / f"{name}_diff.png")) != 0
    print(f"{len(rows) - fails}/{len(rows)} PASS")
    return 1 if fails else 0


def to16(a):
    return (np.clip(a, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def make(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    di = spaces.di_encode(chart().reshape(-1, 3))
    q = to16(di).reshape(H, W, 3)
    tifffile.imwrite(out / "w10_input_dwg_di.tif", q, photometric="rgb")
    rgba = np.concatenate([q.reshape(-1, 3) / 65535.0, np.ones((W * H, 1))], -1).astype(np.float32)
    params = Params(out_space="rec709_g24")
    exp = render(rgba, luts(), params)[:, :3].reshape(H, W, 3)
    tifffile.imwrite(out / "w10_expected_rec709.tif", to16(exp), photometric="rgb")
    print(f"wrote {out / 'w10_input_dwg_di.tif'} and {out / 'w10_expected_rec709.tif'}")


def compare(exported, expected, diff_png):
    a = tifffile.imread(exported).astype(np.float64)
    b = tifffile.imread(expected).astype(np.float64) / 65535.0
    a = a[..., :3] / (65535.0 if a.max() > 255 else 255.0)
    if a.shape != b.shape:
        print(f"FAIL shape exported={a.shape} expected={b.shape}")
        return 2
    err = np.abs(a - b)
    mx, p99 = err.max(), np.quantile(err, 0.99)
    worst = np.unravel_index(err.max(axis=-1).argmax(), err.shape[:2])
    vis = (np.clip(err.max(axis=-1) * 20.0, 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(vis[..., None], 3, -1).copy()
    pymupdf.Pixmap(pymupdf.csRGB, W, H, rgb.tobytes(), False).save(diff_png)
    ok = mx < MAX_TOL and p99 < P99_TOL
    print(f"{'PASS' if ok else 'FAIL'} max={mx:.2e} p99={p99:.2e} mean={err.mean():.2e} worst_pixel(y,x)={tuple(int(v) for v in worst)} "
          f"exported={a[worst].round(4).tolist()} expected={b[worst].round(4).tolist()} diff_png={diff_png}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "make":
        make(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "make-all":
        make_all(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else PROFILES.parent / "profiles")
    elif len(sys.argv) == 4 and sys.argv[1] == "compare-all":
        sys.exit(compare_all(sys.argv[2], sys.argv[3]))
    else:
        print("usage: resolve_check.py make OUT_DIR | compare EXPORTED.tif EXPECTED.tif DIFF.png | make-all OUT_DIR [PROFILES] | compare-all EXPORT_DIR EXPECTED_DIR")
        sys.exit(2)
