import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
import warnings
import zlib

warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import colour
import numpy as np

from filmemu import match, spaces
from tests.test_w04_ofx import HOST, PLUGIN, PROFILE_INDEX, ROOT

HDRI = pathlib.Path(os.environ.get("FIXER_HDRI_DIR", ROOT / "build" / "match-fixtures" / "hdri"))
REF_DIR = pathlib.Path(os.environ.get("FIXER_REF_DIR", ROOT / "build" / "match-fixtures" / "references"))
OUT = ROOT / "build" / "w23-match-v3"
SCENES = ("city", "interior", "courtyard")
REFS = (("desaturated", "03-desaturated/03-001.jpg"), ("warm amber", "10-warm-amber/10-001.jpg"),
        ("teal orange", "01-teal-and-orange/01-002.jpg"), ("lifted blacks", "22-lifted-blacks/22-001.jpg"),
        ("blue sky warm ground", "01-teal-and-orange/01-001.jpg"))
STOCK = "kodak-vision3-250d-2383"
W, H = 1920, 1080
REF_W = 1024
STRENGTH, PROTECT = 1.0, 0.5
SOURCES = {"film": "0", "only": "1"}
MODE_NAME = {"film": "FILM + MATCH", "only": "MATCH ONLY"}
CHROMA_SCALE = 66.0

FONT = {
    "A": "01110100011000111111100011000110001", "B": "11110100011000111110100011000111110", "C": "01110100011000010000100001000101110",
    "D": "11110100011000110001100011000111110", "E": "11111100001000011110100001000011111", "F": "11111100001000011110100001000010000",
    "G": "01110100011000010111100011000101111", "H": "10001100011000111111100011000110001", "I": "01110001000010000100001000010001110",
    "J": "00111000100001000010000101001001100", "K": "10001100101010011000101001001010001", "L": "10000100001000010000100001000011111",
    "M": "10001110111010110101100011000110001", "N": "10001100011100110101100111000110001", "O": "01110100011000110001100011000101110",
    "P": "11110100011000111110100001000010000", "Q": "01110100011000110001101011001001101", "R": "11110100011000111110101001001010001",
    "S": "01111100001000001110000010000111110", "T": "11111001000010000100001000010000100", "U": "10001100011000110001100011000101110",
    "V": "10001100011000110001100010101000100", "W": "10001100011000110101101011010101010", "X": "10001100010101000100010101000110001",
    "Y": "10001100010101000100001000010000100", "Z": "11111000010001000100010001000011111", "0": "01110100011001110101110011000101110",
    "1": "00100011000010000100001000010001110", "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
    "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110", "6": "00110010001000011110100011000101110",
    "7": "11111000010001000100010000100001000", "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100",
    ".": "00000000000000000000000000110001100", "-": "00000000000000011111000000000000000", "+": "00000001000010011111001000010000000",
    "=": "00000000001111100000111110000000000", "/": "00000000010001000100010001000000000", ":": "00000011000110000000011000110000000",
    "_": "00000000000000000000000000000011111", " ": "0" * 35,
}


def text_mask(text, scale=2):
    glyphs = [np.array([int(c) for c in FONT.get(ch, FONT[" "])], bool).reshape(7, 5) for ch in text.upper()]
    if not glyphs:
        return np.zeros((7 * scale, 1), bool)
    row = np.concatenate([np.pad(g, ((0, 0), (0, 1))) for g in glyphs], 1)
    return np.kron(row, np.ones((scale, scale), bool))


def draw_text(img, text, x, y, scale=2, colour_=(1.0, 1.0, 1.0)):
    m = text_mask(text, scale)
    h, w = m.shape
    h, w = min(h, img.shape[0] - y), min(w, img.shape[1] - x)
    if h > 0 and w > 0:
        img[y:y + h, x:x + w][m[:h, :w]] = colour_


def write_png(path, rgb8):
    rgb8 = np.ascontiguousarray(rgb8, np.uint8)
    h, w, _ = rgb8.shape
    raw = b"".join(b"\x00" + rgb8[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    pathlib.Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                                   chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_bmp24(path):
    b = pathlib.Path(path).read_bytes()
    off, w, h, bpp = struct.unpack_from("<I", b, 10)[0], *struct.unpack_from("<ii", b, 18), struct.unpack_from("<H", b, 28)[0]
    if bpp != 24:
        raise ValueError(f"{path}: {bpp} bpp")
    stride = (w * 3 + 3) & ~3
    rows = np.frombuffer(b, np.uint8, stride * abs(h), off).reshape(abs(h), stride)[:, :w * 3].reshape(abs(h), w, 3)[..., ::-1]
    return rows[::-1] if h > 0 else rows


def decode_jpegs(pairs, work):
    lines = ["Add-Type -AssemblyName System.Drawing"]
    for jpg, bmp in pairs:
        lines.append(f"$j = New-Object System.Drawing.Bitmap '{jpg}'; $k = $j.Clone((New-Object System.Drawing.Rectangle 0,0,$j.Width,$j.Height), "
                     f"[System.Drawing.Imaging.PixelFormat]::Format24bppRgb); $k.Save('{bmp}', [System.Drawing.Imaging.ImageFormat]::Bmp); $k.Dispose(); $j.Dispose()")
    script = pathlib.Path(work) / "decode_jpeg.ps1"
    script.write_text("\n".join(lines), encoding="utf-8-sig")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)


def resize(img, h, w):
    sh, sw = img.shape[:2]
    y = np.clip((np.arange(h) + 0.5) * sh / h - 0.5, 0, sh - 1)
    x = np.clip((np.arange(w) + 0.5) * sw / w - 0.5, 0, sw - 1)
    y0, x0 = np.minimum(y.astype(int), sh - 2), np.minimum(x.astype(int), sw - 2)
    fy, fx = (y - y0)[:, None, None], (x - x0)[None, :, None]
    a, b = img[y0][:, x0], img[y0][:, x0 + 1]
    c, d = img[y0 + 1][:, x0], img[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def crop(name, offset=0):
    a = np.load(HDRI / f"{name}.npy").astype(np.float64)
    band = a[a.shape[0] // 4: 3 * a.shape[0] // 4]
    cw = int(round(band.shape[0] * W / H))
    y = band @ match.Y_W
    ly = np.log2(np.clip(y, np.percentile(y, 1), np.percentile(y, 99)) + 1e-6)
    scores = [float(np.roll(ly, -x, 1)[:, :cw].std()) for x in range(0, band.shape[1], 16)]
    x = (int(np.argmax(scores)) * 16 + offset) % band.shape[1]
    lin = np.maximum(resize(np.roll(band, -x, 1)[:, :cw], H, W), 0.0)
    return lin * (0.18 / np.median(np.maximum(lin @ match.Y_W, 1e-6)))


def encode_di(lin709):
    rgb = spaces.di_encode(np.asarray(lin709, np.float64) @ match.REC709_TO_DWG.T)
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1).astype(np.float32)


def srgb8(lin):
    return np.round(np.clip(colour.models.eotf_inverse_sRGB(np.clip(lin, 0.0, 1.0)), 0.0, 1.0) * 255.0).astype(np.uint8)


def host(img, tmp, extra=(), out_space=3, tag="a"):
    tmp = pathlib.Path(tmp)
    src, dst = tmp / f"in_{tag}.raw", tmp / f"out_{tag}.raw"
    src.write_bytes(np.ascontiguousarray(img, np.float32).tobytes())
    if dst.exists():
        dst.unlink()
    h, w = img.shape[:2]
    args = [str(HOST), str(PLUGIN), str(src), str(dst), str(w), str(h), "0", str(out_space), "0", "0", "1", "0", "0", "0", str(w), str(h),
            str(PROFILE_INDEX[STOCK]), "0", "1", "0", "0", "0", "1", "0", *extra]
    e = dict(os.environ, FIXER_TIMING="1", FIXER_NO_PROGRESS="1")
    t0 = time.perf_counter()
    r = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", env=e)
    wall = time.perf_counter() - t0
    out = np.frombuffer(dst.read_bytes(), np.float32).reshape(img.shape) if dst.exists() else None
    line = [l for l in r.stdout.splitlines() if l.startswith("match_state=")]
    state = line[-1][len("match_state="):] if line else None
    timing = [l for l in r.stdout.splitlines() if "create grade read frame" in l]
    total = None
    if timing:
        part = timing[-1].split("total ")[1]
        total = (float(part.split(" ms")[0]), float(part.split("progress window ")[1].split(" ms")[0]))
    return {"r": r, "out": out, "state": json.loads(state) if state else None, "create_ms": total, "wall_s": wall}


def create(img, ref_path, source, tmp, strength=STRENGTH, protect=PROTECT):
    res = host(img, tmp, ["--match-strength", repr(strength), "--match-protect", repr(protect), "--match-source", SOURCES[source],
                          "--match-create", str(ref_path)])
    if "match_create_status=0" not in res["r"].stdout or res["out"] is None:
        raise RuntimeError(res["r"].stdout + res["r"].stderr)
    return res


def before_film(img, tmp):
    res = host(img, tmp)
    if res["r"].returncode:
        raise RuntimeError(res["r"].stdout + res["r"].stderr)
    return res["out"]


def before_only(img):
    lin = spaces.sanitize(spaces.to_working(spaces.sanitize(img[..., :3]), "dwg_di"))
    out = spaces.from_working(lin, "rec709_g24")
    return np.concatenate([out, img[..., 3:]], -1)


def display_stats(out):
    return match.stats(match.downsample(out[..., :3]), "rec709")


def ref_stats(rgb8):
    return match.stats(match.downsample(rgb8 / 255.0), "srgb")


def lab(out, step=3):
    return match.lab_709(spaces.g24_decode(np.clip(match.finite(out[::step, ::step, :3]), 0.0, 1.0)))


def de00(a, b, step=3):
    return float(colour.delta_E(lab(a, step), lab(b, step), method="CIE 2000").mean())


def gaps(a, r):
    zones = [k for k in ("sh", "mid", "hl") if a["tint"][k] and r["tint"][k]]
    return {"L": float(np.mean(np.abs(np.array(a["L"]) - np.array(r["L"])))),
            "chroma": CHROMA_SCALE * abs(a["sat"] - r["sat"]) if a["sat"] is not None and r["sat"] is not None else 0.0,
            "ab": float(np.mean([np.hypot(a["tint"][k][1] - r["tint"][k][1], a["tint"][k][2] - r["tint"][k][2]) for k in zones])) if zones else 0.0}


def closed(g0, g1):
    out = {k: (1.0 - g1[k] / g0[k] if g0[k] > 1e-6 else None) for k in g0}
    s0, s1 = sum(g0.values()), sum(g1.values())
    out["total"] = 1.0 - s1 / s0 if s0 > 1e-6 else None
    return out


def zone_ab(st):
    return {k: (None if v is None else [round(v[1], 3), round(v[2], 3)]) for k, v in st["tint"].items()}


def load_refs(work):
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    refs = [{"style": s, "path": str(REF_DIR / p), "bmp": str(work / f"ref{i}.bmp")} for i, (s, p) in enumerate(REFS)]
    decode_jpegs([(r["path"], r["bmp"]) for r in refs], work)
    for r in refs:
        r["rgb8"] = read_bmp24(r.pop("bmp"))
    return refs


def same_ref(before, work, name, source):
    rgb8 = srgb8(spaces.g24_decode(np.clip(match.downsample(before[..., :3], REF_W), 0.0, 1.0)))
    png = pathlib.Path(work) / f"self_{name}_{source}.png"
    write_png(png, rgb8)
    return png


def fit_box(rgb01, bw, bh):
    h, w = rgb01.shape[:2]
    s = min(bw / w, bh / h)
    th, tw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return np.clip(resize(match.downsample(rgb01, max(tw, 2)), th, tw), 0.0, 1.0)


def contact_sheet(path, title, rows):
    tw, th, pad, lab_h = 488, 275, 8, 22
    width = pad + 3 * (tw + pad)
    top = 64
    height = top + len(rows) * (th + lab_h + pad) + pad
    sheet = np.full((height, width, 3), 0.12)
    draw_text(sheet, title, pad, 10, 3)
    for c, head in enumerate(("BEFORE", "REFERENCE", "RESULT S1.0 P0.5")):
        draw_text(sheet, head, pad + c * (tw + pad), 42, 2, (0.75, 0.75, 0.75))
    y = top
    for label, images in rows:
        draw_text(sheet, label, pad, y + 4, 2, (1.0, 0.9, 0.5))
        for c, im in enumerate(images):
            t = fit_box(im, tw, th)
            x = pad + c * (tw + pad) + (tw - t.shape[1]) // 2
            yy = y + lab_h + (th - t.shape[0]) // 2
            sheet[yy:yy + t.shape[0], x:x + t.shape[1]] = t
        y += th + lab_h + pad
    write_png(path, np.round(sheet * 255.0))


def fmt(v):
    return "NA" if v is None else f"{v:.2f}"


def run(out_dir=None, sheets=True):
    tmpdir = tempfile.TemporaryDirectory()
    tmp = tmpdir.name
    refs = load_refs(pathlib.Path(tmp) / "refs")
    metrics = {"stock": STOCK, "frame_size": [W, H], "strength": STRENGTH, "protect": PROTECT, "chroma_scale": CHROMA_SCALE,
               "gap_definition": "gap = |stat(image) - stat(reference)|; L = mean abs diff of L* percentiles " + str(list(match.PCTS3)) +
                                 ", chroma = CHROMA_SCALE * abs diff of mid-zone median C*/(L*+16), ab = mean over zones of a*b* distance; "
                                 "closed = 1 - gap_result / gap_before; total uses the sum of the three gaps",
               "pairs": [], "identity": [], "desaturated_unprotected": []}
    rows = {s: [] for s in SOURCES}
    try:
        for name in SCENES:
            img = encode_di(crop(name))
            befores = {"film": before_film(img, tmp), "only": before_only(img)}
            for src in SOURCES:
                b = befores[src]
                bs = display_stats(b)
                for ref in refs:
                    rs = ref_stats(ref["rgb8"])
                    c = create(img, ref["path"], src, tmp)
                    out = c["out"]
                    st = display_stats(out)
                    g0, g1 = gaps(bs, rs), gaps(st, rs)
                    gc = closed(g0, g1)
                    e = {"frame": name, "source": src, "style": ref["style"], "ref_file": str(pathlib.Path(ref["path"]).relative_to(REF_DIR)),
                         "ref_size": list(ref["rgb8"].shape[1::-1]), "gap_before": g0, "gap_result": g1, "gap_closed": gc,
                         "L_pcts": {"before": bs["L"], "ref": rs["L"], "result": st["L"]},
                         "chroma": {"before": bs["sat"], "ref": rs["sat"], "result": st["sat"],
                                    "result_over_before": st["sat"] / bs["sat"] if bs["sat"] else None},
                         "zone_ab": {"before": zone_ab(bs), "ref": zone_ab(rs), "result": zone_ab(st)},
                         "ref_stats_plugin_vs_python": gaps(c["state"]["ref"], rs),
                         "fit": {"curve": c["state"]["curve"], "sat": c["state"]["sat"], "tint": c["state"]["tint"]},
                         "finite": bool(np.isfinite(out).all()), "de00_vs_before": de00(out, b),
                         "create_ms": c["create_ms"], "create_wall_s": c["wall_s"]}
                    metrics["pairs"].append(e)
                    rows[src].append((f"{name} / {ref['style']}   gap closed {fmt(gc['total'])}   L {fmt(gc['L'])}  chroma {fmt(gc['chroma'])}  ab {fmt(gc['ab'])}",
                                      [b[..., :3], ref["rgb8"] / 255.0, out[..., :3]]))
                    print(name, src, ref["style"], f"closed {fmt(gc['total'])} L {fmt(gc['L'])} C {fmt(gc['chroma'])} ab {fmt(gc['ab'])}",
                          f"create {c['create_ms']}", flush=True)
                desaturated = next(r for r in refs if r["style"] == "desaturated")
                unprotected = create(img, desaturated["path"], src, tmp, protect=0.0)
                us = display_stats(unprotected["out"])
                metrics["desaturated_unprotected"].append({"frame": name, "source": src,
                    "chroma_ratio": us["sat"] / bs["sat"], "finite": bool(np.isfinite(unprotected["out"]).all())})
                c = create(img, same_ref(b, tmp, name, src), src, tmp)
                metrics["identity"].append({"frame": name, "source": src, "de00_vs_before": de00(c["out"], b),
                                            "finite": bool(np.isfinite(c["out"]).all()), "create_ms": c["create_ms"]})
        if out_dir is not None:
            out_dir = pathlib.Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if sheets:
                for src in SOURCES:
                    contact_sheet(out_dir / f"sheet_{src}.png", f"FIXER REFERENCE MATCH V3  {MODE_NAME[src]}  250D", rows[src])
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    finally:
        tmpdir.cleanup()
    return metrics


if __name__ == "__main__":
    run(OUT)
