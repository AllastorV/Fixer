import json
import pathlib
import sys

import pymupdf

INK = (11 / 255, 13 / 255, 16 / 255)
PAPER = (247 / 255, 248 / 255, 249 / 255)
WHITE = (243 / 255, 244 / 255, 242 / 255)
CAP = (127 / 255, 136 / 255, 147 / 255)
RGB = {"r": (194 / 255, 72 / 255, 60 / 255), "g": (63 / 255, 138 / 255, 87 / 255), "b": (60 / 255, 98 / 255, 181 / 255)}
WORD = "FIXER"
TRACK = 0.018
KERN = {}
AXIS = (0.32, 0.35, 0.39)


def font(path):
    return pymupdf.Font(fontfile=str(path))


def advances(f, text, size, track, kern):
    out = []
    for i, c in enumerate(text):
        a = f.text_length(c, fontsize=size)
        if i + 1 < len(text):
            a += track * size + (kern.get(text[i:i + 2], 0.0) * size if kern else 0.0)
        out.append(a)
    return out


def tracked_width(f, text, size, track, kern=None):
    return sum(advances(f, text, size, track, kern))


def draw_tracked(page, f, fname, text, x, y, size, color, track, kern=None):
    for c, a in zip(text, advances(f, text, size, track, kern)):
        page.insert_text((x, y), c, fontsize=size, fontname=fname, color=color)
        x += a


def wordmark_svg(black, out, fg, bg):
    size = 100
    w = tracked_width(black, WORD, size, TRACK, KERN)
    pad = 0.0
    doc = pymupdf.open()
    page = doc.new_page(width=w + 2 * pad, height=size * 0.78)
    if bg is not None:
        page.draw_rect(page.rect, color=None, fill=bg)
    page.insert_font(fontname="blk", fontbuffer=black.buffer)
    draw_tracked(page, black, "blk", WORD, pad, size * 0.74, size, fg, TRACK, KERN)
    pathlib.Path(out).write_text(page.get_svg_image(text_as_path=True), encoding="utf-8")


def curves(stock_json):
    ch = json.loads(pathlib.Path(stock_json).read_text(encoding="utf-8"))["sensitometry"]["channels"]
    return {k: list(zip(ch[k]["x"], ch[k]["density"])) for k in ("b", "g", "r")}


def banner(black, medium, data, out, scale):
    W, H = 1280, 320
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.draw_rect(page.rect, color=None, fill=INK)
    page.insert_font(fontname="blk", fontbuffer=black.buffer)
    page.insert_font(fontname="med", fontbuffer=medium.buffer)
    ax0, ax1, ay0, ay1 = 812, 1232, 268, 52
    x0, x1, y0, y1 = ax0 + 14, ax1 - 10, ay0 - 12, ay1 + 14
    xs = [p[0] for c in data.values() for p in c]
    ds = [p[1] for c in data.values() for p in c]
    lo, hi, dlo, dhi = min(xs), max(xs), min(ds), max(ds)
    page.draw_line((ax0, ay1), (ax0, ay0), color=AXIS, width=0.9)
    page.draw_line((ax0, ay0), (ax1, ay0), color=AXIS, width=0.9)
    for i in range(1, 5):
        tx = ax0 + i * (ax1 - ax0) / 5
        page.draw_line((tx, ay0), (tx, ay0 + 6), color=AXIS, width=0.9)
        ty = ay0 - i * (ay0 - ay1) / 5
        page.draw_line((ax0 - 6, ty), (ax0, ty), color=AXIS, width=0.9)
    page.insert_text((ax1 - medium.text_length("log E", fontsize=11), ay0 + 22), "log E", fontsize=11, fontname="med", color=AXIS)
    page.insert_text((ax0 - 22, ay1 + 4), "D", fontsize=11, fontname="med", color=AXIS)
    for k in ("b", "g", "r"):
        pts = [pymupdf.Point(x0 + (s - lo) / (hi - lo) * (x1 - x0), y0 + (d - dlo) / (dhi - dlo) * (y1 - y0)) for s, d in data[k]]
        page.draw_polyline(pts, color=RGB[k], width=2.75)
    size = 0.096 * W
    left = 0.06 * W
    base = H / 2 + 0.36 * size * 0.72
    draw_tracked(page, black, "blk", WORD, left, base, size, WHITE, TRACK, KERN)
    cap = "FILM FROM THE DATASHEET · 36 STOCKS · DAVINCI RESOLVE"
    draw_tracked(page, medium, "med", cap, left + 0.004 * W, base + 0.014 * W + 18, 18, CAP, 0.05)
    page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(out)


def photo_banner(black, medium, bg, out, scale):
    W, H = 1280, 320
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    import numpy as np

    pix = pymupdf.Pixmap(str(bg))
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].astype(np.float64)
    u = np.linspace(0.0, 1.0, pix.width)
    t = np.clip((u - 0.08) / 0.60, 0.0, 1.0)
    keep = 0.10 + 0.90 * (t * t * (3 - 2 * t))
    ink = np.array(INK) * 255
    a = ink + (a - ink) * keep[None, :, None]
    q = np.ascontiguousarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))
    page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, pix.width, pix.height, q.tobytes(), False))
    page.insert_font(fontname="blk", fontbuffer=black.buffer)
    page.insert_font(fontname="med", fontbuffer=medium.buffer)
    size = 0.096 * W
    left = 0.055 * W
    base = H / 2 + 0.30 * size * 0.72
    draw_tracked(page, black, "blk", WORD, left, base, size, WHITE, TRACK, KERN)
    draw_tracked(page, medium, "med", "FILM FROM THE DATASHEET", left + 0.004 * W, base + 34, 17, WHITE, 0.08)
    page.insert_text((left + 2, H - 18), "Rendered by FIXER · Vision3 500T → 2383", fontsize=9, fontname="med", color=CAP)
    page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(out)


def icon(black, out, px):
    doc = pymupdf.open()
    page = doc.new_page(width=px, height=px)
    page.draw_rect(page.rect, color=None, fill=INK)
    page.insert_font(fontname="blk", fontbuffer=black.buffer)
    size = px * 0.72
    w = black.text_length("L", fontsize=size)
    page.insert_text(((px - w) / 2 - px * 0.005, px / 2 + size * 0.36), "L", fontsize=size, fontname="blk", color=WHITE)
    page.get_pixmap(alpha=False).save(out)


def main(font_dir, stock_json, out_dir, icon_out, bg=None):
    fd, od = pathlib.Path(font_dir), pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    black, medium = font(fd / "SchibstedGrotesk-900.ttf"), font(fd / "SchibstedGrotesk-500.ttf")
    wordmark_svg(black, od / "fixer-wordmark-white.svg", WHITE, None)
    wordmark_svg(black, od / "fixer-wordmark-ink.svg", INK, None)
    data = curves(stock_json)
    banner(black, medium, data, od / "fixer-banner.png", 1)
    banner(black, medium, data, od / "fixer-banner@2x.png", 2)
    for px in (256, 64, 32, 16):
        icon(black, od / f"fixer-icon-{px}.png", px)
    icon(black, icon_out, 128)
    if bg:
        photo_banner(black, medium, bg, od / "fixer-banner-photo.png", 1)
        photo_banner(black, medium, bg, od / "fixer-banner-photo@2x.png", 2)
    print("ok", sorted(p.name for p in od.iterdir()))


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        sys.exit("usage: brand.py FONT_DIR STOCK_JSON OUT_DIR PLUGIN_ICON_PNG [BACKGROUND_PNG]")
    main(*sys.argv[1:])
