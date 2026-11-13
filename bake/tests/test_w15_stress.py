import json
import os

os.environ["FIXER_NO_PROGRESS"] = "1"
import pathlib
import struct
import subprocess
import tempfile
import unittest
import warnings
import zlib

import numpy as np

warnings.simplefilter("ignore")

from filmemu import match, spaces
from tests.test_w03_cuda import SPACE_ID
from filmemu.pipeline import Params, render
from tests.test_w04_ofx import HOST, PLUGIN, PROFILE_INDEX, bundle_luts
from tests.test_w11_match import scene, srgb8
from tests.test_w11_match_native import states, write_png

STOCKS = ["kodak-vision3-250d-2383", "kodak-trix400-ilford-multigrade-iv-rc", "fuji-velvia50", "kodak-portra400-endura"]
HOSTILE = [np.nan, np.inf, -np.inf, 1e30, -1e30, 1e-40, -0.0, 3.4e38]


def frame(h, w, seed=0, space="dwg_di"):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    lin = np.stack([0.02 + 0.9 * xx / max(w - 1, 1), 0.02 + 0.6 * yy / max(h - 1, 1), 0.18 + 0.0 * xx], -1)
    lin = lin * rng.uniform(0.8, 1.2, (1, 1, 3))
    rgb = spaces.from_working(lin.reshape(-1, 3), space).reshape(h, w, 3)
    return np.concatenate([rgb, np.ones((h, w, 1))], -1).astype(np.float32)


def run(img, tmp, profile="kodak-vision3-250d-2383", window=None, in_space="dwg_di", out_space="dwg_di", exposure=0.0,
        push=0.0, impact=1.0, premult=0, grain=(0.0, 1.0, 0, 0.0), spatial=None, extra=(), env=None):
    tmp = pathlib.Path(tmp)
    (tmp / "in.raw").unlink(missing_ok=True)
    (tmp / "out.raw").unlink(missing_ok=True)
    (tmp / "in.raw").write_bytes(np.ascontiguousarray(img, np.float32).tobytes())
    h, w = img.shape[:2]
    window = window or (0, 0, w, h)
    args = [str(HOST), str(PLUGIN), str(tmp / "in.raw"), str(tmp / "out.raw"), str(w), str(h), str(SPACE_ID[in_space]),
            str(SPACE_ID[out_space]), repr(float(exposure)), repr(float(push)), repr(float(impact)), str(premult),
            *map(str, window), str(PROFILE_INDEX[profile]), *map(str, grain), *(map(str, spatial) if spatial else []), *extra]
    env = env or dict(os.environ, APPDATA=str(tmp / "appdata"))
    r = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=120)
    out = np.frombuffer((tmp / "out.raw").read_bytes(), np.float32).reshape(img.shape) if (tmp / "out.raw").exists() else None
    line = [l for l in r.stdout.splitlines() if l.startswith("match_state=")]
    return r, out, (line[-1][len("match_state="):] if line else None)


def make_jpeg(path):
    script = ("Add-Type -AssemblyName System.Drawing; $b = New-Object System.Drawing.Bitmap 48, 32; "
              "for ($x = 0; $x -lt 48; $x++) { for ($y = 0; $y -lt 32; $y++) { "
              "$b.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(40 + 4 * $x, 30 + 5 * $y, 90)) } }; "
              f"$b.Save('{path}', [System.Drawing.Imaging.ImageFormat]::Jpeg)")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True, timeout=60)
    return pathlib.Path(path).read_bytes()


def png_chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)


def png_raw(w, h, depth, ctype, rows):
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0)) +
            png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b""))


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class HostilePixels(unittest.TestCase):
    def test_nonfinite_and_huge_pixels_stay_local_with_glow_and_grain(self):
        h, w = 72, 128
        clean = frame(h, w, 3)
        dirty = clean.copy()
        for k, v in enumerate(HOSTILE):
            dirty[2 + k, 2:6, k % 3] = v
            dirty[2 + k, 6, :3] = v
        worst = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            for stock in STOCKS:
                for glow in ("glowQualityOption0", "glowQualityOption1"):
                    with self.subTest(stock=stock, glow=glow):
                        kw = dict(profile=stock, grain=(0.6, 1.0, 7, 3.0), spatial=(0.6, 2.0, 0.5), extra=["--press", glow])
                        r0, base, _ = run(clean, tmp, **kw)
                        r1, out, _ = run(dirty, tmp, **kw)
                        self.assertEqual(r1.returncode, 0, r1.stdout + r1.stderr)
                        self.assertIn("cuda_render_status=0", r1.stdout)
                        self.assertTrue(np.isfinite(out).all(), f"non-finite output pixels: {int((~np.isfinite(out)).sum())}")
                        far = np.abs(out[:, 64:, :3] - base[:, 64:, :3]).max()
                        worst = max(worst, float(far))
                        self.assertLess(far, 2e-3, "hostile pixels leaked into the far half of the frame")
        print(f"\n[measure] hostile pixels, far-field max |diff|: {worst:.2e}")

    def test_premultiplied_odd_alpha(self):
        h, w = 16, 32
        img = frame(h, w, 5)
        img[0, :8, 3] = [0.0, -0.0, -1.0, 1e-30, 2.0, 1e30, np.nan, np.inf]
        img[1, :4, 3] = 0.0
        img[1, :4, :3] = [[1.0, 0.5, 0.2], [np.nan, 0.1, 0.1], [5.0, 5.0, 5.0], [-2.0, 0.0, 0.0]]
        with tempfile.TemporaryDirectory() as tmp:
            for stock in STOCKS[:2]:
                with self.subTest(stock=stock):
                    r, out, _ = run(img, tmp, profile=stock, premult=1, spatial=(0.5, 2.0, 0.4))
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertTrue(np.isfinite(out[..., :3]).all(), np.argwhere(~np.isfinite(out[..., :3]))[:8].tolist())
                    fin = np.isfinite(img[..., 3])
                    np.testing.assert_array_equal(out[..., 3][fin], img[..., 3][fin])
                    zero = img[..., 3] == 0.0
                    self.assertLessEqual(float(np.abs(out[..., :3][zero]).max()), 1e-6, "alpha 0 premultiplied pixel got colour")


    def test_premultiplied_odd_alpha_matches_python(self):
        img = frame(4, 16, 6)
        img[0, :8, 3] = [0.0, -0.0, -1.0, 1e-30, 2.0, np.nan, np.inf, -np.inf]
        img[1, :8, 3] = [0.5, 0.0, np.nan, np.inf, -np.inf, -0.5, 1.0, 0.25]
        with tempfile.TemporaryDirectory() as tmp:
            r, out, _ = run(img, tmp, profile="synthetic-test", premult=1)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        want = render(img.reshape(-1, 4), bundle_luts(), Params(premultiplied=True)).reshape(img.shape)
        got = out.copy()
        both = np.isnan(got) & np.isnan(want)
        got[both] = want[both] = 0.0
        self.assertEqual(np.isfinite(got).tolist(), np.isfinite(want).tolist())
        fin = np.isfinite(want)
        err = float(np.max(np.abs(got[fin] - want[fin]) / np.maximum(1.0, np.abs(want[fin]))))
        print(f"\n[measure] premultiplied odd alpha CUDA vs Python: {err:.2e}")
        self.assertLess(err, 1e-4)

@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class Geometry(unittest.TestCase):
    def test_tiny_and_prime_frames_with_big_glow(self):
        with tempfile.TemporaryDirectory() as tmp:
            for h, w in ((1, 1), (1, 17), (17, 1), (2, 2), (3, 5), (7, 33), (31, 3)):
                for glow in ("glowQualityOption0", "glowQualityOption1"):
                    with self.subTest(size=(h, w), glow=glow):
                        img = frame(h, w, h * 31 + w)
                        r, out, _ = run(img, tmp, grain=(1.0, 3.0, 1, 0.0), spatial=(1.0, 50.0, 1.0), extra=["--press", glow])
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                        self.assertIn("cuda_render_status=0", r.stdout)
                        self.assertFalse((out == -7.0).any(), "pixels inside the window were not written")
                        self.assertTrue(np.isfinite(out).all())

    def test_tiles_equal_full_frame_with_fast_glow_and_grain(self):
        h, w = 61, 97
        img = frame(h, w, 9)
        img[20:24, 40:44, :3] = spaces.from_working(np.full((16, 3), 40.0), "dwg_di").reshape(4, 4, 3)
        worst = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            for glow in ("glowQualityOption1", "glowQualityOption0"):
                kw = dict(grain=(0.8, 1.3, 4, 2.0), spatial=(0.7, 3.0, 0.6), extra=["--press", glow])
                r, full, _ = run(img, tmp, **kw)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                tiled = np.full_like(full, -7.0)
                for x1, x2 in ((0, 33), (33, 34), (34, 97)):
                    for y1, y2 in ((0, 21), (21, 22), (22, 61)):
                        r, part, _ = run(img, tmp, window=(x1, y1, x2, y2), **kw)
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                        tiled[y1:y2, x1:x2] = part[y1:y2, x1:x2]
                err = float(np.abs(tiled - full).max())
                worst = max(worst, err)
                with self.subTest(glow=glow):
                    self.assertLess(err, 1e-4, f"tile seams: max |diff| {err:.3e}")
        print(f"\n[measure] odd tiles vs full frame (Full and Fast glow): {worst:.2e}")

    def test_fast_glow_tiles_with_small_sigma_and_spot_beside_the_seam(self):
        h, w = 61, 97
        img = frame(h, w, 10)
        img[19:21, 36:38, :3] = spaces.from_working(np.full((4, 3), 60.0), "dwg_di").reshape(2, 2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            kw = dict(spatial=(1.0, 0.2, 1.0), extra=["--press", "glowQualityOption1"])
            r, full, _ = run(img, tmp, **kw)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            tiled = np.full_like(full, -7.0)
            for x1, x2 in ((0, 33), (33, 97)):
                for y1, y2 in ((0, 16), (16, 61)):
                    r, part, _ = run(img, tmp, window=(x1, y1, x2, y2), **kw)
                    tiled[y1:y2, x1:x2] = part[y1:y2, x1:x2]
            err = float(np.abs(tiled - full).max())
            print(f"\n[measure] Fast glow small sigma, spot 3 px from seam: {err:.2e}")
            self.assertLess(err, 1e-4)

    def test_repeat_render_bit_identical(self):
        img = frame(64, 96, 11)
        with tempfile.TemporaryDirectory() as tmp:
            kw = dict(profile="fuji-velvia50", grain=(0.9, 1.1, 3, 5.0), spatial=(0.8, 4.0, 0.7), extra=["--press", "glowQualityOption0"])
            outs = [run(img, tmp, **kw)[1].tobytes() for _ in range(3)]
            self.assertEqual(len(set(outs)), 1)


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class ExtremeParameters(unittest.TestCase):
    def test_values_past_the_slider_ranges(self):
        img = frame(24, 40, 13)
        cases = [dict(exposure=40.0), dict(exposure=-40.0), dict(push=10.0), dict(push=-10.0), dict(impact=2.0), dict(impact=-1.0),
                 dict(grain=(50.0, 0.0, -5, -3.0)), dict(grain=(1.0, 1e6, 2**31 - 1, 1e9)), dict(spatial=(10.0, 1e5, 10.0)),
                 dict(spatial=(-1.0, -5.0, -1.0)), dict(exposure=float("nan")), dict(impact=float("inf"))]
        with tempfile.TemporaryDirectory() as tmp:
            for space in ("dwg_di", "acescct", "rec709_g24", "dwg_linear"):
                for c in cases:
                    with self.subTest(space=space, case=c):
                        r, out, _ = run(img, tmp, out_space=space, **c)
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                        if "impact" in c or "exposure" in c:
                            self.assertIn("cuda_render_status=0", r.stdout)
                        if "cuda_render_status=0" in r.stdout:
                            self.assertTrue(np.isfinite(out).all(), f"{int((~np.isfinite(out)).sum())} non-finite pixels")

    def test_every_stock_survives_hostile_frame(self):
        img = frame(12, 20, 17)
        img[0, :len(HOSTILE), 0] = HOSTILE
        bad = []
        with tempfile.TemporaryDirectory() as tmp:
            for pid in PROFILE_INDEX:
                r, out, _ = run(img, tmp, profile=pid, out_space="rec709_g24", grain=(0.5, 1.0, 1, 0.0), spatial=(0.3, 1.0, 0.3))
                if r.returncode or "cuda_render_status=0" not in r.stdout or not np.isfinite(out).all():
                    bad.append(pid)
        self.assertEqual(bad, [])


def mutations(good):
    text = json.dumps(good)
    rng = np.random.default_rng(4)
    yield "empty", ""
    yield "spaces", "   "
    yield "null", "null"
    yield "array", "[]"
    yield "number", "1"
    yield "bom", "﻿" + text
    yield "nan curve", text.replace(json.dumps(good["curve"]), json.dumps([[0, float("nan")], [100, 100]]))
    yield "infinity", json.dumps(dict(good, sat=float("inf")))
    yield "1e309", text.replace(f'"sat": {json.dumps(good["sat"])}', '"sat": 1e309')
    yield "string sat", json.dumps(dict(good, sat="1.0"))
    yield "bool sat", json.dumps(dict(good, sat=True))
    yield "nested deep", '{"version":1,"curve":' + "[" * 200000 + "]" * 200000 + "}"
    yield "object deep", '{"a":' * 100000 + "1" + "}" * 100000
    yield "huge string", json.dumps(dict(good, mode="x" * 5_000_000))
    yield "huge curve", json.dumps(dict(good, curve=[[i / 1000, i / 1000] for i in range(100000)]))
    yield "missing tint", json.dumps({k: v for k, v in good.items() if k != "tint"})
    yield "missing curve", json.dumps({k: v for k, v in good.items() if k != "curve"})
    yield "tint short", json.dumps(dict(good, tint={"sh": [1.0], "mid": [1.0, 1.0], "hl": [1.0, 1.0]}))
    yield "duplicate keys", text[:-1] + ', "sat": 2.4}'
    yield "trailing garbage", text + "}}}"
    yield "unterminated string", text[:-1] + ', "x": "abc'
    yield "bad escape", text[:-1] + ', "x": "\\u12"}'
    yield "control char", text[:-1] + ', "x": "a\x01b"}'
    yield "negative zero version", json.dumps(dict(good, version=-0))
    yield "float version", json.dumps(dict(good, version=1.5))
    yield "huge version", json.dumps(dict(good, version=10**30))
    for k in range(40):
        cut = int(rng.integers(1, len(text)))
        yield f"truncated {cut}", text[:cut]
    for k in range(40):
        b = bytearray(text, "utf-8")
        for _ in range(3):
            b[int(rng.integers(0, len(b)))] = int(rng.integers(32, 127))
        yield f"flipped {k}", b.decode("utf-8", "replace")


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class HostileState(unittest.TestCase):
    def test_fuzzed_match_state_never_crashes(self):
        img = frame(16, 24, 21)
        good = states()[0]
        crashed, nonfinite = [], []
        with tempfile.TemporaryDirectory() as tmp:
            _, base, _ = run(img, tmp)
            path = pathlib.Path(tmp) / "s.json"
            for why, text in mutations(good):
                path.write_text(text, encoding="utf-8", errors="surrogatepass")
                for flag in ("--match-state", "--look-load"):
                    r, out, _ = run(img, tmp, extra=[flag, str(path), "--match-strength", "0.8"])
                    if r.returncode != 0 or "cuda_render_status=0" not in r.stdout:
                        crashed.append((why, flag, r.returncode, r.stderr[-200:]))
                    elif not np.isfinite(out).all():
                        nonfinite.append((why, flag))
        self.assertEqual(crashed, [])
        self.assertEqual(nonfinite, [])

    def ramp_drops(self, indices, protects):
        img = frame(8, 256, 23)
        img[4, :, :3] = spaces.from_working(np.repeat(np.geomspace(1e-4, 50.0, 256)[:, None], 3, 1), "dwg_di")
        drops = []
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "s.json"
            for i in indices:
                path.write_text(json.dumps(states()[i]), encoding="utf-8")
                for source in ("0", "1"):
                    for s, p in protects:
                        r, out, _ = run(img, tmp, out_space="dwg_linear",
                                        extra=["--match-state", str(path), "--match-strength", s, "--match-protect", p, "--match-source", source])
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                        self.assertTrue(np.isfinite(out).all(), (i, source, s, p))
                        self.assertNotIn("invalid", r.stdout, (i, source))
                        y = out[4, :, :3] @ match.Y_W
                        worst = float((np.diff(y) / np.maximum(y[1:], 1e-6)).min())
                        if worst < -5e-4:
                            drops.append((i, source, s, p, round(worst, 5)))
        return drops

    def test_fitted_states_keep_grey_ramp_monotone(self):
        self.assertEqual(self.ramp_drops(range(6), (("1.0", "0.0"), ("0.3", "0.0"))), [])

    @unittest.expectedFailure
    def test_known_defect_protect_skin_bends_warm_highlights(self):
        self.assertEqual(self.ramp_drops((0, 2, 4, 5), (("1.0", "1.0"),)), [])

    @unittest.expectedFailure
    def test_known_defect_parser_accepts_non_monotone_edge_states(self):
        self.assertEqual(self.ramp_drops((6, 7), (("1.0", "0.0"),)), [])


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class HostileReference(unittest.TestCase):
    maxDiff = None
    def references(self, d):
        d = pathlib.Path(d)
        files = {}
        files["empty"] = b""
        files["text"] = b"not a png at all\n"
        good = np.round(srgb8(scene(32, 48)) * 255).astype(np.uint8)
        write_png(d / "good.png", good)
        raw = (d / "good.png").read_bytes()
        files["truncated"] = raw[: len(raw) // 2]
        files["bad crc"] = raw[:30] + bytes([raw[30] ^ 0xFF]) + raw[31:]
        files["one pixel"] = png_raw(1, 1, 8, 2, b"\x00\x80\x40\x20")
        files["huge header"] = png_raw(60000, 60000, 8, 2, b"\x00" * 16)
        files["zero size"] = png_raw(0, 0, 8, 2, b"")
        files["grey 16"] = png_raw(4, 2, 16, 0, b"".join(b"\x00" + bytes(range(8)) for _ in range(2)))
        files["rgba"] = png_raw(2, 2, 8, 6, b"".join(b"\x00" + bytes([200, 100, 50, 0, 10, 20, 30, 255]) for _ in range(2)))
        files["palette no plte"] = png_raw(2, 2, 8, 3, b"\x00\x00\x00" * 2)
        files["black"] = png_raw(16, 16, 8, 2, b"".join(b"\x00" + b"\x00" * 48 for _ in range(16)))
        files["white"] = png_raw(16, 16, 8, 2, b"".join(b"\x00" + b"\xff" * 48 for _ in range(16)))
        files["pure red"] = png_raw(16, 16, 8, 2, b"".join(b"\x00" + b"\xff\x00\x00" * 16 for _ in range(16)))
        files["fake jpeg"] = b"\xff\xd8\xff\xe0" + b"\x00" * 64
        jpeg = make_jpeg(d / "source.jpg")
        files["jpeg"] = jpeg
        files["jpeg truncated"] = jpeg[: len(jpeg) // 2]
        files["jpeg motion photo"] = jpeg + bytes(np.random.default_rng(1).integers(0, 256, 3_000_000, dtype=np.uint8)).replace(b"\xff\xd9", b"\xff\x00")
        files["png trailing data"] = raw + b"metadata appended after IEND" * 100
        out = {}
        for k, v in files.items():
            p = d / (k.replace(" ", "_") + (".jpg" if "jpeg" in k else ".png"))
            p.write_bytes(v)
            out[k] = p
        out["directory"] = d
        out["unicode path"] = d / "referans ışık ğ.png"
        out["unicode path"].write_bytes(raw)
        return out

    def test_create_grade_with_broken_references(self):
        img = frame(32, 48, 25)
        problems = []
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as refs:
            for why, path in self.references(refs).items():
                for source in ("matchSourceOption0", "matchSourceOption1"):
                    r, out, js = run(img, tmp, extra=["--press", source, "--match-create", str(path)])
                    status = [l for l in r.stdout.splitlines() if l.startswith("match_create_status=")]
                    if r.returncode != 0 or not status:
                        problems.append((why, source, "crash", r.returncode, r.stderr[-200:]))
                        continue
                    ok = status[-1].endswith("=0")
                    if ok:
                        try:
                            st = json.loads(js)
                        except ValueError:
                            problems.append((why, source, "status 0 but state is not JSON"))
                            continue
                        lo, hi = (0.05, 2.5) if st.get("version") == 3 else (0.85, 1.25)
                        if not (lo - 1e-9 <= st["sat"] <= hi + 1e-9):
                            problems.append((why, source, "saturation out of bounds", st["sat"]))
                        if not np.isfinite(out).all():
                            problems.append((why, source, "non-finite render"))
                    elif js not in ("", None):
                        problems.append((why, source, "failed but left a state behind"))
                    if why in ("empty", "text", "truncated", "zero size", "directory", "fake jpeg", "huge header", "jpeg truncated") and ok:
                        problems.append((why, source, "broken reference accepted"))
                    if why in ("unicode path", "one pixel", "black", "white", "pure red", "jpeg", "jpeg motion photo", "png trailing data") and not ok:
                        problems.append((why, source, "usable reference refused", [l for l in r.stdout.splitlines() if "FIXER" in l and "needs CUDA" not in l][-1:]))
        self.assertEqual(problems, [])

    def test_create_grade_on_flat_and_hostile_frames(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as refs:
            ref = pathlib.Path(refs) / "ref.png"
            write_png(ref, np.round(srgb8(scene(32, 48)) * 255).astype(np.uint8))
            frames = {"black": np.zeros((32, 48, 4), np.float32), "nan": np.full((32, 48, 4), np.nan, np.float32),
                      "one pixel": frame(1, 1, 1), "huge": np.full((32, 48, 4), 1e30, np.float32), "hostile mix": frame(32, 48, 2)}
            frames["hostile mix"][::3, ::5, :3] = np.nan
            frames["hostile mix"][1::3, ::7, :3] = np.inf
            problems = []
            for why, img in frames.items():
                r, out, js = run(img, tmp, extra=["--match-create", str(ref)])
                status = [l for l in r.stdout.splitlines() if l.startswith("match_create_status=")]
                if r.returncode != 0 or not status:
                    problems.append((why, "crash", r.returncode))
                    continue
                if status[-1].endswith("=0"):
                    st = json.loads(js)
                    vals = [st["sat"], *sum((st["tint"][z] for z in ("sh", "mid", "hl")), []), *sum(st["curve"], [])]
                    if not np.isfinite(vals).all():
                        problems.append((why, "non-finite fit"))
                if out is None or not np.isfinite(out[..., :3]).all():
                    problems.append((why, "non-finite render"))
            self.assertEqual(problems, [])


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class HostileLookNames(unittest.TestCase):
    maxDiff = None
    def test_look_names_stay_inside_the_looks_folder(self):
        img = frame(8, 8, 27)
        names = ["CON", "nul", "COM1", "LPT9.x", "a" * 240, "a" * 300, "tab\tname", "ctrl\x01x", "end.", "end ", " start",
                 "..", ". .", "..\\..\\evil", "../evil", "C:evil", "ok‮gnp.exe", "🎞 film", "x" * 4000, "trailing.json",
                 "ünïcödé İı", "nul.txt", "AUX", "space in name"]
        problems = []
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "appdata"
            root.mkdir()
            env = dict(os.environ, APPDATA=str(root))
            state = pathlib.Path(tmp) / "state.json"
            state.write_text(json.dumps(states()[0]), encoding="utf-8")
            looks = root / "FIXER" / "looks"
            for name in names:
                before = {p for p in pathlib.Path(tmp).rglob("*")}
                r, _, _ = run(img, tmp, env=env, extra=["--match-state", str(state), "--look-save", name])
                if r.returncode != 0 or "look_save_status=" not in r.stdout:
                    problems.append((name[:40], "crash", r.returncode))
                    continue
                ok = "look_save_status=0" in r.stdout
                new = {p for p in pathlib.Path(tmp).rglob("*")} - before - {pathlib.Path(tmp) / "in.raw", pathlib.Path(tmp) / "out.raw"}
                outside = [str(p) for p in new if looks not in p.parents and p not in (root / "FIXER", looks) and root / "NVIDIA" not in (p, *p.parents)]
                if outside:
                    problems.append((name[:40], "wrote outside the looks folder", outside))
                if ok:
                    jsons = [p for p in new if p.suffix == ".json" and p.parent == looks]
                    cubes = [p for p in new if p.suffix == ".cube" and p.parent == looks]
                    if len(jsons) != 1 or len(cubes) != 1:
                        problems.append((name[:40], "reported success but files are missing", sorted(p.name for p in new)))
                    else:
                        rr, _, js = run(img, tmp, env=env, extra=["--look-load", str(jsons[0])])
                        if "look_load_status=0" not in rr.stdout:
                            problems.append((name[:40], "saved look cannot be loaded back"))
                    for p in new:
                        if p.is_file():
                            p.unlink()
        self.assertEqual(problems, [])

    def test_look_load_odd_files(self):
        img = frame(8, 8, 29)
        good = json.dumps(states()[0])
        files = {"utf8 bom": ("﻿" + good).encode("utf-8"), "utf16": good.encode("utf-16"), "binary": bytes(range(256)) * 64,
                 "huge": b"{" + b" " * 50_000_000 + b"}", "crlf pretty": json.dumps(states()[0], indent=2).replace("\n", "\r\n").encode()}
        problems = []
        with tempfile.TemporaryDirectory() as tmp:
            for why, data in files.items():
                p = pathlib.Path(tmp) / "look.json"
                p.write_bytes(data)
                r, out, js = run(img, tmp, extra=["--look-load", str(p)])
                if r.returncode != 0 or "look_load_status=" not in r.stdout:
                    problems.append((why, "crash", r.returncode))
                    continue
                ok = "look_load_status=0" in r.stdout
                if why in ("crlf pretty", "utf8 bom") and not ok:
                    problems.append((why, "valid look refused", [l for l in r.stdout.splitlines() if "FIXER" in l and "needs CUDA" not in l][-1:]))
                if why in ("binary", "huge") and ok:
                    problems.append((why, "garbage accepted"))
                if not np.isfinite(out).all():
                    problems.append((why, "non-finite render"))
            r, _, _ = run(img, tmp, extra=["--look-load", str(pathlib.Path(tmp) / "nope" / "missing.json")])
            if r.returncode != 0 or "look_load_status=0" in r.stdout:
                problems.append(("missing file", r.returncode))
            r, _, _ = run(img, tmp, extra=["--look-load", tmp])
            if r.returncode != 0 or "look_load_status=0" in r.stdout:
                problems.append(("directory", r.returncode))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
