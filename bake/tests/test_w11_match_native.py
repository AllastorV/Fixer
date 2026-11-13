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

from filmemu import lut, match, spaces
from filmemu.pipeline import Params, render
from tests.test_w03_cuda import CLI, SPACE_ID, make_image, make_luts
from tests.test_w04_ofx import HOST, PLUGIN, PROFILE_INDEX, bundle_luts
from tests.test_w11_match import legacy_v2, scene, srgb8, warm_sat


def states():
    lin = scene()
    frame = match.stats(match.downsample(spaces.g24_encode(np.clip(lin, 0.0, 1.0))), "rec709")
    ref = match.stats(match.downsample(srgb8(warm_sat(lin * 0.7))), "srgb")
    bent = {"version": 1, "mode": "simple", "source": "only",
            "curve": match.fit_curve([3, 12, 30, 48, 70, 88, 97], [8, 20, 33, 55, 80, 93, 99.5]), "sat": 1.25,
            "tint": {"sh": [1.07, 0.93], "mid": [0.97, 1.04], "hl": [0.93, 1.07]}, "ref": ref, "frame": frame}
    y = (lin @ match.Y_W)[..., None]
    desat = match.fit(frame, match.stats(match.downsample(srgb8(y + 0.05 * (lin - y))), "srgb"))
    wash = match.fit(frame, match.stats(match.downsample(srgb8(0.05 + 0.85 * lin * [1.6, 0.8, 0.5])), "srgb"))
    edge = {"version": 3, "mode": "simple", "source": "film", "curve": [[1.0, 12.0], [30.0, 18.0], [60.0, 168.0 / 2.0], [99.0, 110.0]],
            "sat": 2.5, "tint": {"sh": [0.25, 4.0], "mid": [4.0, 0.25], "hl": [1.5, 0.6]}, "zone_L": [10.0, 50.0, 90.0], "ref": ref, "frame": frame}
    wide = dict(edge, tint={"sh": [1.0 / 64.0, 64.0], "mid": [64.0, 1.0 / 64.0], "hl": [40.0, 24.0]})
    return [match.fit(frame, ref), bent, match.fit(frame, frame), legacy_v2(frame, ref), desat, wash, edge, wide]


def write_png(path, rgb8):
    h, w, _ = rgb8.shape
    raw = b"".join(b"\x00" + rgb8[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    pathlib.Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                                   chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def cli_bake(state, s, p, tmp):
    tmp = pathlib.Path(tmp)
    (tmp / "state.json").write_text(json.dumps(state), encoding="utf-8")
    r = subprocess.run([str(CLI), "--match-bake", str(tmp / "state.json"), repr(s), repr(p), str(tmp / "m.bin")], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return np.fromfile(tmp / "m.bin", np.float32).reshape(33, 33, 33, 3)


def cli_render(img, luts, p, tmp, flags=0, mlut=None, reserved=0):
    tmp = pathlib.Path(tmp)
    blob = struct.pack(
        "<iifffiiiffIiiifffffffiii", SPACE_ID[p.in_space], SPACE_ID[p.out_space], p.exposure_ev, p.push_pull, p.impact,
        int(p.premultiplied), 0, reserved, p.grain_amount, p.grain_sigma_px, p.grain_seed, p.frame, p.origin_x, p.origin_y,
        p.halation_amount, p.halation_radius_px, p.bloom_amount, p.bloom_radius_px, *p.print_log_offset, 0, flags, 0)
    (tmp / "p.bin").write_bytes(blob)
    (tmp / "in.raw").write_bytes(np.ascontiguousarray(img, np.float32).tobytes())
    names = []
    for slot in ("under", "normal", "over"):
        (tmp / f"{slot}.bin").write_bytes(np.ascontiguousarray(luts[slot], np.float32).tobytes())
        names.append(str(tmp / f"{slot}.bin"))
    extra = []
    if mlut is not None:
        (tmp / "match.bin").write_bytes(np.ascontiguousarray(mlut, np.float32).tobytes())
        extra = ["--match", str(tmp / "match.bin"), str(mlut.shape[0])]
    r = subprocess.run([str(CLI), str(tmp / "p.bin"), str(tmp / "in.raw"), str(tmp / "out.raw"), str(img.shape[1]), str(img.shape[0]),
                        str(luts["normal"].shape[0]), *names, *extra], capture_output=True, text=True)
    return r, (np.frombuffer((tmp / "out.raw").read_bytes(), np.float32).reshape(img.shape) if r.returncode == 0 else None)


def host_run(img, tmp, in_space=0, out_space=0, extra=(), env=None):
    tmp = pathlib.Path(tmp)
    (tmp / "in.raw").write_bytes(np.ascontiguousarray(img, np.float32).tobytes())
    h, w = img.shape[:2]
    args = [str(HOST), str(PLUGIN), str(tmp / "in.raw"), str(tmp / "out.raw"), str(w), str(h), str(in_space), str(out_space),
            "0", "0", "1", "0", "0", "0", str(w), str(h), str(PROFILE_INDEX["synthetic-test"]), *extra]
    r = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", env=env)
    out = np.frombuffer((tmp / "out.raw").read_bytes(), np.float32).reshape(img.shape) if (tmp / "out.raw").exists() else None
    line = [l for l in r.stdout.splitlines() if l.startswith("match_state=")]
    return r, out, (line[-1][len("match_state="):] if line else None)


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


@unittest.skipUnless(CLI.exists(), "fe_cli not built")
class NativeLut(unittest.TestCase):
    def test_bake_parity(self):
        worst = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            for i, st in enumerate(states()):
                for s, p in ((0.7, 0.5), (1.0, 0.0), (1.0, 1.0), (0.3, 0.8)):
                    err = float(np.abs(cli_bake(st, s, p, tmp) - match.bake_lut(st, s, p)).max())
                    worst = max(worst, err)
                    self.assertLess(err, 1e-5, (i, s, p))
        print(f"\n[measure] C++ vs Python match LUT max |diff|: {worst:.2e}")


@unittest.skipUnless(CLI.exists(), "fe_cli not built")
class EngineMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.luts = make_luts()
        cls.mlut = match.bake_lut(states()[1], 0.8, 0.5)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_match_off_bit_identical(self):
        for i, p in enumerate((Params(), Params(in_space="acescct", out_space="rec709_g24", push_pull=0.6, impact=0.7),
                               Params(halation_amount=0.4, halation_radius_px=3.0, bloom_amount=0.2, bloom_radius_px=4.0))):
            img = make_image(p.in_space, 20 + i)
            _, base = cli_render(img, self.luts, p, self.tmp.name)
            r, loaded = cli_render(img, self.luts, p, self.tmp.name, flags=0, mlut=self.mlut, reserved=33)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(base.tobytes(), loaded.tobytes(), p)

    def test_match_on_film_and_bypass(self):
        img = make_image("dwg_di", 30)
        worst = 0.0
        for p in (Params(out_space="dwg_linear"), Params(out_space="dwg_linear", halation_amount=0.5, halation_radius_px=3.0)):
            film = render(img, self.luts, p)[..., :3]
            r, gpu = cli_render(img, self.luts, p, self.tmp.name, flags=1, mlut=self.mlut, reserved=33)
            self.assertEqual(r.returncode, 0, r.stderr)
            err = rel(gpu[..., :3], lut.apply(self.mlut, film, tetrahedral=True))
            worst = max(worst, err)
            self.assertLess(err, 1e-4, p)
            self.assertGreater(float(np.abs(gpu[..., :3] - film).max()), 1e-3)
        lin = spaces.sanitize(spaces.to_working(spaces.sanitize(img[..., :3]), "dwg_di"))
        r, gpu = cli_render(img, self.luts, Params(out_space="dwg_linear", grain_amount=1.0), self.tmp.name, flags=3, mlut=self.mlut,
                            reserved=33)
        self.assertEqual(r.returncode, 0, r.stderr)
        err = rel(gpu[..., :3], lut.apply(self.mlut, lin, tetrahedral=True))
        self.assertLess(err, 1e-4)
        r, gpu = cli_render(img, self.luts, Params(out_space="dwg_linear"), self.tmp.name, flags=2)
        self.assertLess(rel(gpu[..., :3], lin), 1e-5)
        print(f"\n[measure] CUDA match vs Python LUT apply: {max(worst, err):.2e}")

    def test_bad_match_setup_refused(self):
        img = make_image("dwg_di", 31)
        r, _ = cli_render(img, self.luts, Params(), self.tmp.name, flags=1, reserved=33)
        self.assertNotEqual(r.returncode, 0)
        r, _ = cli_render(img, self.luts, Params(), self.tmp.name, flags=1, mlut=self.mlut, reserved=17)
        self.assertNotEqual(r.returncode, 0)


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class PluginMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.luts = bundle_luts()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = pathlib.Path(cls.tmp.name)
        cls.state = states()[0]
        (cls.dir / "state.json").write_text(json.dumps(cls.state), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_render_with_state(self):
        img = make_image("dwg_di", 40)
        r, out, _ = host_run(img, self.tmp.name, out_space=1, extra=["--match-state", str(self.dir / "state.json"),
                                                                    "--match-strength", "0.7", "--match-protect", "0.5"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        film = render(img, self.luts, Params(out_space="dwg_linear"))[..., :3]
        err = rel(out[..., :3], lut.apply(match.bake_lut(self.state, 0.7, 0.5), film, tetrahedral=True))
        print(f"\n[measure] OFX match render vs Python: {err:.2e}")
        self.assertLess(err, 1e-4)

    def test_off_states_bit_identical(self):
        img = make_image("dwg_di", 41)
        _, base, _ = host_run(img, self.tmp.name)
        state = str(self.dir / "state.json")
        (self.dir / "spike.json").write_text("{\"spike\":1,\"ok\":1}", encoding="utf-8")
        for extra in (["--match-state", state, "--match-strength", "0"], ["--match-state", state, "--match-strength", "0.7", "--match-show", "1"],
                      ["--match-strength", "0.7"], ["--match-state", str(self.dir / "spike.json"), "--match-strength", "0.7"]):
            r, out, _ = host_run(img, self.tmp.name, extra=extra)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(out.tobytes(), base.tobytes(), extra)

    def test_create_grade_matches_python(self):
        lin = scene(200, 300, 7)
        img = np.concatenate([spaces.from_working(lin @ match.REC709_TO_DWG.T, "dwg_di"), np.ones((200, 300, 1))], -1).astype(np.float32)
        ref8 = np.round(srgb8(warm_sat(lin * 0.8)) * 255.0).astype(np.uint8)
        write_png(self.dir / "ref.png", ref8)
        ref = match.stats(match.downsample(ref8 / 255.0), "srgb")
        down = match.downsample(img).astype(np.float32)
        for src, flag in (("film", "0"), ("only", "1")):
            r, _, js = host_run(img, self.tmp.name, extra=["--match-source", flag, "--match-create", str(self.dir / "ref.png")])
            self.assertIn("match_create_status=0", r.stdout, r.stdout + r.stderr)
            got = json.loads(js)
            if src == "film":
                graded = render(down, self.luts, Params(out_space="rec709_g24"))[..., :3]
            else:
                graded = spaces.from_working(spaces.sanitize(spaces.to_working(down[..., :3], "dwg_di")), "rec709_g24")
            # Fit both implementations from identical GPU samples so analysis rounding
            # cannot choose different steps in the discrete refinement search.
            native_result, native_graded = cli_render(down, self.luts, Params(out_space="rec709_g24"),
                                                       self.tmp.name, flags=2 if src == "only" else 0)
            self.assertEqual(native_result.returncode, 0, native_result.stderr)
            np.testing.assert_allclose(native_graded[..., :3], graded, atol=2e-5)
            graded = native_graded[..., :3]
            want = match.refine(match.fit(match.stats(graded, "rec709"), ref, src), graded)
            self.assertEqual((got["version"], got["mode"], got["source"]), (3, "simple", src))
            self.assertEqual(len(got["curve"]), len(want["curve"]))
            np.testing.assert_allclose(got["zone_L"], want["zone_L"], atol=2e-3)
            np.testing.assert_allclose(got["ref"]["L"], want["ref"]["L"], atol=1e-9)
            np.testing.assert_allclose(got["ref"]["sat"], want["ref"]["sat"], atol=1e-9)
            np.testing.assert_allclose(got["frame"]["L"], want["frame"]["L"], atol=2e-3)
            np.testing.assert_allclose(got["curve"], want["curve"], atol=2e-3)
            np.testing.assert_allclose(got["sat"], want["sat"], atol=1e-4)
            # GPU analysis rounding can select adjacent colour-refinement steps.
            pixels = spaces.g24_decode(graded) @ match.REC709_TO_DWG.T
            actual_lab = match.lab_709(match.blended(got, 1.0, 0.5)(pixels) @ match.DWG_TO_709.T)
            expected_lab = match.lab_709(match.blended(want, 1.0, 0.5)(pixels) @ match.DWG_TO_709.T)
            self.assertLess(float(match.colour.delta_E(actual_lab, expected_lab, method="CIE 2000").mean()), 1.0)
            for k in ("sh", "mid", "hl"):
                np.testing.assert_allclose(got["tint"][k], want["tint"][k], atol=1e-4)
                for side in ("ref", "frame"):
                    self.assertEqual(got[side]["tint"][k] is None, want[side]["tint"][k] is None, (side, k))
            print(f"\n[measure] OFX create grade ({src}) curve max |diff| vs Python: "
                  f"{np.abs(np.array(got['curve']) - np.array(want['curve'])).max():.2e}")

    def test_create_grade_errors(self):
        img = make_image("dwg_di", 42)
        r, _, js = host_run(img, self.tmp.name, extra=["--match-create", str(self.dir / "missing.png")])
        self.assertIn("match_create_status=1", r.stdout)
        self.assertIn("cannot read the reference image", r.stdout)
        self.assertEqual(js, "")
        r, _, _ = host_run(img, self.tmp.name, extra=["--look-save", "x"])
        self.assertIn("create a grade first", r.stdout)

    def test_save_and_load_look(self):
        img = make_image("dwg_di", 43)
        with tempfile.TemporaryDirectory() as appdata:
            env = dict(os.environ, APPDATA=appdata)
            r, _, _ = host_run(img, self.tmp.name, env=env, extra=["--match-state", str(self.dir / "state.json"), "--match-strength", "0.7",
                                                                  "--match-protect", "0.5", "--look-save", "warm look"])
            self.assertIn("look_save_status=0", r.stdout, r.stdout + r.stderr)
            looks = pathlib.Path(appdata) / "FIXER" / "looks"
            self.assertEqual((looks / "warm look.json").read_text(encoding="utf-8"), (self.dir / "state.json").read_text(encoding="utf-8"))
            match.write_cube(self.dir / "py.cube", self.state, 0.7, 0.5)
            got = (looks / "warm look.cube").read_text(encoding="ascii").split("\n")
            want = (self.dir / "py.cube").read_text(encoding="ascii").split("\n")
            self.assertEqual(got[:5], want[:5])
            self.assertEqual(len(got), len(want))
            rows = lambda lines: np.array([[float(v) for v in l.split()] for l in lines[5:] if l])
            err = float(np.abs(rows(got) - rows(want)).max())
            print(f"\n[measure] .cube C++ vs Python max |diff|: {err:.2e}")
            self.assertLessEqual(err, 1.5e-6)
            only = dict(states()[1])
            (looks / "only.json").write_text(json.dumps(only), encoding="utf-8")
            r, _, js = host_run(img, self.tmp.name, env=env, extra=["--look-load", str(looks / "only.json")])
            self.assertIn("look_load_status=0", r.stdout, r.stdout + r.stderr)
            self.assertIn("match_source=1", r.stdout)
            self.assertEqual(json.loads(js)["curve"], only["curve"])
            (looks / "bad.json").write_text("{\"spike\":1}", encoding="utf-8")
            r, _, js = host_run(img, self.tmp.name, env=env, extra=["--look-load", str(looks / "bad.json")])
            self.assertIn("cannot read the look", r.stdout)
            self.assertEqual(js, "")

    def test_turkish_look_name_and_folder(self):
        img = make_image("dwg_di", 44)
        name = "Işık çğüşöİ"
        with tempfile.TemporaryDirectory() as tmp:
            appdata = pathlib.Path(tmp) / "Kullanıcı Şeyma Ğİ"
            appdata.mkdir()
            env = dict(os.environ, APPDATA=str(appdata))
            r, _, _ = host_run(img, self.tmp.name, env=env, extra=["--match-state", str(self.dir / "state.json"), "--match-strength", "0.7",
                                                                  "--look-save", name])
            self.assertIn("look_save_status=0", r.stdout, r.stdout + r.stderr)
            looks = appdata / "FIXER" / "looks"
            self.assertEqual(sorted(p.name for p in looks.iterdir()), [f"{name}.cube", f"{name}.json"])
            r, _, js = host_run(img, self.tmp.name, env=env, extra=["--look-load", str(looks / f"{name}.json")])
            self.assertIn("look_load_status=0", r.stdout, r.stdout + r.stderr)
            self.assertEqual(json.loads(js)["curve"], self.state["curve"])
            self.assertIn("labels=Load look;", r.stdout)
            r, _, js = host_run(img, self.tmp.name, env=env, extra=["--look-pick", "1"])
            self.assertIn("look_option 0 (none)", r.stdout)
            self.assertIn(f"look_option 1 {name}", r.stdout)
            self.assertIn("look_pick_status=0", r.stdout, r.stdout + r.stderr)
            self.assertEqual(json.loads(js)["curve"], self.state["curve"])

    def test_invalid_state_values_rejected(self):
        img = make_image("dwg_di", 45)
        good = dict(self.state)
        cases = {"saturation": dict(good, sat=2.6), "tint": dict(good, tint=dict(good["tint"], mid=[65.0, 1.0])),
                 "a tint": dict(good, tint=dict(good["tint"], mid=[1.0 / 65.0, 1.0])),
                 "strictly increasing": dict(good, curve=[[50.0, 50.0], [40.0, 55.0]]),
                 "slope": dict(good, curve=[[10.0, 10.0], [20.0, 80.0]]), "zone_L": dict(good, zone_L=[60.0, 50.0, 80.0]),
                 "y is outside": dict(good, curve=[[1.0, -0.4], [50.0, 50.0]]), "2 to 9 points": dict(good, curve=[[50.0, 50.0]]),
                 "unknown version": dict(good, version=4),
                 "legacy saturation": dict(states()[3], sat=2.0)}
        _, base, _ = host_run(img, self.tmp.name)
        for why, st in cases.items():
            path = self.dir / "invalid.json"
            path.write_text(json.dumps(st), encoding="utf-8")
            r, _, js = host_run(img, self.tmp.name, extra=["--look-load", str(path)])
            self.assertIn("cannot read the look", r.stdout, why)
            self.assertIn(why.replace("legacy saturation", "saturation is outside [0.85, 1.25]"), r.stdout)
            r, out, _ = host_run(img, self.tmp.name, extra=["--match-state", str(path), "--match-strength", "0.7"])
            self.assertIn("stored reference match is invalid", r.stdout, why)
            self.assertEqual(out.tobytes(), base.tobytes(), why)

    def test_order_follows_state_and_bypass_only_when_active(self):
        img = make_image("dwg_di", 46)
        only = self.dir / "only.json"
        only.write_text(json.dumps(states()[1]), encoding="utf-8")
        _, base, _ = host_run(img, self.tmp.name)
        r1, as_only, _ = host_run(img, self.tmp.name, extra=["--match-state", str(only), "--match-strength", "0.8", "--match-source", "1"])
        self.assertNotIn("Order changed", r1.stdout)
        r0, as_film, _ = host_run(img, self.tmp.name, extra=["--match-state", str(only), "--match-strength", "0.8", "--match-source", "0"])
        self.assertEqual(r0.stdout.count("Order changed: press Create grade again"), 1, r0.stdout)
        self.assertEqual(as_film.tobytes(), as_only.tobytes())
        for extra in (["--match-strength", "0"], ["--match-strength", "0.8", "--match-show", "1"]):
            r, out, _ = host_run(img, self.tmp.name, extra=["--match-state", str(only), "--match-source", "1", *extra])
            self.assertEqual(out.tobytes(), base.tobytes(), extra)

    def test_create_grade_frame_and_reference_checks(self):
        lin = scene(120, 200, 8)
        img = np.concatenate([spaces.from_working(lin @ match.REC709_TO_DWG.T, "dwg_di"), np.ones((120, 200, 1))], -1).astype(np.float32)
        ref8 = np.round(srgb8(warm_sat(scene(1500, 2000, 9))) * 255.0).astype(np.uint8)
        write_png(self.dir / "big.png", ref8)
        r, _, js = host_run(img, self.tmp.name, extra=["--match-create-device", "1", "--match-create", str(self.dir / "big.png")])
        self.assertIn("match_create_status=1", r.stdout)
        self.assertIn("in GPU memory", r.stdout)
        bad = img.copy()
        bad[::9, ::7, :3] = np.nan
        bad[4::11, ::5, 1] = np.inf
        r, _, js = host_run(bad, self.tmp.name, extra=["--match-create", str(self.dir / "big.png")])
        self.assertIn("match_create_status=0", r.stdout, r.stdout + r.stderr)
        got = json.loads(js)
        self.assertTrue(np.isfinite(got["frame"]["L"]).all() and np.isfinite(got["curve"]).all())
        want = match.stats(match.downsample(ref8 / 255.0), "srgb")
        print(f"\n[measure] 2000x1500 reference scaled in WIC: ref L* max |diff| vs Python full size "
              f"{np.abs(np.array(got['ref']['L']) - want['L']).max():.3f}")
        np.testing.assert_allclose(got["ref"]["L"], want["L"], atol=1.0)
        write_png(self.dir / "tall.png", np.full((30001, 1, 3), 128, np.uint8))
        r, _, js = host_run(img, self.tmp.name, extra=["--match-create", str(self.dir / "tall.png")])
        self.assertIn("match_create_status=1", r.stdout)
        self.assertIn("too large", r.stdout)

    def test_progress_paths(self):
        lin = scene(60, 90, 10)
        img = np.concatenate([spaces.from_working(lin @ match.REC709_TO_DWG.T, "dwg_di"), np.ones((60, 90, 1))], -1).astype(np.float32)
        write_png(self.dir / "small.png", np.round(srgb8(warm_sat(lin)) * 255.0).astype(np.uint8))
        r, _, _ = host_run(img, self.tmp.name, extra=["--match-create", str(self.dir / "small.png")])
        self.assertIn("match_create_status=0", r.stdout, r.stdout + r.stderr)
        self.assertIn("progress_path=none", r.stdout)
        if os.environ.get("FIXER_PROGRESS_TEST") == "1":
            env = {k: v for k, v in os.environ.items() if k != "FIXER_NO_PROGRESS"}
            r, _, _ = host_run(img, self.tmp.name, env=env, extra=["--match-create", str(self.dir / "small.png")])
            self.assertIn("match_create_status=0", r.stdout, r.stdout + r.stderr)
            line = [l for l in r.stdout.splitlines() if l.startswith("progress_path=")]
            print(f"\n[measure] without host progress suite: {line}")
            self.assertTrue(line and line[0].startswith("progress_path=own_window windows=1"), r.stdout)
            self.assertGreaterEqual(int(line[0].split("shown_ms=")[1]), 450)
            self.assertIn("labels=Create grade;", r.stdout)
        r, _, _ = host_run(img, self.tmp.name, extra=["--progress-suite", "1", "--match-create", str(self.dir / "small.png")])
        self.assertIn("match_create_status=0", r.stdout, r.stdout + r.stderr)
        line = [l for l in r.stdout.splitlines() if l.startswith("progress_path=")]
        print(f"[measure] with host progress suite: {line}")
        self.assertTrue(line and line[0].startswith("progress_path=host_suite starts=1"), r.stdout)
        self.assertIn("ends=1", line[0])
        r, _, _ = host_run(img, self.tmp.name, extra=["--match-create", str(self.dir / "missing.png")])
        self.assertIn("progress_path=none", r.stdout)
        self.assertIn("cannot read the reference image", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
