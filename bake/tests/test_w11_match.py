import pathlib
import tempfile
import unittest
import warnings

import colour
import numpy as np

warnings.simplefilter("ignore")

from filmemu import match, spaces

AXIS = np.linspace(0.0, 1.0, 33)


def scene(h=128, w=512, seed=1):
    rng = np.random.default_rng(seed)
    y = np.exp(rng.uniform(np.log(0.003), np.log(0.85), (h, w)))
    hue = rng.uniform(0.0, 2.0 * np.pi, (h, w))
    chroma = np.where(rng.random((h, w)) < 0.6, rng.uniform(0.0, 0.02, (h, w)), rng.uniform(0.1, 0.5, (h, w)))
    d = np.stack([np.cos(hue), np.cos(hue - 2.1), np.cos(hue + 2.1)], -1)
    rgb = y[..., None] * (1.0 + chroma[..., None] * d)
    return np.maximum(rgb / (rgb @ match.Y_W)[..., None] * y[..., None], 0.0)


def warm_sat(lin):
    lin = lin * [1.04, 1.0, 0.96]
    y = (lin @ match.Y_W)[..., None]
    return np.maximum(y + 1.15 * (lin - y), 0.0)


def srgb8(lin):
    return np.round(np.clip(colour.models.eotf_inverse_sRGB(np.clip(lin, 0.0, 1.0)), 0.0, 1.0) * 255.0) / 255.0


def identity_grid():
    b, g, r = np.meshgrid(AXIS, AXIS, AXIS, indexing="ij")
    return np.stack([r, g, b], -1)


def lab_to_dwg(lab):
    lin = colour.XYZ_to_RGB(colour.Lab_to_XYZ(np.asarray(lab, np.float64), match.D65), match.REC709, apply_cctf_encoding=False)
    return lin @ match.REC709_TO_DWG.T


def frame_stats(lin):
    return match.stats(match.downsample(spaces.g24_encode(np.clip(lin, 0.0, 1.0))), "rec709")


def graded(state, lin, s=1.0):
    out = match.transform(state)(lin @ match.REC709_TO_DWG.T) @ match.DWG_TO_709.T
    return frame_stats(lin + s * (out - lin))


def legacy_v2(frame, ref):
    curve = match.fit_curve(frame["L"][1:8], ref["L"][1:8])
    return {"version": 2, "mode": "simple", "source": "film", "curve": curve, "sat": 1.1,
            "tint": {"sh": [1.03, 0.97], "mid": [1.02, 0.98], "hl": [0.98, 1.03]}, "zone_L": [12.0, 45.0, 85.0], "ref": ref, "frame": frame}


def slopes(curve):
    c = np.array(curve)
    return np.diff(c[:, 1]) / np.diff(c[:, 0])


class Downsample(unittest.TestCase):
    def test_area_average(self):
        img = np.arange(4 * 512 * 3, dtype=np.float64).reshape(4, 512, 3)
        out = match.downsample(img)
        self.assertEqual(out.shape, (2, 256, 3))
        np.testing.assert_allclose(out[0, 0], img[:2, :2].mean((0, 1)))
        self.assertAlmostEqual(out.mean(), img.mean(), places=6)
        self.assertEqual(match.downsample(np.zeros((1080, 1920, 3))).shape, (144, 256, 3))


class Fit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lin = scene()
        cls.frame = frame_stats(cls.lin)

    def test_identity_fit(self):
        st = match.fit(self.frame, self.frame)
        c = np.array(st["curve"])
        self.assertTrue(2 <= len(c) <= 9)
        self.assertLess(np.abs(c[:, 1] - c[:, 0]).max(), 0.5)
        self.assertLess(abs(st["sat"] - 1.0), 0.01)
        for k in ("sh", "mid", "hl"):
            self.assertLess(np.abs(np.array(st["tint"][k]) - 1.0).max(), 0.01, k)
        self.assertEqual((st["version"], st["mode"], st["source"]), (3, "simple", "film"))
        zl = st["zone_L"]
        self.assertTrue(0.0 <= zl[0] < zl[1] < zl[2] <= 100.0, zl)

    def test_identity_transform_near_identity(self):
        st = match.fit(self.frame, self.frame)
        g = match.lut.di_decode(identity_grid())
        L = match.L_of_Y(np.maximum(g @ match.DWG_TO_709.T @ match.Y_W, 0.0))
        inside = L < st["curve"][-1][0]
        np.testing.assert_allclose(match.transform(st)(g)[inside], g[inside], atol=2e-3, rtol=1e-2)

    def test_known_warm_saturated_reference(self):
        ref = match.stats(match.downsample(srgb8(warm_sat(self.lin))), "srgb")
        st = match.fit(self.frame, ref)
        self.assertGreater(st["sat"], 1.03)
        self.assertLessEqual(st["sat"], match.SAT3_MAX)
        for k in ("sh", "mid", "hl"):
            r, b = st["tint"][k]
            self.assertGreater(r, 1.0, k)
            self.assertLess(b, 1.0, k)
        s = slopes(st["curve"])
        self.assertTrue(np.all(s >= match.SLOPE3_MIN - 1e-9) and np.all(s <= match.SLOPE3_MAX + 1e-9))
        after = graded(st, self.lin)
        for k in ("sh", "mid", "hl"):
            before_err = np.hypot(*(np.array(self.frame["tint"][k][1:]) - ref["tint"][k][1:]))
            self.assertLess(np.hypot(*(np.array(after["tint"][k][1:]) - ref["tint"][k][1:])), 0.5 * before_err, k)
        self.assertLess(abs(after["sat"] - ref["sat"]), 0.5 * abs(self.frame["sat"] - ref["sat"]))

    def test_desaturated_reference_removes_most_chroma(self):
        y = (self.lin @ match.Y_W)[..., None]
        ref = match.stats(match.downsample(srgb8(y + 0.1 * (self.lin - y))), "srgb")
        st = match.fit(self.frame, ref)
        self.assertLess(st["sat"], 0.2)
        self.assertGreaterEqual(st["sat"], match.SAT3_MIN)
        after = graded(st, self.lin)
        print(f"\n[measure] desaturated reference: median chroma ratio after/before {after['sat'] / self.frame['sat']:.3f}, "
              f"reference/before {ref['sat'] / self.frame['sat']:.3f}")
        self.assertLess(after["sat"] / self.frame["sat"], 0.4)

    def test_strong_cast_transfers(self):
        y = (self.lin @ match.Y_W)[..., None]
        for name, gain in (("red wash", [1.6, 0.85, 0.75]), ("green", [0.75, 1.15, 0.7]), ("blue", [0.75, 0.95, 1.6])):
            ref = match.stats(match.downsample(srgb8(self.lin * gain / (np.array(gain) @ match.Y_W))), "srgb")
            after = graded(match.fit(self.frame, ref), self.lin)
            for k in ("sh", "mid", "hl"):
                f, r, o = (np.array(s["tint"][k][1:]) for s in (self.frame, ref, after))
                closed = 1.0 - np.linalg.norm(o - r) / np.linalg.norm(r - f)
                with self.subTest(cast=name, zone=k):
                    self.assertGreater(np.linalg.norm(r - f), 5.0)
                    self.assertGreater(closed, 0.6, (name, k, f, r, o))

    def test_lifted_and_crushed_blacks(self):
        lifted = match.stats(match.downsample(srgb8(0.04 + 0.9 * self.lin)), "srgb")
        st = match.fit(self.frame, lifted)
        self.assertGreater(float(match.curve_eval3(st["curve"], 0.0)), 10.0)
        after = graded(st, self.lin)
        self.assertLess(abs(after["L"][1] - lifted["L"][1]), 0.3 * abs(self.frame["L"][1] - lifted["L"][1]))
        crushed = match.stats(match.downsample(srgb8(np.maximum(self.lin * 1.2 - 0.02, 0.0))), "srgb")
        st = match.fit(self.frame, crushed)
        self.assertEqual(float(match.curve_eval3(st["curve"], 2.0)), 0.0)

    def test_curve_bounds_merge_and_ends(self):
        c = match.fit_curve3([2, 3, 4, 5, 6, 7, 8, 9, 10], [40, 60, 80, 90, 95, 97, 99, 99.5, 100])
        s = slopes(c)
        self.assertTrue(np.all(s >= match.SLOPE3_MIN - 1e-9) and np.all(s <= match.SLOPE3_MAX + 1e-9), s)
        clip = match.fit_curve3([5, 20, 40, 60, 100, 100, 100, 100, 100], [5, 20, 40, 60, 70, 80, 90, 95, 99])
        self.assertEqual(len(clip), 5)
        self.assertAlmostEqual(clip[-1][1], 86.8)
        self.assertAlmostEqual(float(match.curve_eval3(clip, 110.0)), 86.8 + 10.0 * match.SLOPE3_MIN)
        flat = match.fit_curve3([50] * 9, [50] * 9)
        self.assertEqual(len(flat), 2)
        self.assertTrue(np.all(np.diff(np.array(flat)[:, 0]) > 0))
        Ls = np.linspace(-5.0, 150.0, 2000)
        for curve in (c, clip, flat):
            self.assertTrue(np.all(np.diff(match.curve_eval3(curve, Ls)) >= 0.0))
            self.assertGreaterEqual(float(match.curve_eval3(curve, -5.0)), 0.0)

    def test_low_sample_cases_identity(self):
        dark = np.full((64, 256, 3), 0.0)
        dark_st = match.stats(spaces.g24_encode(dark), "rec709")
        self.assertIsNone(dark_st["sat"])
        self.assertEqual(dark_st["tint"], {"sh": None, "mid": None, "hl": None})
        st = match.fit(dark_st, self.frame)
        self.assertEqual(st["sat"], 1.0)
        self.assertEqual(st["tint"], {"sh": [1.0, 1.0], "mid": [1.0, 1.0], "hl": [1.0, 1.0]})
        self.assertTrue(0.0 <= st["zone_L"][0] < st["zone_L"][1] < st["zone_L"][2] <= 100.0, st["zone_L"])


class SunriseCast(unittest.TestCase):
    def test_warm_scene_with_blue_neutrals_does_not_go_blue(self):
        rng = np.random.default_rng(5)
        n = 64 * 256
        y = np.exp(rng.uniform(np.log(0.02), np.log(0.8), n))
        sky = rng.random(n) < 0.35
        warm = np.array([1.5, 0.95, 0.45])
        blue = np.array([0.93, 1.0, 1.12])
        rgb = np.where(sky[:, None], blue, warm) * (y / np.where(sky, blue @ match.Y_W, warm @ match.Y_W))[:, None]
        ref = match.stats(srgb8(rgb.reshape(64, 256, 3)), "srgb")
        lin = scene()
        neutral = (lin @ match.Y_W)[..., None] * np.ones(3)
        fr = frame_stats(neutral)
        after = graded(match.fit(fr, ref), neutral)
        for k in ("sh", "mid", "hl"):
            self.assertGreater(after["tint"][k][2], fr["tint"][k][2] + 3.0, k)


class Legacy(unittest.TestCase):
    def test_version1_and_2_keep_old_weights(self):
        frame = frame_stats(scene())
        ref = match.stats(match.downsample(srgb8(warm_sat(scene()))), "srgb")
        v2 = legacy_v2(frame, ref)
        v1 = dict(v2, version=1)
        del v1["zone_L"]
        g = match.lut.di_decode(identity_grid())
        self.assertTrue(np.isfinite(match.transform(v1)(g)).all())
        self.assertGreater(float(np.abs(match.transform(v1)(g) - match.transform(v2)(g)).max()), 1e-4)
        self.assertGreater(float(np.abs(match.transform(v2)(g) - match.transform(dict(v2, version=3))(g)).max()), 1e-3)


class NonFinite(unittest.TestCase):
    def test_nan_inf_become_zero(self):
        img = spaces.g24_encode(np.clip(scene(64, 512), 0.0, 1.0))
        bad = img.copy()
        bad[::7, ::5] = np.nan
        bad[3::11, ::3, 1] = np.inf
        bad[5::13, ::4, 2] = -np.inf
        zero = np.where(np.isfinite(bad), bad, 0.0)
        np.testing.assert_array_equal(match.downsample(bad), match.downsample(zero))
        st = match.stats(bad, "rec709")
        self.assertEqual(st, match.stats(zero, "rec709"))
        self.assertTrue(np.isfinite(st["L"]).all())


class Skin(unittest.TestCase):
    def test_window(self):
        inside = lab_to_dwg([[60.0, 20.0, 25.0], [50.0, 15.0, 30.0], [70.0, 25.0, 15.0]])
        np.testing.assert_allclose(match.skin(inside), 1.0, atol=1e-9)
        outside = lab_to_dwg([[60.0, -30.0, -30.0], [60.0, 0.0, 0.0], [60.0, 10.0, -40.0], [5.0, 20.0, 25.0], [98.0, 5.0, 6.0],
                              [60.0, 60.0, 70.0]])
        np.testing.assert_allclose(match.skin(outside), 0.0, atol=1e-9)
        edge = float(match.skin(lab_to_dwg([[60.0, 25.0, 7.0]])[0]))
        self.assertTrue(0.0 < edge < 1.0)


class Lut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lin = scene()
        frame = frame_stats(lin)
        ref = match.stats(match.downsample(srgb8(warm_sat(lin * 0.7))), "srgb")
        cls.state = match.fit(frame, ref)
        y = (lin @ match.Y_W)[..., None]
        cls.extreme = [match.fit(frame, match.stats(match.downsample(srgb8(r)), "srgb")) for r in
                       (y + 0.05 * (lin - y), 0.06 + 0.8 * lin * [1.6, 0.8, 0.5], np.minimum(lin * 3.0, 1.0) * [0.6, 1.2, 0.7])]

    def test_tetrahedral_preserves_neutral_ramp(self):
        table = np.ones((3, 3, 3, 3), np.float32)
        axis = np.linspace(0.0, 1.0, 3)
        table[np.arange(3), np.arange(3), np.arange(3)] = axis[:, None]
        ramp = np.repeat(np.linspace(0.0, 1.0, 4097)[:, None], 3, axis=1)
        output = match.lut.sample_tetrahedral(table, ramp)
        np.testing.assert_allclose(output, ramp, atol=1e-12)
        self.assertTrue(np.all(np.diff(output, axis=0) >= 0.0))

    def test_strength_zero_is_identity(self):
        t = match.bake_lut(self.state, 0.0, 0.5)
        self.assertEqual((t.shape, t.dtype), ((33, 33, 33, 3), np.float32))
        self.assertLess(np.abs(t - identity_grid()).max(), 1e-6)

    def test_finite_and_grey_monotone(self):
        for st in (self.state, *self.extreme):
            for s, p in ((0.7, 0.5), (1.0, 0.0), (1.0, 1.0)):
                t = match.bake_lut(st, s, p)
                self.assertTrue(np.isfinite(t).all())
                grey = t[np.arange(33), np.arange(33), np.arange(33)]
                self.assertTrue(np.all(np.diff(grey, axis=0) >= -1e-7), (s, p))
                self.assertGreater(np.abs(t - identity_grid()).max(), 1e-3)

    def test_protect_scales_skin(self):
        c = lab_to_dwg([[60.0, 20.0, 25.0]])
        m = match.transform(self.state)(c)
        full = match.blended(self.state, 1.0, 1.0)(c)
        np.testing.assert_allclose(full, c, atol=1e-12)
        half = match.blended(self.state, 1.0, 0.5)(c)
        np.testing.assert_allclose(half, c + 0.5 * (m - c), atol=1e-12)

    def test_cube_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "look.cube"
            match.write_cube(path, self.state, 0.7, 0.5)
            lines = path.read_text(encoding="ascii").splitlines()
        self.assertIn(match.CUBE_HEADER, "\n".join(lines))
        size = [int(l.split()[1]) for l in lines if l.startswith("LUT_3D_SIZE")]
        self.assertEqual(size, [33])
        rows = np.array([[float(v) for v in l.split()] for l in lines if l and (l[0].isdigit() or l[0] == "-")])
        self.assertEqual(rows.shape, (35937, 3))
        self.assertTrue(np.isfinite(rows).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "id.cube"
            match.write_cube(path, self.state, 0.0, 0.5)
            rows0 = np.array([[float(v) for v in l.split()] for l in path.read_text(encoding="ascii").splitlines()
                              if l and (l[0].isdigit() or l[0] == "-")])
        np.testing.assert_allclose(rows0, identity_grid().reshape(-1, 3), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
