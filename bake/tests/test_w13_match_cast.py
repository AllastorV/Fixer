"""Regressions for strong casts and flat-patch reference quantization."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

from filmemu import match, spaces
from tests.test_w11_match import srgb8


def chart():
    out = np.zeros((1080, 1920, 3))
    colours = np.array([[.65,.04,.04], [.65,.65,.04], [.04,.65,.04],
                        [.04,.65,.65], [.04,.04,.65], [.65,.04,.65]])
    for row, level in enumerate((1.0, .3, .08)):
        for col, rgb in enumerate(colours):
            out[row*200:(row+1)*200, col*320:(col+1)*320] = rgb * level
    for col, lab in enumerate(((35,12,18), (50,16,24), (65,14,20), (80,10,15))):
        rgb = match.colour.XYZ_to_RGB(match.colour.Lab_to_XYZ(lab, match.D65), match.REC709)
        out[600:800, col*320:(col+1)*320] = np.clip(rgb, 0, 1)
    out[600:800, 1280:1600] = .18
    out[600:800, 1600:] = .9
    for col, stop in enumerate(range(-8, 4)):
        out[800:920, col*160:(col+1)*160] = .18 * 2.0**stop
    out[920:] = np.linspace(0, 1, 1920)[None, :, None]
    return out


class JointRefinement(unittest.TestCase):
    def test_strong_cast_does_not_amplify_source_saturation(self):
        rng = np.random.default_rng(29)
        lin = rng.uniform(.01, 1, (64, 128, 3)) * np.exp(rng.uniform(np.log(.01), np.log(.5), (64, 128, 1)))
        encoded = spaces.g24_encode(lin)
        frame = match.stats(encoded)
        y = (lin @ match.Y_W)[..., None]
        for name, gains in (("red", (12, .1)), ("magenta", (40, 24))):
            with self.subTest(cast=name):
                target = (y + .15 * (lin - y)) * match.gain_vec(*gains)
                ref = match.stats(srgb8(target), "srgb")
                fitted = match.refine(match.fit(frame, ref, "only"), encoded)
                out = match.transform(fitted)(lin @ match.REC709_TO_DWG.T) @ match.DWG_TO_709.T
                error = match.colour.delta_E(match.lab_709(np.clip(out, 0, 1)), match.lab_709(target), method="CIE 2000")
                self.assertLess(float(error.mean()), 2.5)
                self.assertLess(fitted["sat"], .5)
                self.assertTrue(match.grey_monotone(fitted))

    def test_quantized_self_reference_is_identity(self):
        linear = match.downsample(chart())
        encoded, reference = spaces.g24_encode(linear), srgb8(linear)
        fitted = match.refine(match.fit(match.stats(encoded), match.stats(reference, "srgb")), encoded, reference)
        dwg = linear @ match.REC709_TO_DWG.T
        np.testing.assert_array_equal(match.transform(fitted)(dwg), dwg)

    def test_different_reference_is_not_identity(self):
        linear = match.downsample(chart())
        encoded, reference = spaces.g24_encode(linear), srgb8(linear * [.7, 1, 1.1])
        fitted = match.refine(match.fit(match.stats(encoded), match.stats(reference, "srgb")), encoded, reference)
        self.assertNotEqual(fitted["tint"]["mid"], [1.0, 1.0])


class NativeChart(unittest.TestCase):
    def test_film_and_only_quantized_self_reference(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
        import match_visual as mv
        if not mv.HOST.exists() or not mv.PLUGIN.exists():
            self.skipTest("native plugin/host not built")
        image = mv.encode_di(chart())
        with tempfile.TemporaryDirectory() as tmp:
            for mode in mv.SOURCES:
                with self.subTest(mode=mode):
                    before = mv.before_film(image, tmp) if mode == "film" else mv.before_only(image)
                    ref = mv.same_ref(before, tmp, "chart", mode)
                    result = mv.create(image, ref, mode, tmp)
                    self.assertLess(mv.de00(result["out"], before), .05)
                    self.assertEqual(result["state"]["curve"], [[0.0, 0.0], [100.0, 100.0]])


if __name__ == "__main__":
    unittest.main()
