import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore")

from filmemu import spaces
from filmemu.pipeline import Params, render
from tests.test_w03_cuda import CLI, run_cuda

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE = ROOT / "build" / "release" / "Fixer.ofx.bundle" / "Contents" / "Resources" / "profiles" / "synthetic-spectral"


def load_profile():
    meta = json.loads((PROFILE / "profile.json").read_text(encoding="utf-8"))
    n = meta["lut_size"]
    luts = {s: np.fromfile(PROFILE / f"{s}.bin", np.float32).reshape(n, n, n, 3) for s in ("under", "normal", "over", "print")}
    luts["print_domain"] = (meta["print_lo"], meta["print_hi"])
    return meta, luts


def px(lin):
    lin = np.atleast_2d(np.asarray(lin, np.float64))
    return np.concatenate([spaces.di_encode(lin), np.ones((len(lin), 1))], -1).astype(np.float32)


@unittest.skipUnless((PROFILE / "profile.json").exists(), "spectral profile not baked")
class SpectralChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from filmemu import spectral

        cls.spectral = spectral
        cls.chain = spectral.Chain(spectral.synthetic_system(1.0))
        cls.meta, cls.luts = load_profile()

    def test_profile_metadata(self):
        self.assertEqual(self.meta["kind"], "two_stage")
        self.assertIn("synthetic", self.meta["data_source"])

    def test_mid_grey_is_neutral(self):
        out = self.chain.display_linear(self.chain.print_log_exposure(np.full((1, 3), 0.18)))[0]
        print(f"\n[measure] direct chain mid-grey: {out.round(5)}")
        self.assertLess(np.abs(out / out.mean() - 1.0).max(), 2e-3)

    def test_neutral_ramp_monotonic(self):
        lin = np.geomspace(1e-3, 30, 200)[:, None].repeat(3, 1)
        y = self.chain.display_linear(self.chain.print_log_exposure(lin))
        self.assertTrue(np.all(np.diff(y, axis=0) > 0))

    def test_saturated_colour_keeps_hue_order(self):
        red = self.chain.display_linear(self.chain.print_log_exposure(np.array([[0.6, 0.05, 0.05]])))[0]
        self.assertGreater(red[0], red[1])
        self.assertGreater(red[0], red[2])

    def test_lut_vs_direct_chain(self):
        rng = np.random.default_rng(0)
        lin = spaces.di_decode(rng.uniform(0.25, 0.75, (400, 3)))
        direct = spaces.di_encode(self.chain.display_linear(self.chain.print_log_exposure(lin)))
        via = render(px(lin), self.luts, Params())[:, :3]
        err = np.abs(direct - via)
        print(f"\n[measure] two-stage 65^3 LUT vs direct chain, DI units: max={err.max():.2e} p99={np.quantile(err, 0.99):.2e}")
        self.assertLess(err.max(), 5e-3)

    def test_printer_points(self):
        grey = px([[0.18, 0.18, 0.18]])
        base = spaces.di_decode(render(grey, self.luts, Params())[0, :3])
        red = spaces.di_decode(render(grey, self.luts, Params(print_log_offset=(0.025 * 6, 0.0, 0.0)))[0, :3])
        dark = spaces.di_decode(render(grey, self.luts, Params(print_log_offset=(0.025 * 6,) * 3))[0, :3])
        self.assertLess(red[0], base[0])
        self.assertLess(abs(red[1] - base[1]) / base[1], 0.15)
        self.assertTrue(np.all(dark < base))

    def test_push_pull_zero_is_normal_slot(self):
        img = px(np.random.default_rng(1).uniform(0.01, 2, (64, 3)))
        a = render(img, self.luts, Params(push_pull=0.0))
        same = dict(self.luts, under=self.luts["normal"], over=self.luts["normal"])
        b = render(img, same, Params(push_pull=1.4))
        self.assertLess(np.abs(a - b).max(), 1e-6)

    @unittest.skipUnless(CLI.exists(), "fe_cli not built")
    def test_cuda_parity(self):
        rng = np.random.default_rng(2)
        img = px(spaces.di_decode(rng.uniform(0.0, 1.0, (37 * 23, 3)))).reshape(23, 37, 4)
        with tempfile.TemporaryDirectory() as d:
            for p in (Params(), Params(push_pull=0.8, print_log_offset=(0.1, -0.05, 0.02), out_space="rec709_g24"),
                      Params(push_pull=-1.5, exposure_ev=1.0, grain_amount=1.0, halation_amount=0.4, halation_radius_px=3.0)):
                gpu = run_cuda(img, self.luts, p, d)
                ora = render(img, self.luts, p)
                err = np.max(np.abs(gpu - ora) / np.maximum(1.0, np.abs(ora)))
                print(f"\n[measure] two-stage CUDA vs oracle: {err:.2e}")
                self.assertLess(err, 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
