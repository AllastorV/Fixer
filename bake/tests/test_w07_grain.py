import tempfile
import unittest
import warnings
from dataclasses import replace

import numpy as np

warnings.simplefilter("ignore")

from filmemu import grain, spaces
from filmemu.pipeline import Params, render
from tests.test_w03_cuda import CLI, make_luts, run_cuda


def grey(h, w, level=0.18):
    v = spaces.di_encode(level)
    img = np.full((h, w, 4), v, np.float32)
    img[..., 3] = 1.0
    return img


class GrainOracle(unittest.TestCase):
    def test_unit_variance_and_correlation(self):
        for s in (0.5, 1.0, 3.0):
            n = grain.noise_field(11, 0, 0, 0, 256, 256, s)
            self.assertLess(abs(n.std() - 1.0), 0.05, s)
            c = np.corrcoef(n.reshape(-1, 3).T)
            self.assertLess(abs(c[0, 1]), 0.1, s)

    def test_tile_consistency_and_temporal(self):
        a = grain.noise_field(3, 5, -20, 40, 64, 48, 1.5)
        b = grain.noise_field(3, 5, 0, 50, 30, 20, 1.5)
        self.assertEqual(np.abs(a[10:30, 20:50] - b).max(), 0.0)
        c = grain.noise_field(3, 6, -20, 40, 64, 48, 1.5)
        self.assertGreater(np.abs(a - c).mean(), 0.5)
        d = grain.noise_field(4, 5, -20, 40, 64, 48, 1.5)
        self.assertGreater(np.abs(a - d).mean(), 0.5)

    def test_size_controls_spatial_correlation(self):
        def lag1(s):
            n = grain.noise_field(1, 0, 0, 0, 256, 256, s)[..., 0]
            return np.corrcoef(n[:, :-1].ravel(), n[:, 1:].ravel())[0, 1]

        self.assertLess(lag1(0.5), lag1(1.0))
        self.assertLess(lag1(1.0), lag1(3.0))

    def test_amount_scales_log_std(self):
        luts = make_luts()
        img = grey(96, 96)
        base = render(img, luts, Params())
        stds = []
        for amt in (0.5, 1.0, 2.0):
            out = render(img, luts, Params(grain_amount=amt, grain_sigma_px=1.0, grain_seed=9))
            stds.append(np.std(np.log10(spaces.di_decode(out[..., 1]) / spaces.di_decode(base[..., 1]))))
        self.assertAlmostEqual(stds[1] / stds[0], 2.0, delta=0.05)
        self.assertAlmostEqual(stds[2] / stds[1], 2.0, delta=0.05)
        print(f"\n[measure] mid-grey log10 std at amount 1: {stds[1]:.4f} (model target {grain.BASE_SIGMA_LOG10:.4f})")


@unittest.skipUnless(CLI.exists(), "fe_cli not built")
class GrainCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.luts = make_luts()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parity_with_oracle(self):
        rng = np.random.default_rng(0)
        img = np.concatenate([spaces.di_encode(rng.uniform(0.01, 2.0, (40, 52, 3))), np.ones((40, 52, 1))], -1).astype(np.float32)
        for p in (Params(grain_amount=1.0, grain_sigma_px=0.8, grain_seed=5, frame=12, origin_x=100, origin_y=-7),
                  Params(grain_amount=2.0, grain_sigma_px=3.3, grain_seed=77, frame=-3, push_pull=0.6, out_space="rec709_g24")):
            gpu = run_cuda(img, self.luts, p, self.tmp.name)
            ora = render(img, self.luts, p)
            err = np.max(np.abs(gpu - ora) / np.maximum(1.0, np.abs(ora)))
            print(f"\n[measure] grain CUDA vs oracle (sigma={p.grain_sigma_px}): {err:.2e}")
            self.assertLess(err, 1e-4)

    def test_tiles_match_full_frame(self):
        img = grey(48, 64)
        p = Params(grain_amount=1.5, grain_sigma_px=2.0, grain_seed=3, frame=4)
        full = run_cuda(img, self.luts, p, self.tmp.name)
        tile = run_cuda(img[10:30, 20:50].copy(), self.luts, replace(p, origin_x=20, origin_y=10), self.tmp.name)
        self.assertLess(np.abs(full[10:30, 20:50] - tile).max(), 1e-6)

    def test_render_order_independent(self):
        img = grey(32, 32)
        frames = [run_cuda(img, self.luts, Params(grain_amount=1.0, frame=f), self.tmp.name) for f in (3, 1, 2)]
        again = [run_cuda(img, self.luts, Params(grain_amount=1.0, frame=f), self.tmp.name) for f in (1, 2, 3)]
        self.assertEqual(frames[0].tobytes(), again[2].tobytes())
        self.assertEqual(frames[1].tobytes(), again[0].tobytes())
        self.assertNotEqual(frames[0].tobytes(), frames[1].tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
