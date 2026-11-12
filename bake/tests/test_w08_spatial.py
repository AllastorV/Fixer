import tempfile
import unittest
import warnings
from dataclasses import replace

import numpy as np

warnings.simplefilter("ignore")

from filmemu import spaces, spatial
from filmemu.pipeline import Params, render
from tests.test_w03_cuda import CLI, make_luts, run_cuda


def scene(h, w, seed=0):
    rng = np.random.default_rng(seed)
    lin = rng.uniform(0.02, 0.4, (h, w, 3))
    lin[h // 3:h // 3 + 4, w // 2:w // 2 + 4] = 40.0
    lin[2 * h // 3, :] = 8.0
    img = np.concatenate([spaces.di_encode(lin), np.ones((h, w, 1))], -1).astype(np.float32)
    return img


class SpatialOracle(unittest.TestCase):
    def test_flat_field_unchanged(self):
        lin = np.full((40, 40, 3), 0.3)
        self.assertLess(np.abs(spatial.halation(lin, 1.0, 6.0) - lin).max(), 1e-12)
        self.assertLess(np.abs(spatial.bloom(lin, 1.0, 6.0) - lin).max(), 1e-12)

    def test_energy_conserved_away_from_edges(self):
        lin = np.zeros((161, 161, 3))
        lin[80, 80] = 100.0
        out = spatial.halation(lin, 0.5, 8.0)
        self.assertAlmostEqual(out[..., 0].sum(), lin[..., 0].sum(), places=6)
        self.assertAlmostEqual(out[..., 2].sum(), lin[..., 2].sum(), places=6)

    def test_halation_is_red_dominant(self):
        lin = np.zeros((81, 81, 3))
        lin[40, 40] = 100.0
        out = spatial.halation(lin, 0.5, 6.0)
        ring = out[40, 46]
        self.assertGreater(ring[0], 3.0 * ring[1])
        self.assertEqual(ring[2], 0.0)

    def test_halo_grows_with_radius(self):
        lin = np.zeros((161, 161, 3))
        lin[80, 80] = 100.0
        near = spatial.halation(lin, 0.5, 4.0)[80, 92, 0]
        far = spatial.halation(lin, 0.5, 12.0)[80, 92, 0]
        self.assertGreater(far, near)


@unittest.skipUnless(CLI.exists(), "fe_cli not built")
class SpatialCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.luts = make_luts()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parity_with_oracle(self):
        img = scene(60, 80)
        cases = [
            Params(halation_amount=0.6, halation_radius_px=5.0),
            Params(bloom_amount=0.4, bloom_radius_px=7.0, out_space="rec709_g24"),
            Params(halation_amount=0.3, halation_radius_px=3.0, bloom_amount=0.2, bloom_radius_px=4.0, grain_amount=1.0,
                   grain_sigma_px=1.2, grain_seed=4, frame=2, origin_x=-30, origin_y=11),
        ]
        for p in cases:
            gpu = run_cuda(img, self.luts, p, self.tmp.name)
            ora = render(img, self.luts, p)
            err = np.max(np.abs(gpu - ora) / np.maximum(1.0, np.abs(ora)))
            print(f"\n[measure] spatial CUDA vs oracle (hal={p.halation_amount}, bloom={p.bloom_amount}): {err:.2e}")
            self.assertLess(err, 1e-4)

    def test_window_matches_oracle_window(self):
        img = scene(60, 80, 1)
        p = Params(halation_amount=0.5, halation_radius_px=4.0, bloom_amount=0.3, bloom_radius_px=3.0, origin_x=100, origin_y=50)
        win = (120, 60, 160, 95)
        gpu = run_cuda(img, self.luts, p, self.tmp.name, window=win)
        ora = render(img, self.luts, p, window=win)
        x1, y1, x2, y2 = (win[0] - 100, win[1] - 50, win[2] - 100, win[3] - 50)
        err = np.max(np.abs(gpu[y1:y2, x1:x2] - ora) / np.maximum(1.0, np.abs(ora)))
        self.assertLess(err, 1e-4)
        mask = np.ones(img.shape[:2], bool)
        mask[y1:y2, x1:x2] = False
        self.assertTrue(np.all(gpu[mask] == -7.0))

    def test_tiles_match_full_frame_inside(self):
        img = scene(64, 96, 2)
        p = Params(halation_amount=0.5, halation_radius_px=3.0, bloom_amount=0.3, bloom_radius_px=2.0)
        full = run_cuda(img, self.luts, p, self.tmp.name)
        left = run_cuda(img, self.luts, p, self.tmp.name, window=(0, 0, 48, 64))
        right = run_cuda(img, self.luts, p, self.tmp.name, window=(48, 0, 96, 64))
        stitched = np.concatenate([left[:, :48], right[:, 48:]], axis=1)
        err = np.abs(stitched - full).max()
        print(f"\n[measure] spatial tile seam vs full frame: {err:.2e}")
        self.assertLess(err, 1e-5)

    def test_off_is_bitwise_fast_path(self):
        img = scene(32, 32, 3)
        a = run_cuda(img, self.luts, Params(), self.tmp.name)
        b = run_cuda(img, self.luts, Params(halation_amount=0.0, bloom_amount=0.0, halation_radius_px=9.0), self.tmp.name)
        self.assertEqual(a.tobytes(), b.tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
