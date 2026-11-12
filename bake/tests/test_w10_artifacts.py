import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore")

from filmemu import spaces
from filmemu.pipeline import Params, render
from tests.test_w03_cuda import CLI, make_luts, run_cuda

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILES = ROOT / "build" / "release" / "Fixer.ofx.bundle" / "Contents" / "Resources" / "profiles"
STOCKS = ("kodak-vision3-250d-2383", "kodak-portra400-endura", "fuji-velvia50", "ilford-hp5plus-ilford-multigrade-iv-rc")


def profile(pid):
    d = PROFILES / pid
    meta = json.loads((d / "profile.json").read_text(encoding="utf-8"))
    n = meta["lut_size"]
    out = {s: np.fromfile(d / f"{s}.bin", np.float32).reshape(n, n, n, 3) for s in ("under", "normal", "over", "print")}
    out["print_domain"] = (meta["print_lo"], meta["print_hi"])
    if "grain" in meta:
        out["grain"] = np.fromfile(d / "grain.bin", np.float32).reshape(3, -1)
    out["grain_mono"] = meta.get("grain_mono") == "true"
    return out


def patches(h=64, w=64):
    cols = [(0.35, 0.03, 0.03), (0.03, 0.25, 0.03), (0.03, 0.05, 0.35), (0.18, 0.18, 0.18), (0.6, 0.3, 0.1)]
    lin = np.concatenate([np.broadcast_to(np.array(c), (h, w, 3)) for c in cols], 1)
    return np.concatenate([spaces.di_encode(lin), np.ones(lin.shape[:2] + (1,))], -1).astype(np.float32)


def gradient(h=8, w=512, top=16.0):
    lin = np.broadcast_to(np.geomspace(0.01, top, w)[None, :, None], (h, w, 3)).copy()
    return np.concatenate([spaces.di_encode(lin), np.ones((h, w, 1))], -1).astype(np.float32)


def banding(row):
    d = np.diff(row)
    return float(d.min()), float(np.abs(np.diff(d)).max() / max(np.median(np.abs(d)), 1e-9))


class GrainZeroIsNeutral(unittest.TestCase):
    def test_zero_amount_is_bit_identical_python(self):
        img = patches()
        for pid in STOCKS:
            if not (PROFILES / pid / "profile.json").exists():
                self.skipTest("profiles not baked")
            lt = profile(pid)
            plain = render(img, lt, Params(out_space="rec709_g24"))
            zero = render(img, lt, Params(out_space="rec709_g24", grain_amount=0.0, grain_sigma_px=3.0, grain_seed=5))
            self.assertTrue(np.array_equal(plain, zero), pid)

    def test_grain_does_not_shift_patch_colour(self):
        img = patches(160, 160)
        for pid in STOCKS:
            if not (PROFILES / pid / "profile.json").exists():
                self.skipTest("profiles not baked")
            lt = profile(pid)
            plain = render(img, lt, Params(out_space="rec709_g24"))[..., :3]
            grainy = render(img, lt, Params(out_space="rec709_g24", grain_amount=0.5, grain_sigma_px=1.0, grain_seed=5))[..., :3]
            shift = np.abs(grainy.reshape(160, 5, 160, 3).mean((0, 2)) - plain.reshape(160, 5, 160, 3).mean((0, 2))).max()
            print(f"\n[measure] {pid} grain 0.5 mean patch shift {shift:.4f}")
            self.assertLess(shift, 0.025, pid)


@unittest.skipUnless(CLI.exists(), "fe_cli not built")
class GrainZeroCuda(unittest.TestCase):
    def test_zero_amount_is_bit_identical_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            luts, img = make_luts(), patches(32, 32)
            plain = run_cuda(img, luts, Params(), tmp)
            zero = run_cuda(img, luts, Params(grain_amount=0.0, grain_sigma_px=3.0, grain_seed=5, frame=7), tmp)
            self.assertTrue(np.array_equal(plain, zero))


class HalationNoBanding(unittest.TestCase):
    def test_gradient_stays_smooth_python(self):
        img = gradient()
        for pid in STOCKS:
            if not (PROFILES / pid / "profile.json").exists():
                self.skipTest("profiles not baked")
            lt = profile(pid)
            for amount, radius in ((0.3, 6.0), (1.0, 30.0)):
                row = render(img, lt, Params(out_space="rec709_g24", halation_amount=amount, halation_radius_px=radius,
                                             bloom_amount=0.5))[4, :, 0]
                dmin, jump = banding(row[40:-40])
                self.assertGreater(dmin, -1e-4, (pid, amount))
                self.assertLess(jump, 6.0, (pid, amount))


@unittest.skipUnless(CLI.exists(), "fe_cli not built")
class HalationNoBandingCuda(unittest.TestCase):
    def test_gradient_stays_smooth_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            luts, img = make_luts(), gradient()
            for amount, radius in ((0.3, 6.0), (1.0, 30.0)):
                out = run_cuda(img, luts, Params(halation_amount=amount, halation_radius_px=radius, bloom_amount=0.5), tmp)
                ref = render(img, luts, Params(halation_amount=amount, halation_radius_px=radius, bloom_amount=0.5))
                dmin, jump = banding(out[4, 40:-40, 0])
                self.assertGreater(dmin, -1e-4, amount)
                self.assertLess(jump, 6.0, amount)
                self.assertLess(np.abs(out - ref).max(), 1e-3, amount)

    def test_fast_glow_is_close_to_full_and_smooth(self):
        from tests.test_w08_spatial import scene

        with tempfile.TemporaryDirectory() as tmp:
            luts = make_luts()
            p = Params(halation_amount=0.6, halation_radius_px=8.0, bloom_amount=0.4, bloom_radius_px=10.0)
            img = scene(90, 120)
            full = run_cuda(img, luts, p, tmp)
            fast = run_cuda(img, luts, p, tmp, spatial_half=1)
            d = np.abs(fast - full)[..., :3]
            print(f"\n[measure] fast vs full glow: max {d.max():.2e} mean {d.mean():.2e}")
            self.assertFalse(np.array_equal(fast, full))
            self.assertLess(d.mean(), 3e-3)
            self.assertLess(d.max(), 5e-2)
            row = run_cuda(gradient(), luts, p, tmp, spatial_half=1)[4, 40:-40, 0]
            dmin, jump = banding(row)
            self.assertGreater(dmin, -1e-4)
            self.assertLess(jump, 6.0)
