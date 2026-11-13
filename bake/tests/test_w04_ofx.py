import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore")

from filmemu.pipeline import Params, render
from tests.test_w03_cuda import SPACE_ID, make_image

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "release"
HOST = BUILD / "ofx_test_host.exe"
BUNDLE = BUILD / "Fixer.ofx.bundle"
PLUGIN = BUNDLE / "Contents" / "Win64" / "Fixer.ofx"
RES = BUNDLE / "Contents" / "Resources"


os.environ["FE_TEST_PROFILES"] = "1"
def profile_label(pid):
    return json.loads((RES / "profiles" / pid / "profile.json").read_text(encoding="utf-8"))["label"]


def natural_key(pid):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", profile_label(pid))]


PROFILE_IDS = sorted((p.name for p in (RES / "profiles").iterdir() if (p / "profile.json").exists()), key=natural_key) if (RES / "profiles").exists() else []
PROFILE_INDEX = {pid: i for i, pid in enumerate(PROFILE_IDS)}


def bundle_luts(profile_id="synthetic-test"):
    d = RES / "profiles" / profile_id
    n = json.loads((d / "profile.json").read_text(encoding="utf-8"))["lut_size"]
    return {s: np.fromfile(d / f"{s}.bin", np.float32).reshape(n, n, n, 3) for s in ("under", "normal", "over")}


def run_host(img, p, window, tmp, profile_id="synthetic-test", grain=None, spatial=None, stored_id=None, light=0, balance=0):
    tmp = pathlib.Path(tmp)
    (tmp / "in.raw").write_bytes(np.ascontiguousarray(img, np.float32).tobytes())
    h, w = img.shape[:2]
    args = [str(HOST), str(PLUGIN), str(tmp / "in.raw"), str(tmp / "out.raw"), str(w), str(h), str(SPACE_ID[p.in_space]),
            str(SPACE_ID[p.out_space]), str(p.exposure_ev), str(p.push_pull), str(p.impact), str(int(p.premultiplied)),
            *map(str, window), str(PROFILE_INDEX[profile_id]), *(map(str, grain) if grain else []), *(map(str, spatial) if spatial else []),
            *(["--profile-id", stored_id] if stored_id is not None else []), *(["--light", str(light)] if light else []), *(["--balance", str(balance)] if balance else [])]
    r = subprocess.run(args, capture_output=True, text=True)
    out = np.frombuffer((tmp / "out.raw").read_bytes(), np.float32).reshape(img.shape) if (tmp / "out.raw").exists() else None
    return r, out


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class OfxAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.luts = bundle_luts()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, p, window, seed):
        img = make_image(p.in_space, seed)
        r, out = run_host(img, p, window, self.tmp.name)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        x1, y1, x2, y2 = window
        expect = render(img[y1:y2, x1:x2].reshape(-1, 4), self.luts, p).reshape(y2 - y1, x2 - x1, 4)
        got = out[y1:y2, x1:x2]
        err = np.max(np.abs(got - expect) / np.maximum(1.0, np.abs(expect)))
        mask = np.ones(img.shape[:2], bool)
        mask[y1:y2, x1:x2] = False
        self.assertTrue(np.all(out[mask] == -7.0), "pixels outside the render window were written")
        return r.stdout, err

    def test_describe_and_full_frame(self):
        log, err = self.check(Params(), (0, 0, 37, 23), 0)
        self.assertIn("plugin=com.alpcavas.fixer", log)
        self.assertIn("cuda_render_supported=true", log)
        self.assertIn("params=48 clips=2", log)
        self.assertIn("profile_id=synthetic-test", log)
        for pid, i in PROFILE_INDEX.items():
            self.assertIn(f"profile_option {i} {profile_label(pid)}", log)
        self.assertIn("is_identity=no", log)
        self.assertLess(err, 2e-5)
        print(f"\n[measure] OFX full frame vs oracle: {err:.2e}")

    def test_render_window_offset(self):
        log, err = self.check(Params(in_space="acescct", out_space="rec709_g24", exposure_ev=0.5, push_pull=-0.6), (5, 3, 33, 21), 1)
        self.assertLess(err, 2e-5)
        print(f"\n[measure] OFX sub-window vs oracle: {err:.2e}")

    def test_premultiplied(self):
        _, err = self.check(Params(premultiplied=True, push_pull=1.1), (0, 0, 37, 23), 2)
        self.assertLess(err, 2e-5)

    def test_cpu_render_refused_with_message(self):
        log, _ = self.check(Params(), (0, 0, 37, 23), 3)
        self.assertIn("cpu_render_status=1", log)
        self.assertIn("needs CUDA", log)

    def test_identity_profile_passes_image_through(self):
        img = make_image("dwg_di", 5)
        r, out = run_host(img, Params(), (0, 0, 37, 23), self.tmp.name, "identity")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        inside = (img[..., :3] > 0) & (img[..., :3] < 0.95)
        err = np.max(np.abs(out[..., :3] - img[..., :3])[inside])
        print(f"\n[measure] identity profile max |out-in| (DI, in-domain): {err:.2e}")
        self.assertLess(err, 1e-5)

    def test_test_profiles_hidden_from_users(self):
        img = make_image("dwg_di", 7)
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "in.raw").write_bytes(img.tobytes())
            args = [str(HOST), str(PLUGIN), str(pathlib.Path(d) / "in.raw"), str(pathlib.Path(d) / "out.raw"),
                    "37", "23", "0", "0", "0", "0", "1", "0", "0", "0", "37", "23", "0"]
            env = {k: v for k, v in os.environ.items() if k != "FE_TEST_PROFILES"}
            r = subprocess.run(args, capture_output=True, text=True, env=env)
        for pid in PROFILE_IDS:
            meta = json.loads((RES / "profiles" / pid / "profile.json").read_text(encoding="utf-8"))
            if meta.get("test") == "true":
                self.assertNotIn(meta["label"], r.stdout)
            else:
                self.assertIn(meta["label"], r.stdout)

    def test_corrupt_profile_is_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            copy = pathlib.Path(d) / "Fixer.ofx.bundle"
            shutil.copytree(BUNDLE, copy)
            bad = copy / "Contents" / "Resources" / "profiles" / "identity" / "under.bin"
            bad.write_bytes(bad.read_bytes()[:1000])
            img = make_image("dwg_di", 6)
            (pathlib.Path(d) / "in.raw").write_bytes(img.tobytes())
            args = [str(HOST), str(copy / "Contents" / "Win64" / "Fixer.ofx"), str(pathlib.Path(d) / "in.raw"),
                    str(pathlib.Path(d) / "out.raw"), "37", "23", "0", "0", "0", "0", "1", "0", "0", "0", "37", "23", "0"]
            r = subprocess.run(args, capture_output=True, text=True)
        self.assertIn(f"profile_option 0 {profile_label([p for p in PROFILE_IDS if p != 'identity'][0])}", r.stdout)
        self.assertNotIn("Identity (test)", r.stdout)

    def test_grain_through_plugin_matches_oracle(self):
        img = make_image("dwg_di", 7)
        amount, size, seed, time = 1.2, 1.5, 42, 17.0
        window = (4, 2, 30, 20)
        r, out = run_host(img, Params(), window, self.tmp.name, grain=(amount, size, seed, time))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        x1, y1, x2, y2 = window
        sigma = size * img.shape[0] / 1080.0
        p = Params(grain_amount=amount, grain_sigma_px=sigma, grain_seed=seed, frame=int(time), origin_x=x1, origin_y=y1)
        expect = render(img[y1:y2, x1:x2], self.luts, p)
        err = np.max(np.abs(out[y1:y2, x1:x2] - expect) / np.maximum(1.0, np.abs(expect)))
        print(f"\n[measure] OFX grain vs oracle: {err:.2e}")
        self.assertLess(err, 1e-4)

    def test_halation_bloom_through_plugin_match_oracle(self):
        from tests.test_w08_spatial import scene
        img = scene(108, 64, 5)
        hal, hal_r, bloom = 0.5, 30.0, 0.3
        window = (6, 10, 58, 90)
        r, out = run_host(img, Params(), window, self.tmp.name, grain=(0.0, 1.0, 0, 0.0), spatial=(hal, hal_r, bloom))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        k = img.shape[0] / 1080.0
        p = Params(halation_amount=hal, halation_radius_px=hal_r * k, bloom_amount=bloom, bloom_radius_px=15.0 * k)
        expect = render(img, self.luts, p, window=window)
        x1, y1, x2, y2 = window
        err = np.max(np.abs(out[y1:y2, x1:x2] - expect) / np.maximum(1.0, np.abs(expect)))
        print(f"\n[measure] OFX halation+bloom vs oracle: {err:.2e}")
        self.assertLess(err, 1e-4)

    def test_stored_profile_id_wins_over_shifted_index(self):
        img = make_image("dwg_di", 8)
        r, out = run_host(img, Params(), (0, 0, 37, 23), self.tmp.name, "identity", stored_id="synthetic-test")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"profile_choice_after_create={PROFILE_INDEX['synthetic-test']}", r.stdout)
        expect = render(img.reshape(-1, 4), self.luts, Params()).reshape(img.shape)
        self.assertLess(np.max(np.abs(out - expect) / np.maximum(1.0, np.abs(expect))), 2e-5)

    def test_unknown_stored_id_falls_back_to_choice(self):
        img = make_image("dwg_di", 9)
        r, out = run_host(img, Params(), (0, 0, 37, 23), self.tmp.name, "synthetic-test", stored_id="removed-profile")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"profile_choice_after_create={PROFILE_INDEX['synthetic-test']}", r.stdout)
        expect = render(img.reshape(-1, 4), self.luts, Params()).reshape(img.shape)
        self.assertLess(np.max(np.abs(out - expect) / np.maximum(1.0, np.abs(expect))), 2e-5)

    def test_empty_stored_id_keeps_default_stock(self):
        img = make_image("dwg_di", 9)
        r, out = run_host(img, Params(), (0, 0, 37, 23), self.tmp.name, "synthetic-test", stored_id="")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"profile_choice_after_create={PROFILE_INDEX['kodak-vision3-250d-2383']}", r.stdout)
        r2, ref = run_host(img, Params(), (0, 0, 37, 23), self.tmp.name, "synthetic-test", stored_id="kodak-vision3-250d-2383")
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        self.assertEqual(out.tobytes(), ref.tobytes())

    def test_impact_zero_is_identity(self):
        img = make_image("dwg_di", 4)
        r, out = run_host(img, Params(impact=0.0), (0, 0, 37, 23), self.tmp.name)
        self.assertIn("is_identity=yes", r.stdout)
        self.assertEqual(out.tobytes(), img.tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
