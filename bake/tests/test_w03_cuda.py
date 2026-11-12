import pathlib
import struct
import subprocess
import tempfile
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore")

from filmemu import film, lut, spaces
from filmemu.pipeline import Params, render

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLI = ROOT / "build" / "release" / "fe_cli.exe"
SPACE_ID = {"dwg_di": 0, "dwg_linear": 1, "acescct": 2, "rec709_g24": 3}
W, H = 37, 23


def make_luts():
    return {k: lut.bake(lambda x, p=p: film.render_film(x, p)) for k, p in film.synthetic_slots().items()}


def make_image(space, seed):
    rng = np.random.default_rng(seed)
    lin = np.concatenate([rng.uniform(-0.05, 1.5, (W * H - 40, 3)), np.geomspace(1e-4, 60, 40)[:, None].repeat(3, 1)])
    rgb = spaces.from_working(lin, space)
    alpha = rng.uniform(0, 1, (W * H, 1))
    alpha[:5] = 0
    return np.concatenate([rgb, alpha], -1).astype(np.float32).reshape(H, W, 4)


def run_cuda(img, luts, p, tmp, window=None, spatial_half=0):
    tmp = pathlib.Path(tmp)
    blob = struct.pack(
        "<iifffiiiffIiiifffffffiii", SPACE_ID[p.in_space], SPACE_ID[p.out_space], p.exposure_ev, p.push_pull, p.impact,
        int(p.premultiplied), 0, 0, p.grain_amount, p.grain_sigma_px, p.grain_seed, p.frame, p.origin_x, p.origin_y,
        p.halation_amount, p.halation_radius_px, p.bloom_amount, p.bloom_radius_px, *p.print_log_offset, spatial_half, 0, 0
    )
    (tmp / "p.bin").write_bytes(blob)
    (tmp / "in.raw").write_bytes(np.ascontiguousarray(img, np.float32).tobytes())
    names = []
    for slot in ("under", "normal", "over"):
        f = tmp / f"{slot}.bin"
        f.write_bytes(np.ascontiguousarray(luts[slot], np.float32).tobytes())
        names.append(str(f))
    n = luts["normal"].shape[0]
    extra = ["--window", *map(str, window)] if window else []
    if luts.get("print") is not None:
        f = tmp / "print.bin"
        f.write_bytes(np.ascontiguousarray(luts["print"], np.float32).tobytes())
        extra += ["--print", str(f), repr(float(luts["print_domain"][0])), repr(float(luts["print_domain"][1]))]
    if luts.get("grain") is not None:
        g = tmp / "grain.bin"
        g.write_bytes(np.ascontiguousarray(luts["grain"], np.float32).tobytes())
        extra += ["--grain", str(g)]
    if luts.get("grain_mono"):
        extra += ["--grain-mono"]
    r = subprocess.run(
        [str(CLI), str(tmp / "p.bin"), str(tmp / "in.raw"), str(tmp / "out.raw"), str(img.shape[1]), str(img.shape[0]), str(n), *names,
         *extra],
        capture_output=True, text=True,
    )
    if r.returncode:
        raise RuntimeError(r.stderr)
    return np.frombuffer((tmp / "out.raw").read_bytes(), np.float32).reshape(img.shape)


@unittest.skipUnless(CLI.exists(), "fe_cli not built")
class CudaParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.luts = make_luts()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def compare(self, img, p):
        gpu = run_cuda(img, self.luts, p, self.tmp.name)
        exact = render(img.reshape(-1, 4), self.luts, p).reshape(img.shape)
        quant = render(img.reshape(-1, 4), self.luts, p, quantize_weights=True).reshape(img.shape)
        scale = np.maximum(1.0, np.abs(quant))
        return np.max(np.abs(gpu - exact) / scale), np.max(np.abs(gpu - quant) / scale), gpu

    def test_parity_matrix(self):
        rows = []
        cases = [
            Params(),
            Params(in_space="acescct", out_space="rec709_g24", exposure_ev=1.0),
            Params(in_space="rec709_g24", out_space="dwg_di", exposure_ev=-1.5, push_pull=1.3),
            Params(in_space="dwg_linear", out_space="acescct", push_pull=-0.7, impact=0.5),
            Params(push_pull=3.0, premultiplied=True),
            Params(push_pull=-2.0, exposure_ev=2.0),
        ]
        for i, p in enumerate(cases):
            img = make_image(p.in_space, i)
            e_exact, e_quant, _ = self.compare(img, p)
            rows.append((p, e_exact, e_quant))
            self.assertLess(e_exact, 2e-5, p)
        print()
        for p, a, b in rows:
            print(f"[measure] {p.in_space}->{p.out_space} ev={p.exposure_ev} pp={p.push_pull} imp={p.impact} pre={p.premultiplied}: vs exact {a:.2e}")

    def test_impact_zero_bitwise(self):
        img = make_image("dwg_di", 9)
        img[0, 0, :3] = [np.nan, np.inf, -np.inf]
        gpu = run_cuda(img, self.luts, Params(impact=0.0), self.tmp.name)
        self.assertEqual(gpu.tobytes(), img.tobytes())

    def test_nonfinite_and_alpha(self):
        img = make_image("dwg_di", 10)
        img[0, :3, :3] = [[np.nan, 0.3, 0.3], [np.inf, 0.3, 0.3], [-np.inf, 1e30, 0.3]]
        gpu = run_cuda(img, self.luts, Params(), self.tmp.name)
        self.assertTrue(np.all(np.isfinite(gpu)))
        self.assertTrue(np.array_equal(gpu[..., 3], img[..., 3]))

    def test_repeat_render_identical(self):
        img = make_image("dwg_di", 11)
        a = run_cuda(img, self.luts, Params(push_pull=0.4), self.tmp.name)
        b = run_cuda(img, self.luts, Params(push_pull=0.4), self.tmp.name)
        self.assertEqual(a.tobytes(), b.tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
