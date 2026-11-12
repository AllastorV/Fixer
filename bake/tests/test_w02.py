import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore")
import colour

from filmemu import film, lut, spaces
from filmemu.pipeline import Params, render

RAMP = np.concatenate([np.linspace(-0.1, 0.01, 2001), np.geomspace(0.01, 100.0, 4001)])


def film_luts(points=(0.0, 0.0, 0.0)):
    return {k: lut.bake(lambda x, p=p: film.render_film(x, p, points)) for k, p in film.synthetic_slots().items()}


class Transfers(unittest.TestCase):
    def test_di_matches_independent_oracle(self):
        x = RAMP[RAMP >= 0]
        ours = spaces.di_encode(x)
        ref = colour.models.oetf_DaVinciIntermediate(x)
        self.assertLessEqual(np.max(np.abs(ours - ref)), 1e-6)
        y = np.linspace(0.0, 1.0, 5001)
        self.assertLessEqual(np.max(np.abs(spaces.di_decode(y) - colour.models.oetf_inverse_DaVinciIntermediate(y))), 1e-6 * 100)

    def test_acescct_matches_independent_oracle(self):
        x = RAMP[RAMP > 0]
        ref = colour.models.log_encoding_ACEScct(x)
        self.assertLessEqual(np.max(np.abs(spaces.acescct_encode(x) - ref)), 1e-6)

    def test_roundtrip_all_spaces(self):
        rgb = np.stack([RAMP, RAMP[::-1] * 0.5, np.clip(RAMP, -0.05, 4)], axis=-1)
        for name, (_, enc, dec) in spaces.SPACES.items():
            back = dec(enc(RAMP))
            self.assertLessEqual(np.max(np.abs(back - RAMP) / np.maximum(1, np.abs(RAMP))), 1e-6, name)
            w = spaces.to_working(spaces.from_working(rgb, name), name)
            self.assertLessEqual(np.max(np.abs(w - rgb) / np.maximum(1, np.abs(rgb))), 1e-6, name)

    def test_di_is_continuous_at_cut(self):
        e = 1e-9
        self.assertAlmostEqual(spaces.di_encode(spaces.DI_LIN_CUT - e), spaces.di_encode(spaces.DI_LIN_CUT + e), places=6)


class Exposure(unittest.TestCase):
    def test_integer_ev_is_bitwise(self):
        x = np.float32(RAMP)
        for ev in (-3, -1, 1, 4):
            self.assertTrue(np.array_equal(np.float32(x * np.exp2(ev)), x * np.float32(2.0 ** ev)))


class Lut(unittest.TestCase):
    def test_identity_lut_exact_trilinear(self):
        ident = lut.bake(lambda x: x)
        rng = np.random.default_rng(1)
        lin = spaces.di_decode(rng.uniform(0, 1, (20000, 3)))
        err = np.abs(lut.apply(ident, lin) - lin) / np.maximum(1, np.abs(lin))
        self.assertLessEqual(err.max(), 1e-5)

    def test_hardware_weight_quantization_error_is_measured(self):
        luts = film_luts()
        rng = np.random.default_rng(2)
        c = rng.uniform(0, 1, (20000, 3))
        d = np.abs(lut.sample(luts["normal"], c) - lut.sample(luts["normal"], c, quantize_weights=True))
        print(f"\n[measure] 1/256 weight quantization, DI units: max={d.max():.2e} mean={d.mean():.2e}")
        self.assertLess(d.max(), 5e-3)

    def test_bake_error_vs_direct_film(self):
        luts = film_luts()
        prof = film.synthetic_slots()["normal"]
        rng = np.random.default_rng(3)
        lin = spaces.di_decode(rng.uniform(0.05, 0.95, (20000, 3)))
        direct = spaces.di_encode(film.render_film(lin, prof))
        via = spaces.di_encode(lut.apply(luts["normal"], lin))
        d = np.abs(direct - via)
        print(f"\n[measure] 65^3 bake error, DI units: max={d.max():.2e} p99={np.quantile(d, 0.99):.2e}")
        self.assertLess(d.max(), 5e-3)

    def test_out_of_domain_clamps_to_edge(self):
        ident = lut.bake(lambda x: x)
        hi = lut.sample(ident, np.array([[2.0, 2.0, 2.0]]))
        self.assertTrue(np.allclose(hi, ident[-1, -1, -1]))


class Slots(unittest.TestCase):
    def test_partition_of_unity_and_continuity(self):
        ps = np.linspace(-3, 3, 6001)
        ws = [lut.slot_weights(p) for p in ps]
        self.assertLessEqual(max(abs(sum(w.values()) - 1) for w in ws), 1e-7)
        over = np.array([w["over"] - w["under"] for w in ws])
        self.assertLess(np.max(np.abs(np.diff(over))), 2e-3)
        self.assertEqual(lut.slot_weights(0.0), {"under": 0.0, "normal": 1.0, "over": 0.0})

    def test_push_pull_zero_equals_normal_slot(self):
        luts = film_luts()
        rgba = np.concatenate([np.random.default_rng(4).uniform(0, 1, (64, 3)), np.ones((64, 1))], axis=-1)
        a = render(rgba, luts, Params(push_pull=0.0))
        b = render(rgba, {"under": luts["normal"], "normal": luts["normal"], "over": luts["normal"]}, Params(push_pull=1.3))
        self.assertTrue(np.allclose(a, b, atol=1e-6))

    def test_missing_slot_refuses(self):
        with self.assertRaises(ValueError):
            render(np.zeros((1, 4)), {"normal": lut.bake(lambda x: x)}, Params())


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.luts = film_luts()

    def test_impact_zero_is_bitwise_including_nonfinite(self):
        x = np.array([[np.nan, np.inf, -np.inf, 0.3], [-0.5, 7.0, 0.1, 0.0]], dtype=np.float32)
        out = render(x, self.luts, Params(impact=0.0))
        self.assertEqual(out.tobytes(), x.tobytes())

    def test_nonfinite_input_gives_finite_output(self):
        x = np.array([[np.nan, np.inf, -np.inf, 1.0]], dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(render(x, self.luts, Params()))))

    def test_alpha_untouched_and_premultiplied_roundtrip(self):
        rng = np.random.default_rng(5)
        rgb = rng.uniform(0, 1, (256, 3))
        a = rng.uniform(0, 1, (256, 1))
        a[:8] = 0
        straight = render(np.concatenate([rgb, a], -1), self.luts, Params())
        pre = render(np.concatenate([rgb * a, a], -1), self.luts, Params(premultiplied=True))
        self.assertTrue(np.array_equal(straight[:, 3], np.float32(a[:, 0])))
        self.assertTrue(np.allclose(pre[8:, :3], straight[8:, :3] * np.float32(a[8:]), atol=1e-6))

    def test_mid_grey_maps_to_mid_grey(self):
        g = np.array([[spaces.di_encode(0.18)] * 3 + [1.0]])
        out = render(g, self.luts, Params())
        self.assertLess(abs(spaces.di_decode(out[0, 0]) - 0.18), 2e-3)

    def test_neutral_ramp_monotonic_and_c1(self):
        lin = np.geomspace(1e-4, 50, 4000)
        prof = film.synthetic_slots()["normal"]
        y = np.log10(film.render_film(np.stack([lin] * 3, -1), prof)[:, 0])
        dy = np.diff(y)
        self.assertTrue(np.all(dy > 0))
        self.assertLess(np.max(np.abs(np.diff(dy / np.diff(np.log10(lin))))), 1e-2)

    def test_printer_point_shifts_print_exposure(self):
        prof = film.synthetic_slots()["normal"]
        g = np.full((1, 3), 0.18)
        base = film.render_film(g, prof)
        shifted = film.render_film(g, prof, (4.0, 0.0, -4.0))
        self.assertLess(shifted[0, 0] / base[0, 0], 1.0)
        self.assertGreater(shifted[0, 2] / base[0, 2], 1.0)
        self.assertAlmostEqual(shifted[0, 1], base[0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
