import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore")

from filmemu import spaces, spectral, stocks
from filmemu.pipeline import Params, render
from tests.test_w03_cuda import CLI, run_cuda
from tests.test_w06_spectral import px

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE = ROOT / "build" / "release" / "Fixer.ofx.bundle" / "Contents" / "Resources" / "profiles" / "kodak-vision3-250d-2383"
NEG, PRT = "kodak-vision3-250d", "kodak-2383"


class StockData(unittest.TestCase):
    def test_sensitometry_monotonic_and_plausible(self):
        for sid, dmin_max, dmax_min in ((NEG, 0.9, 1.8), (PRT, 0.12, 4.0)):
            for c, ch in stocks.load(sid)["sensitometry"]["channels"].items():
                d = np.asarray(ch["density"])
                self.assertTrue(np.all(np.diff(d) >= -1e-3), f"{sid}/{c} not monotonic")
                self.assertLess(d[0], dmin_max)
                self.assertGreater(d[-1], dmax_min)

    def test_dye_and_sensitivity_peaks(self):
        expect = {NEG: {"cyan": (670, 700), "magenta": (525, 555), "yellow": (435, 460)},
                  PRT: {"cyan": (645, 670), "magenta": (530, 555), "yellow": (435, 455)}}
        for sid, ranges in expect.items():
            data = stocks.load(sid)
            for dye, (lo, hi) in ranges.items():
                c = data["dye_density"]["curves"][dye]
                peak = c["wavelength"][int(np.argmax(c["value"]))]
                self.assertTrue(lo <= peak <= hi, f"{sid}/{dye} peak {peak}")
            peaks = [data["spectral_sensitivity"]["channels"][c]["wavelength"][int(np.argmax(data["spectral_sensitivity"]["channels"][c]["value"]))] for c in "bgr"]
            self.assertEqual(peaks, sorted(peaks))

    def test_midscale_neutral_is_base_plus_dyes(self):
        data = stocks.load(NEG)
        dyes = np.stack([stocks._spectrum(data["dye_density"]["curves"][k]) for k in stocks.DYES])
        base = stocks._spectrum(data["dye_density"]["curves"]["minimum"])
        ref = stocks._spectrum(data["dye_density"]["curves"]["midscale_neutral"])
        m = spectral.WL >= 400
        amounts, *_ = np.linalg.lstsq(dyes[:, m].T, (ref - base)[m], rcond=None)
        rms = np.sqrt(np.mean((base + amounts @ dyes - ref)[m] ** 2))
        print(f"\n[measure] 250D midscale neutral vs base+dyes fit: rms={rms:.4f}")
        self.assertLess(rms, 0.01)


class StockChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = spectral.Chain(stocks.system(NEG, PRT))

    def run_chain(self, lin):
        return self.chain.display_linear(self.chain.print_log_exposure(np.asarray(lin, np.float64)))

    def test_mid_grey_is_neutral(self):
        out = self.run_chain([[0.18] * 3])[0]
        print(f"\n[measure] 250D+2383 mid-grey: {out.round(5)}")
        self.assertLess(np.abs(out / out.mean() - 1.0).max(), 2e-3)

    def test_print_white_is_display_white(self):
        grey = self.run_chain([[0.18] * 3])[0].mean()
        top = self.run_chain([[0.18 * 2.0 ** 10] * 3])[0]
        print(f"\n[measure] 250D+2383 brightest {top.round(4)}, white/grey {top.max() / grey:.2f} (Kodak LAD ~8.7)")
        self.assertLessEqual(top.max(), 1.0 + 1e-6)
        self.assertGreater(top.max(), 0.9)
        self.assertLess(abs(top.max() / grey / 8.7 - 1.0), 0.15)

    def test_neutral_ramp_monotonic(self):
        y = self.run_chain(np.geomspace(1e-3, 30, 200)[:, None].repeat(3, 1))
        self.assertTrue(np.all(np.diff(y, axis=0) >= 0))
        self.assertTrue(np.all(np.diff(y.mean(1)) > 0))

    def test_grey_scale_neutral(self):
        stops = np.arange(-7.0, 7.01, 0.5)
        out = self.run_chain((0.18 * 2.0 ** stops)[:, None].repeat(3, 1))
        dev = np.abs(out / out[:, 1:2] - 1.0).max(axis=1)
        print(f"\n[measure] 250D+2383 grey scale channel deviation: -7..+2 {dev[stops <= 2].max():.2e}, +2..+7 {dev[stops > 2].max():.2e}")
        self.assertLess(dev[stops <= 2].max(), 1e-3)
        self.assertLess(dev[stops > 2].max(), 0.03)

    def test_primaries_keep_hue(self):
        for i in range(3):
            lin = np.full((1, 3), 0.05)
            lin[0, i] = 0.4
            out = self.run_chain(lin)[0]
            self.assertEqual(int(np.argmax(out)), i)


class EveryNegative(unittest.TestCase):
    def test_each_negative_prints_neutral_and_plausible(self):
        from bake_luts import NEGATIVES
        from filmemu import grain

        found = [n for n, _ in NEGATIVES if (stocks.STOCKS / f"{n}.json").exists()]
        self.assertIn(NEG, found)
        for neg in found:
            with self.subTest(neg=neg):
                fs = stocks.system(neg, PRT)
                ch = spectral.Chain(fs)
                run = lambda lin: ch.display_linear(ch.print_log_exposure(np.asarray(lin, np.float64)))
                ramp = run((0.18 * 2.0 ** np.arange(-7.0, 2.01, 0.5))[:, None].repeat(3, 1))
                dev = np.abs(ramp / ramp[:, 1:2] - 1.0).max()
                grey = run([[0.18] * 3])[0].mean()
                top = run([[0.18 * 2.0 ** 10] * 3])[0].max()
                hue = [int(np.argmax(run(np.where(np.eye(3)[i] > 0, 0.4, 0.05)[None, :])[0])) for i in range(3)]
                table = stocks.grain_table(fs, neg)
                g0 = [float(np.interp(0.0, grain.table_axis(), table[c])) for c in range(3)]
                print(f"\n[measure] {neg}: grey dev {dev:.1e}, white/grey {top / grey:.2f}, grain@grey R/G/B {g0[0]:.2f}/{g0[1]:.2f}/{g0[2]:.2f}")
                self.assertLess(dev, 1e-3)
                self.assertTrue(np.all(np.diff(ramp.mean(1)) > -1e-6))
                self.assertEqual(hue, [0, 1, 2])
                self.assertLess(abs(top / grey / 8.7 - 1.0), 0.15)
                self.assertLessEqual(top, 1.0 + 1e-6)
                self.assertGreater(g0[2], g0[1])


class EveryStillNegative(unittest.TestCase):
    def test_paper_profiles_grey_balanced_and_plausible(self):
        from bake_luts import STILL_NEGATIVES, STILL_PAPER

        found = [n for n, _ in STILL_NEGATIVES if (stocks.STOCKS / f"{n}.json").exists()]
        self.assertTrue(found)
        for neg in found:
            with self.subTest(neg=neg):
                ch = spectral.Chain(stocks.system(neg, STILL_PAPER))
                bal = stocks.grey_balance(ch)
                run = lambda lin: bal(ch.display_linear(ch.print_log_exposure(np.asarray(lin, np.float64))))
                ramp = run((0.18 * 2.0 ** np.arange(-8.0, 8.01, 0.5))[:, None].repeat(3, 1))
                dev = np.abs(ramp / ramp[:, 1:2] - 1.0).max()
                grey = run([[0.18] * 3])[0].mean()
                top = run([[0.18 * 2.0 ** 10] * 3])[0].max()
                hue = [int(np.argmax(run(np.where(np.eye(3)[i] > 0, 0.4, 0.05)[None, :])[0])) for i in range(3)]
                fit = stocks.derived_dyes(stocks.load(neg))[1] if "cyan" not in stocks.load(neg)["dye_density"]["curves"] else {"rms": 0.0}
                print(f"\n[measure] {neg} on Endura: grey dev {dev:.1e}, grey {grey:.3f}, white/grey {top / grey:.2f}, derived-dye rms {fit['rms']:.3f}")
                self.assertLess(dev, 2e-3)
                self.assertTrue(np.all(np.diff(ramp.mean(1)) >= -1e-4))
                self.assertEqual(hue, [0, 1, 2])
                self.assertLessEqual(top, 1.0 + 1e-6)
                self.assertLess(fit["rms"], 0.06)


class BlackAndWhite(unittest.TestCase):
    def test_bw_negatives_on_multigrade(self):
        from bake_luts import BW_NEGATIVES, BW_PAPER

        found = [n for n, _ in BW_NEGATIVES if (stocks.STOCKS / f"{n}.json").exists()]
        if not found or not (stocks.STOCKS / f"{BW_PAPER}.json").exists():
            self.skipTest("black-and-white data not digitized yet")
        for neg in found:
            with self.subTest(neg=neg):
                fs = stocks.system(neg, BW_PAPER)
                ch = stocks.make_chain(fs)
                run = lambda lin: ch.display_linear(ch.print_log_exposure(np.asarray(lin, np.float64)))
                ramp = run((0.18 * 2.0 ** np.arange(-7.0, 7.01, 0.5))[:, None].repeat(3, 1))
                prim = run(np.array([[0.4, 0.05, 0.05], [0.05, 0.3, 0.05], [0.05, 0.05, 0.4]]))
                grey = run([[0.18] * 3])[0]
                print(f"\n[measure] {neg}: grey {grey[0]:.3f}, white/grey {ramp.max() / grey[0]:.2f}, red/green/blue patches {prim[:, 0].round(3)}")
                self.assertTrue(np.all(ramp == ramp[:, :1]))
                self.assertTrue(np.all(np.diff(ramp[:, 0]) >= -1e-9))
                self.assertLessEqual(ramp.max(), 1.0 + 1e-6)
                self.assertTrue(0.12 < grey[0] < 0.3)
                self.assertTrue(np.all(prim == prim[:, :1]))
                t = stocks.grain_table(fs, neg)
                self.assertTrue(np.all(t == t[:1]))

    def test_bw_profile_mono_grain_matches_cuda(self):
        from bake_luts import BW_NEGATIVES

        profs = [PROFILE.parent / f"{n}-ilford-multigrade-iv-rc" for n, _ in BW_NEGATIVES]
        profs = [p for p in profs if (p / "profile.json").exists()]
        if not profs or not CLI.exists():
            self.skipTest("black-and-white profile not baked")
        prof = profs[0]
        meta = json.loads((prof / "profile.json").read_text(encoding="utf-8"))
        n = meta["lut_size"]
        luts = {s: np.fromfile(prof / f"{s}.bin", np.float32).reshape(n, n, n, 3) for s in ("under", "normal", "over", "print")}
        luts.update(print_domain=(meta["print_lo"], meta["print_hi"]), grain=np.fromfile(prof / "grain.bin", np.float32).reshape(3, -1),
                    grain_mono=meta.get("grain_mono") == "true")
        self.assertTrue(luts["grain_mono"])
        img = px(spaces.di_decode(np.random.default_rng(4).uniform(0.0, 1.0, (37 * 23, 3)))).reshape(23, 37, 4)
        p = Params(grain_amount=1.0, grain_sigma_px=1.2, grain_seed=9, frame=1)
        with tempfile.TemporaryDirectory() as d:
            gpu = run_cuda(img, luts, p, d)
        ora = render(img, luts, p)
        print(f"\n[measure] {prof.name} CUDA vs oracle (mono grain): {np.max(np.abs(gpu - ora)):.2e}")
        self.assertLess(np.max(np.abs(gpu - ora) / np.maximum(1.0, np.abs(ora))), 1e-4)
        self.assertLess(np.abs(gpu[..., 0] - gpu[..., 1]).max(), 1e-6)


class GreyBalanceSwitch(unittest.TestCase):
    def test_characteristic_default_and_balanced_option(self):
        from tests.test_w04_ofx import HOST, PLUGIN, run_host
        from tests.test_w03_cuda import make_image

        for pid in ("kodak-vision3-250d-2383", "kodak-gold200-endura"):
            prof = PROFILE.parent / pid
            if not (HOST.exists() and PLUGIN.exists() and (prof / "normal_bal.bin").exists()):
                self.skipTest("balanced LUTs not built")
            meta = json.loads((prof / "profile.json").read_text(encoding="utf-8"))
            n = meta["lut_size"]
            rd = lambda f: np.fromfile(prof / f, np.float32).reshape(n, n, n, 3)
            base = {"print_domain": (meta["print_lo"], meta["print_hi"])}
            char = dict(base, print=rd("print.bin"), **{s: rd(f"{s}.bin") for s in ("under", "normal", "over")})
            bal = dict(base, print=rd(meta.get("balanced_print", "print.bin")), **{s: rd(f"{s}_bal.bin") for s in ("under", "normal", "over")})
            grey = px((0.18 * 2.0 ** np.array([-4.0, 0.0, 4.0]))[:, None].repeat(3, 1))
            gc = spaces.di_decode(render(grey, char, Params())[:, :3])
            gb = spaces.di_decode(render(grey, bal, Params())[:, :3])
            print(f"\n[measure] {pid} -4 stops characteristic {gc[0].round(4)} balanced {gb[0].round(4)}")
            ramp = px((0.18 * 2.0 ** np.arange(-6.0, 4.01, 1.0))[:, None].repeat(3, 1))
            rc = spaces.di_decode(render(ramp, char, Params())[:, :3])
            rb = spaces.di_decode(render(ramp, bal, Params())[:, :3])
            self.assertGreater(np.abs(rc / rc[:, 1:2] - rb / rb[:, 1:2]).max(), 0.01)
            db = spaces.di_encode(gb)
            self.assertLess(np.abs(db - db[:, 1:2]).max(), 5e-3)
            img = make_image("dwg_di", 31)
            with tempfile.TemporaryDirectory() as d:
                for balance, luts in ((0, char), (1, bal)):
                    r, out = run_host(img, Params(), (0, 0, 37, 23), d, pid, balance=balance)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    err = np.max(np.abs(out - render(img, luts, Params())) / np.maximum(1.0, np.abs(out)))
                    self.assertLess(err, 2e-5)


class ShootingLight(unittest.TestCase):
    def test_cast_direction_follows_balance(self):
        for neg, scene, warm in (("kodak-vision3-250d", "tungsten", True), ("kodak-vision3-500t", "daylight", False)):
            with self.subTest(neg=neg):
                fs = stocks.system(neg, PRT)
                bal = stocks.balance(neg)
                matched = spectral.Chain(fs, stocks.light(bal), stocks.light(bal))
                off = spectral.Chain(fs, stocks.light(scene), stocks.light(bal))
                grey = np.full((1, 3), 0.18)
                m = matched.display_linear(matched.print_log_exposure(grey))[0]
                o = off.display_linear(off.print_log_exposure(grey))[0]
                print(f"\n[measure] {neg} under {scene}: grey {o.round(4)} (matched {m.round(4)})")
                self.assertLess(np.abs(m / m.mean() - 1).max(), 1e-3)
                self.assertEqual(o[0] > o[2], warm)
                self.assertGreater(abs(np.log2(o[0] / o[2])), 0.5)

    def test_plugin_uses_alternate_luts(self):
        from tests.test_w04_ofx import HOST, PLUGIN, run_host
        from tests.test_w03_cuda import make_image

        prof = PROFILE.parent / "kodak-vision3-500t-2383"
        if not (HOST.exists() and PLUGIN.exists() and (prof / "under_alt.bin").exists()):
            self.skipTest("plugin or 500T alt LUTs not built")
        meta = json.loads((prof / "profile.json").read_text(encoding="utf-8"))
        if "alt_light" not in meta:
            self.skipTest("alternate shooting-light LUTs not baked")
        self.assertEqual((meta["balance"], meta["alt_light"]), ("tungsten", "daylight"))
        n = meta["lut_size"]
        rd = lambda f: np.fromfile(prof / f, np.float32).reshape(n, n, n, 3)
        base = {"print": rd("print.bin"), "print_domain": (meta["print_lo"], meta["print_hi"])}
        main = dict(base, **{s: rd(f"{s}.bin") for s in ("under", "normal", "over")})
        alt = dict(base, **{s: rd(f"{s}_alt.bin") for s in ("under", "normal", "over")})
        img = make_image("dwg_di", 21)
        with tempfile.TemporaryDirectory() as d:
            for light, luts in ((0, main), (2, main), (1, alt)):
                r, out = run_host(img, Params(), (0, 0, 37, 23), d, prof.name, light=light)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                err = np.max(np.abs(out - render(img, luts, Params())) / np.maximum(1.0, np.abs(out)))
                print(f"\n[measure] OFX 500T shooting light {light} vs oracle: {err:.2e}")
                self.assertLess(err, 2e-5)


@unittest.skipUnless((PROFILE / "profile.json").exists(), "250D+2383 profile not baked")
class StockProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        meta = json.loads((PROFILE / "profile.json").read_text(encoding="utf-8"))
        n = meta["lut_size"]
        cls.meta = meta
        cls.luts = {s: np.fromfile(PROFILE / f"{s}.bin", np.float32).reshape(n, n, n, 3) for s in ("under", "normal", "over", "print")}
        cls.luts["print_domain"] = (meta["print_lo"], meta["print_hi"])
        cls.luts["grain"] = np.fromfile(PROFILE / "grain.bin", np.float32).reshape(3, -1)
        cls.luts["grain_mono"] = meta.get("grain_mono") == "true"
        cls.chain = spectral.Chain(stocks.system(NEG, PRT, balanced=False))

    def test_grain_table_follows_film(self):
        from filmemu import grain

        t, u = self.luts["grain"], grain.table_axis()
        at = lambda c, x: float(np.interp(x, u, t[c]))
        print(f"\n[measure] 250D grain table at grey R/G/B: {at(0, 0):.2f} {at(1, 0):.2f} {at(2, 0):.2f}; "
              f"-2 stops G {at(1, -0.6):.2f}; +5 stops G {at(1, 1.5):.2f}")
        self.assertEqual(self.meta["grain"], "grain.bin")
        gran = stocks.load(NEG)["granularity"]
        for c, ch in enumerate("rgb"):
            ref = stocks._granularity(gran, ch, u) / stocks._granularity(gran, "g", np.array([0.0]))[0]
            self.assertLess(np.abs(t[c] / at(1, 0.0) - ref).max(), 1e-2, ch)
        self.assertGreater(at(2, 0.0), at(1, 0.0))
        self.assertTrue(0.2 < at(1, 0.0) < 1.0)

    def test_grain_through_plugin_uses_table(self):
        from tests.test_w04_ofx import HOST, PLUGIN, PROFILE_INDEX, run_host
        from tests.test_w03_cuda import make_image

        if not (HOST.exists() and PLUGIN.exists()):
            self.skipTest("plugin not built")
        img = make_image("dwg_di", 12)
        amount, size, seed, time = 1.0, 1.2, 7, 3.0
        with tempfile.TemporaryDirectory() as d:
            r, out = run_host(img, Params(), (0, 0, 37, 23), d, PROFILE.name, grain=(amount, size, seed, time))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        p = Params(grain_amount=amount, grain_sigma_px=size * img.shape[0] / 1080.0, grain_seed=seed, frame=int(time))
        expect = render(img, self.luts, p)
        no_table = render(img, {k: v for k, v in self.luts.items() if k != "grain"}, p)
        err = np.max(np.abs(out - expect) / np.maximum(1.0, np.abs(expect)))
        print(f"\n[measure] OFX 250D grain vs oracle: {err:.2e}")
        self.assertLess(err, 1e-4)
        self.assertGreater(np.max(np.abs(out - no_table)), 1e-3)

    def test_metadata(self):
        self.assertEqual(self.meta["kind"], "two_stage")
        self.assertIn("Kodak", self.meta["data_source"])

    def test_lut_vs_direct_chain(self):
        lin = spaces.di_decode(np.random.default_rng(0).uniform(0.25, 0.75, (400, 3)))
        direct = spaces.di_encode(stocks.family_balance(PRT)(self.chain.display_linear(self.chain.print_log_exposure(lin))))
        err = np.abs(direct - render(px(lin), self.luts, Params())[:, :3])
        print(f"\n[measure] 250D+2383 LUT vs direct chain, DI: max={err.max():.2e} p99={np.quantile(err, 0.99):.2e}")
        self.assertLess(err.max(), 5e-3)

    @unittest.skipUnless(CLI.exists(), "fe_cli not built")
    def test_cuda_parity(self):
        img = px(spaces.di_decode(np.random.default_rng(2).uniform(0.0, 1.0, (37 * 23, 3)))).reshape(23, 37, 4)
        with tempfile.TemporaryDirectory() as d:
            for p in (Params(), Params(push_pull=0.8, print_log_offset=(0.1, -0.05, 0.02), out_space="rec709_g24"),
                      Params(grain_amount=1.0, grain_sigma_px=1.5, grain_seed=5, frame=2)):
                gpu = run_cuda(img, self.luts, p, d)
                ora = render(img, self.luts, p)
                err = np.max(np.abs(gpu - ora) / np.maximum(1.0, np.abs(ora)))
                print(f"\n[measure] 250D+2383 CUDA vs oracle: {err:.2e}")
                self.assertLess(err, 1e-4)



class LabScan(unittest.TestCase):
    def test_every_stock_has_a_fitted_scan_look(self):
        from bake_luts import BW_NEGATIVES, NEGATIVES, REVERSALS, STILL_NEGATIVES

        for neg, _ in NEGATIVES + STILL_NEGATIVES + BW_NEGATIVES + REVERSALS:
            look = stocks.load(neg)["scan"]
            self.assertIn("Lomography", look["note"], neg)
            self.assertIn("not manufacturer data", look["note"], neg)
            self.assertTrue(0.88 <= look["saturation"] <= 1.15 and 0.92 <= look["contrast"] <= 1.1, neg)
            for k in ("shadow_tint", "highlight_tint", "mid_tint"):
                self.assertTrue(all(0.93 <= v <= 1.07 for v in look[k]), (neg, k))
            fn = stocks.scan_look(neg)
            out = fn(np.array([[0.4, 0.05, 0.05], [0.0, 0.0, 0.0]]))
            self.assertTrue(np.all(np.isfinite(out)) and np.all(out >= 0.0), neg)
            self.assertLess(np.abs(out[1]).max(), 1e-9, neg)

    def test_scan_luts_baked_for_every_kind(self):
        for name in ("kodak-gold200-endura", PROFILE.name, "ilford-hp5plus-ilford-multigrade-iv-rc", "fuji-velvia100"):
            prof = PROFILE.parent / name
            if not (prof / "profile.json").exists():
                self.skipTest("profiles not baked")
            meta = json.loads((prof / "profile.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["scan_print"], "print_scan.bin", name)
            n = meta["lut_size"]
            a = np.fromfile(prof / "print.bin", np.float32).reshape(n, n, n, 3)
            b = np.fromfile(prof / "print_scan.bin", np.float32).reshape(n, n, n, 3)
            c = np.fromfile(prof / "print_bal_scan.bin", np.float32).reshape(n, n, n, 3)
            self.assertGreater(np.abs(a - b).max(), 1e-3, name)
            self.assertTrue(np.all(np.isfinite(c)), name)


class Reversal(unittest.TestCase):
    def test_every_reversal_stock(self):
        from bake_luts import REVERSALS

        for neg, _ in REVERSALS:
            with self.subTest(neg=neg):
                fs = stocks.system(neg, None)
                ch = stocks.make_chain(fs)
                run = lambda lin: ch.display_linear(ch.print_log_exposure(np.asarray(lin, np.float64)))
                ramp = run((0.18 * 2.0 ** np.arange(-6.0, 3.01, 0.5))[:, None].repeat(3, 1))
                self.assertTrue(np.all(np.diff(ramp.mean(1)) > -1e-6))
                self.assertLessEqual(ramp.max(), 1.0 + 1e-6)
                hue = [int(np.argmax(run(np.where(np.eye(3)[i] > 0, 0.4, 0.05)[None, :])[0])) for i in range(3)]
                self.assertEqual(hue, [0, 1, 2])
                grey = stocks.grey_balance(ch)(run(np.full((1, 3), 0.18)))[0]
                self.assertLess(np.abs(grey / grey[1] - 1.0).max(), 1e-3)
                self.assertTrue(np.all(stocks.grain_table(fs, neg) > 0))
                print(f"\n[measure] {neg}: grey {run(np.full((1, 3), 0.18))[0].round(4)}, white/grey {ramp.max() / run(np.full((1, 3), 0.18))[0].mean():.2f}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
