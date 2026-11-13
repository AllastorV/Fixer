import os

os.environ["FIXER_NO_PROGRESS"] = "1"
import pathlib
import sys
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import match_visual as mv

READY = (mv.HOST.exists() and mv.PLUGIN.exists() and all((mv.HDRI / f"{n}.npy").exists() for n in mv.SCENES))
REFS_READY = mv.REF_DIR.is_dir() and all((mv.REF_DIR / p).exists() for _, p in mv.REFS)
MEDIAN_CLOSED = 0.6
DESAT_RATIO = 0.4
IDENTITY_DE = 1.0
CREATE_MS = 2000.0


def report(lines):
    print("", *("[measure] " + l for l in lines), sep="\n")


@unittest.skipUnless(REFS_READY, f"reference folder missing: {mv.REF_DIR}")
@unittest.skipUnless(READY, "plugin, host or HDRI arrays missing")
class RealReferenceMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = mv.run()
        cls.film = [e for e in cls.m["pairs"] if e["source"] == "film"]

    def test_0_median_gap_closed(self):
        lines = []
        for src in mv.SOURCES:
            p = [e for e in self.m["pairs"] if e["source"] == src]
            med = {k: float(np.median([e["gap_closed"][k] for e in p])) for k in ("total", "L", "chroma", "ab")}
            lines.append(f"{src}: median gap closed total {med['total']:.3f}, L {med['L']:.3f}, chroma {med['chroma']:.3f}, ab {med['ab']:.3f}")
            lines += [f"  {e['frame']}/{e['style']}: {e['gap_closed']['total']:.3f}" for e in p]
        report(lines)
        for src in mv.SOURCES:
            with self.subTest(source=src):
                self.assertGreaterEqual(float(np.median([e["gap_closed"]["total"] for e in self.m["pairs"] if e["source"] == src])), MEDIAN_CLOSED)

    def test_1_desaturated_chroma(self):
        for e in self.m["desaturated_unprotected"]:
            with self.subTest(frame=e["frame"], source=e["source"]):
                self.assertTrue(e["finite"])
                self.assertLessEqual(e["chroma_ratio"], DESAT_RATIO)

    def test_2_blue_sky_keeps_mid_and_highlight_b(self):
        lines = []
        for e in (e for e in self.film if e["style"] == "blue sky warm ground"):
            z = e["zone_ab"]
            for k in ("mid", "hl"):
                if z["before"][k] is None or z["result"][k] is None:
                    lines.append(f"{e['frame']}/{k}: zone empty, skipped")
                    continue
                b0, r, b1 = z["before"][k][1], (z["ref"][k] or [None, None])[1], z["result"][k][1]
                lines.append(f"{e['frame']}/{k}: b* before {b0:.2f}, ref {r}, result {b1:.2f}, {k} gains {np.round(e['fit']['tint'][k], 3).tolist()}")
                with self.subTest(frame=e["frame"], zone=k):
                    self.assertGreaterEqual(b1, b0, lines[-1])
        report(lines)

    def test_3_identity(self):
        lines = [f"{e['frame']}/{e['source']}: mean dE00 vs before {e['de00_vs_before']:.3f}" for e in self.m["identity"]]
        report(lines)
        for e in self.m["identity"]:
            with self.subTest(frame=e["frame"]):
                self.assertLess(e["de00_vs_before"], IDENTITY_DE)

    def test_4_finite(self):
        bad = [f"{e['frame']}/{e['source']}/{e.get('style', 'identity')}" for e in self.m["pairs"] + self.m["identity"] if not e["finite"]]
        self.assertEqual(bad, [])

    def test_5_create_grade_time(self):
        ms = [(f"{e['frame']}/{e['source']}/{e.get('style', 'identity')}", e["create_ms"]) for e in self.m["pairs"] + self.m["identity"]]
        self.assertTrue(all(t is not None for _, t in ms), [n for n, t in ms if t is None])
        worst = max(ms, key=lambda x: x[1][1])
        report([f"create grade 1920x1080 over {len(ms)} runs: max {worst[1][1]:.1f} ms ({worst[0]}), median {np.median([t[1] for _, t in ms]):.1f} ms"])
        for name, t in ms:
            self.assertLess(t[1], CREATE_MS, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
