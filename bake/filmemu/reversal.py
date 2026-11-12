from dataclasses import dataclass

import numpy as np
import colour

from . import spectral
from .spectral import MID_GREY, PRINT_WHITE_DEPTH, SHAPE, WL, XYZ_TO_DWG, CMFS, Stock, _norm, scene_spectra, luminance

GREY_AIM_DENSITY = 1.0


@dataclass(frozen=True)
class ReversalCurve:
    x: np.ndarray
    d: np.ndarray
    x0: float
    dmin: float
    gain: float = 1.0

    def density(self, log_h):
        return self.dmin + self.gain * (np.interp(log_h, self.x, self.d) - self.dmin)


@dataclass(frozen=True)
class ReversalSystem:
    print_: Stock
    projector: np.ndarray
    label: str


def system(stock_id, contrast=1.0):
    from . import stocks

    data = stocks.load(stock_id)
    dyes, sens = stocks._stock(data, "decline")
    eff = stocks.effective_dyes(dyes)
    curves = []
    for k, c in enumerate("rgb"):
        ch = data["sensitometry"]["channels"][c]
        x, d = np.asarray(ch["x"], float), np.asarray(ch["density"], float)
        order = np.argsort(x)
        x, d = x[order], np.minimum.accumulate(d[order])
        dmin = float(d.min())
        aim = dmin + GREY_AIM_DENSITY
        x0 = float(np.interp(-aim, -d, x))
        curves.append(ReversalCurve(x, d, x0, dmin, contrast))
    view = data.get("viewing", "D50")
    if view not in colour.SDS_ILLUMINANTS:
        view = "D50"
    sd = colour.SDS_ILLUMINANTS[view].copy().align(SHAPE, extrapolator_kwargs={"method": "Constant"}).values
    stock = Stock(sens, eff, np.zeros_like(WL, dtype=np.float64), tuple(curves), data["label"])
    return ReversalSystem(stock, sd / sd.max(), data["label"])


def is_reversal(stock_id):
    from . import stocks

    return stocks.load(stock_id).get("role") == "reversal"


class ReversalChain(spectral.Chain):
    def __init__(self, system, scene_light=None, design_light=None):
        self.s = system
        self.scene_light = scene_light
        dwg = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
        white = _norm(system.projector[None, :], CMFS.T)[0]
        self.cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(white / white[1], colour.xy_to_XYZ(dwg.whitepoint), "Bradford")
        grey = scene_spectra(np.full((1, 3), MID_GREY), illuminant=design_light)
        self.e_grey = _norm(grey, system.print_.sensitivity)[0]
        self.light_gain = 1.0
        if scene_light is not None:
            lit = scene_spectra(np.full((1, 3), MID_GREY), illuminant=scene_light)
            self.light_gain = float(luminance(grey)[0] / luminance(lit)[0])
        dp_mid = np.array([c.density(c.x0) for c in system.print_.curves])
        rgb_mid = self.display_xyz(dp_mid[None, :])[0] @ XYZ_TO_DWG.T
        self.display_gain = MID_GREY / rgb_mid
        dp_white = np.array([c.density(c.x0 + PRINT_WHITE_DEPTH) for c in system.print_.curves])
        rgb_white = (self.display_xyz(dp_white[None, :])[0] @ XYZ_TO_DWG.T) * self.display_gain
        self.display_gain = self.display_gain / rgb_white.max()

    def print_log_exposure(self, lin_dwg, coefficients=None, curves=None):
        e = _norm(scene_spectra(lin_dwg, coefficients, self.scene_light) * self.light_gain, self.s.print_.sensitivity) / self.e_grey
        return np.log10(np.maximum(e, 2.0 ** -30)) + np.array([c.x0 for c in self.s.print_.curves])


def grain_table(fs, stock_id):
    from . import grain, stocks

    data = stocks.load(stock_id)
    g = fs.print_.curves[1]
    gamma = abs(float(g.density(g.x0 + 0.01) - g.density(g.x0 - 0.01))) / 0.02
    level = data["rms_granularity"] / 1000.0 / stocks.GRAIN_REFERENCE_SIGMA if "rms_granularity" in data else 1.0
    return np.full((3, grain.TABLE_SIZE), level / max(gamma, 1e-3))
