from dataclasses import dataclass

import numpy as np

from .spectral import MID_GREY, WL, scene_spectra
from .stocks import GRAIN_FALLBACK_STOCK, GRAIN_REFERENCE_SIGMA, TabulatedCurve, load

SPEED_DENSITY = 0.1
GREY_STOPS_ABOVE_SPEED = 4.0
PAPER_GREY_AIM = 0.70


@dataclass(frozen=True)
class BWPaper:
    x: np.ndarray
    d: np.ndarray
    x0: float
    label: str

    @property
    def curves(self):
        c = TabulatedCurve(self.x, self.d, self.x0)
        return (c, c, c)


@dataclass(frozen=True)
class BWSystem:
    sensitivity: np.ndarray
    curve: TabulatedCurve
    print_: BWPaper
    label: str


def _sensitivity(data):
    if "spectral_sensitivity" not in data:
        data = load(data["spectral_fallback"])
    ch = data["spectral_sensitivity"]["channels"]["k"]
    w = np.asarray(ch["wavelength"], float)
    v = np.power(10.0, np.asarray(ch["value"], float))
    return np.where((WL >= w[0]) & (WL <= w[-1]), np.interp(WL, w, v), 0.0)


def negative(stock_id, contrast=1.0):
    data = load(stock_id)
    ch = data["sensitometry"]["channels"]["k"]
    x, d = np.asarray(ch["x"], float), np.asarray(ch["density"], float)
    speed = float(np.interp(d[0] + SPEED_DENSITY, d, x))
    grey = speed + GREY_STOPS_ABOVE_SPEED * np.log10(2.0)
    return TabulatedCurve(x - grey, d - d[0], 0.0, contrast), _sensitivity(data), data["label"]


def paper(paper_id, grade=None):
    data = load(paper_id)
    g = data["grades"][grade or data["default_grade"]]
    x, d = np.asarray(g["x"], float), np.asarray(g["density"], float)
    return BWPaper(x, d, float(np.interp(d[0] + PAPER_GREY_AIM, d, x)), data["label"])


def system(negative_id, paper_id, contrast=1.0):
    curve, sens, label = negative(negative_id, contrast)
    return BWSystem(sens, curve, paper(paper_id), label)


class BWChain:
    def __init__(self, system, scene_light=None, design_light=None):
        self.s = system
        self.h_grey = float(scene_spectra(np.full((1, 3), MID_GREY))[0] @ system.sensitivity)
        self.d_grey = float(system.curve.density(0.0))
        self.dmin = float(system.print_.d[0])

    def print_log_exposure(self, lin_dwg, coefficients=None, curves=None):
        h = scene_spectra(lin_dwg, coefficients) @ self.s.sensitivity / self.h_grey
        d = self.s.curve.density(np.log10(np.maximum(h, 2.0 ** -20)))
        e = -(d - self.d_grey) + self.s.print_.x0
        return np.repeat(e[:, None], 3, axis=1)

    def display_linear(self, log_e_print):
        e = np.asarray(log_e_print, np.float64).reshape(-1, 3)[:, 0]
        dp = np.interp(e, self.s.print_.x, self.s.print_.d)
        y = np.power(10.0, -(dp - self.dmin))
        return np.repeat(y[:, None], 3, axis=1)


GRAIN_DELTA = 1e-3


def grain_table(fs, negative_id):
    from . import grain

    data = load(negative_id)
    chain = BWChain(fs)
    u = grain.table_axis()
    d = fs.curve.density(u)
    e0 = -(d - chain.d_grey) + fs.print_.x0
    y = lambda e: np.interp(e, fs.print_.x, fs.print_.d)
    gamma = np.abs(y(e0 - GRAIN_DELTA) - y(e0)) / GRAIN_DELTA
    if "granularity" in data:
        g = data["granularity"]["channels"]["k"]
        sigma = np.power(10.0, np.interp(u, g["x"], g["log10_sigma"]))
    else:
        g = load(GRAIN_FALLBACK_STOCK)["granularity"]["channels"]["g"]
        sigma = np.power(10.0, np.interp(u, g["x"], g["log10_sigma"]))
        if "rms_granularity" in data:
            sigma = sigma * data["rms_granularity"] / 1000.0 / GRAIN_REFERENCE_SIGMA
    out = sigma / (np.interp(0.0, u, gamma) * GRAIN_REFERENCE_SIGMA)
    return np.tile(out, (3, 1))
