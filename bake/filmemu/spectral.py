import pathlib
from dataclasses import dataclass, field

import numpy as np
import colour

from .film import Curve
from .spaces import _matrix

SHAPE = colour.SpectralShape(380, 780, 1)
WL = SHAPE.wavelengths
MID_GREY = 0.18
PRINTER_POINT = 0.025
PRINT_WHITE_DEPTH = 3.0
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SHAPE).values
D65 = colour.SDS_ILLUMINANTS["D65"].copy().align(SHAPE).values
XYZ_TO_DWG = np.linalg.inv(colour.RGB_COLOURSPACES["DaVinci Wide Gamut"].matrix_RGB_to_XYZ)
DWG_TO_REC2020 = _matrix("DaVinci Wide Gamut", "ITU-R BT.2020")


def gaussian(center, width, peak=1.0):
    return peak * np.exp(-0.5 * ((WL - center) / width) ** 2)


@dataclass(frozen=True)
class Stock:
    sensitivity: np.ndarray
    dyes: np.ndarray
    base: np.ndarray
    curves: tuple
    label: str = "synthetic"


@dataclass(frozen=True)
class FilmSystem:
    negative: Stock
    print_: Stock
    printer_light: np.ndarray
    projector: np.ndarray = field(default_factory=lambda: D65.copy())


def synthetic_system(contrast=1.0):
    neg = Stock(
        sensitivity=np.stack([gaussian(640, 28), gaussian(545, 26), gaussian(450, 24)]),
        dyes=np.stack([gaussian(690, 45) + gaussian(440, 30, 0.08), gaussian(550, 40) + gaussian(440, 30, 0.12),
                       gaussian(445, 35)]),
        base=0.05 + gaussian(450, 60, 0.25),
        curves=tuple(Curve(0.0, 2.0, 1.6 * contrast, 0.0) for _ in range(3)),
    )
    prt = Stock(
        sensitivity=np.stack([gaussian(680, 18), gaussian(550, 18), gaussian(460, 18)]),
        dyes=np.stack([gaussian(670, 40), gaussian(550, 38), gaussian(440, 32)]),
        base=np.full_like(WL, 0.03, dtype=np.float64),
        curves=tuple(Curve(0.05, 3.4, 2.6, 0.0) for _ in range(3)),
    )
    light = gaussian(660, 14) + gaussian(545, 14) + gaussian(455, 14)
    return FilmSystem(neg, prt, light)


def sd_from_coefficients(c):
    lam = WL[None, :]
    x = c[:, 0:1] * lam * lam + c[:, 1:2] * lam + c[:, 2:3]
    return 0.5 + x / (2.0 * np.sqrt(1.0 + x * x))


GAMUT_THRESHOLD = 0.7
GAMUT_TARGET = 0.95
UNIT_BRIGHTNESS = 0.5


def _compress(d):
    t, span = GAMUT_THRESHOLD, GAMUT_TARGET - GAMUT_THRESHOLD
    return np.where(d <= t, d, t + span * (1.0 - np.exp(-np.maximum(d - t, 0.0) / span)))


def upsampling_units(lin_dwg):
    rgb = np.asarray(lin_dwg, np.float64).reshape(-1, 3) @ DWG_TO_REC2020.T
    ach = rgb.max(axis=1, keepdims=True)
    pos = ach > 0
    safe = np.where(pos, ach, 1.0)
    d = np.where(pos, (ach - rgb) / safe, 0.0)
    rgb_c = ach - _compress(d) * ach
    unit = np.where(pos, rgb_c / safe * UNIT_BRIGHTNESS, 0.0)
    scale = np.where(pos, ach / UNIT_BRIGHTNESS, 0.0)
    return unit, scale


_NODES = {}


def node_table(size):
    if size not in _NODES:
        d = np.load(DATA / f"jakob_nodes_{size}.npz")
        _NODES[size] = d["coefficients"][d["inverse"]]
    return _NODES[size]


def solve_coefficients(unit):
    from colour.recovery import find_coefficients_Jakob2019

    cs = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(colour.SpectralShape(360, 780, 5))
    ill = colour.SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
    out = np.zeros((len(unit), 3))
    c0 = np.zeros(3)
    for i, u in enumerate(unit):
        out[i] = c0 = find_coefficients_Jakob2019(colour.RGB_to_XYZ(u, cs), cmfs, ill, coefficients_0=c0, max_error=1e-4)[0]
    return out


def scene_spectra(lin_dwg, coefficients=None, illuminant=None):
    unit, scale = upsampling_units(lin_dwg)
    c = solve_coefficients(unit) if coefficients is None else coefficients
    return sd_from_coefficients(c) * scale * (D65 if illuminant is None else illuminant)[None, :]


def luminance(spectrum):
    return np.asarray(spectrum, np.float64) @ CMFS[:, 1]


def _norm(spectrum, response):
    return np.einsum("nl,kl->nk", spectrum, response)


class Chain:
    def __init__(self, system, scene_light=None, design_light=None):
        self.s = system
        self.scene_light = scene_light
        dwg = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
        white = _norm(system.projector[None, :], CMFS.T)[0]
        self.cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(white / white[1], colour.xy_to_XYZ(dwg.whitepoint), "Bradford")
        grey = scene_spectra(np.full((1, 3), MID_GREY), illuminant=design_light)
        self.h_grey = _norm(grey, system.negative.sensitivity)[0]
        self.light_gain = 1.0
        if scene_light is not None:
            lit = scene_spectra(np.full((1, 3), MID_GREY), illuminant=scene_light)
            self.light_gain = float(luminance(grey)[0] / luminance(lit)[0])
        neg_mid = np.array([c.density(0.0) for c in system.negative.curves])
        t_mid = self.neg_transmittance(neg_mid[None, :])
        self.e_grey = _norm(t_mid * system.printer_light[None, :], system.print_.sensitivity)[0]
        dp_mid = np.array([c.density(c.x0) for c in system.print_.curves])
        rgb_mid = self.display_xyz(dp_mid[None, :])[0] @ XYZ_TO_DWG.T
        self.display_gain = MID_GREY / rgb_mid
        dp_white = np.array([c.density(c.x0 - PRINT_WHITE_DEPTH) for c in system.print_.curves])
        rgb_white = (self.display_xyz(dp_white[None, :])[0] @ XYZ_TO_DWG.T) * self.display_gain
        self.display_gain = self.display_gain / rgb_white.max()

    def neg_transmittance(self, dens):
        return np.power(10.0, -(self.s.negative.base[None, :] + dens @ self.s.negative.dyes))

    def negative_log_exposure(self, lin_dwg, coefficients=None):
        h = _norm(scene_spectra(lin_dwg, coefficients, self.scene_light) * self.light_gain, self.s.negative.sensitivity) / self.h_grey
        return np.log10(np.maximum(h, 2.0 ** -20))

    def print_log_exposure(self, lin_dwg, coefficients=None, curves=None):
        curves = curves or self.s.negative.curves
        log_h = self.negative_log_exposure(lin_dwg, coefficients)
        dens = np.stack([c.density(log_h[:, i]) for i, c in enumerate(curves)], axis=-1)
        e = _norm(self.neg_transmittance(dens) * self.s.printer_light[None, :], self.s.print_.sensitivity) / self.e_grey
        return np.log10(np.maximum(e, 2.0 ** -30)) + np.array([c.x0 for c in self.s.print_.curves])

    def display_xyz(self, print_dens):
        t = np.power(10.0, -(self.s.print_.base[None, :] + print_dens @ self.s.print_.dyes))
        return _norm(t * self.s.projector[None, :], CMFS.T) @ self.cat.T

    def display_linear(self, log_e_print):
        log_e = np.asarray(log_e_print, np.float64).reshape(-1, 3)
        dp = np.stack([c.density(log_e[:, i]) for i, c in enumerate(self.s.print_.curves)], axis=-1)
        return (self.display_xyz(dp) @ XYZ_TO_DWG.T) * self.display_gain
