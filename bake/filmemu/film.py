from dataclasses import dataclass

import numpy as np

LIN_FLOOR = 2.0 ** -20
MID_GREY = 0.18
PRINTER_POINT = 0.025


@dataclass(frozen=True)
class Curve:
    d_min: float
    d_max: float
    k: float
    x0: float

    def density(self, log_h):
        return self.d_min + (self.d_max - self.d_min) / (1.0 + np.exp(-self.k * (log_h - self.x0)))

    def slope(self, log_h):
        s = 1.0 / (1.0 + np.exp(-self.k * (log_h - self.x0)))
        return (self.d_max - self.d_min) * self.k * s * (1.0 - s)


@dataclass(frozen=True)
class SyntheticProfile:
    negative: tuple
    print_: tuple
    label: str = "synthetic-test"


def log_exposure(lin):
    return np.log10(np.maximum(np.asarray(lin, dtype=np.float64), LIN_FLOOR) / MID_GREY)


def render_film(lin, profile, printer_points=(0.0, 0.0, 0.0)):
    log_h = log_exposure(lin)
    neg = np.stack([c.density(log_h[..., i]) for i, c in enumerate(profile.negative)], axis=-1)
    neg_mid = np.array([c.density(0.0) for c in profile.negative])
    p0 = np.array([c.x0 for c in profile.print_]) + neg_mid
    log_h_print = -neg + p0 + PRINTER_POINT * np.asarray(printer_points, dtype=np.float64)
    dp = np.stack([c.density(log_h_print[..., i]) for i, c in enumerate(profile.print_)], axis=-1)
    dp_mid = np.array([c.density(c.x0) for c in profile.print_])
    return MID_GREY * np.power(10.0, -(dp - dp_mid))


def synthetic_slots():
    def make(contrast):
        neg = tuple(Curve(0.2, 2.2, 1.6 * contrast, 0.0) for _ in range(3))
        prt = tuple(Curve(0.05, 3.6, 2.4, 0.0) for _ in range(3))
        return SyntheticProfile(neg, prt)

    return {"under": make(0.85), "normal": make(1.0), "over": make(1.15)}
