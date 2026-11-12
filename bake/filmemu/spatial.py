import numpy as np
from scipy.ndimage import convolve1d

HALATION_WEIGHTS = np.array([1.0, 0.25, 0.0])
HALATION_NARROW = 0.3
HALATION_MIX = 0.5
BLOOM_RADIUS_REF = 15.0
MAX_SIGMA_PX = 120.0


def kernel(sigma):
    sigma = float(np.float32(min(max(sigma, 0.5), MAX_SIGMA_PX)))
    r = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum(), r


def blur(img, sigma):
    k, _ = kernel(sigma)
    out = convolve1d(img, k, axis=1, mode="nearest")
    return convolve1d(out, k, axis=0, mode="nearest")


def margin(sigma):
    return kernel(sigma)[1]


def halation_psf(img, radius_px):
    return HALATION_MIX * blur(img, HALATION_NARROW * radius_px) + (1.0 - HALATION_MIX) * blur(img, radius_px)


def halation(lin, amount, radius_px):
    return lin + amount * HALATION_WEIGHTS * (halation_psf(lin, radius_px) - lin)


def bloom(lin, amount, radius_px):
    return lin + amount * (blur(lin, radius_px) - lin)
