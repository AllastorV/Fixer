import multiprocessing as mp
import pathlib
import sys
import time
import warnings

import numpy as np

MAX_ERROR = 1e-4


def _setup():
    warnings.simplefilter("ignore")
    import colour

    cs = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(colour.SpectralShape(360, 780, 5))
    ill = colour.SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
    return colour, cs, cmfs, ill


def solve_chunk(units):
    colour, cs, cmfs, ill = _setup()
    from colour.recovery import find_coefficients_Jakob2019

    out = np.zeros((len(units), 3))
    errs = np.zeros(len(units))
    c0 = np.zeros(3)
    for i, u in enumerate(units):
        xyz = colour.RGB_to_XYZ(u, cs)
        c, e = find_coefficients_Jakob2019(xyz, cmfs, ill, coefficients_0=c0, max_error=MAX_ERROR)
        if not np.all(np.isfinite(c)):
            c, e = find_coefficients_Jakob2019(xyz, cmfs, ill, max_error=MAX_ERROR)
        out[i], errs[i] = c, e
        c0 = c
    return out, errs


def unit_nodes(n):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from filmemu import spaces, spectral

    axis = np.linspace(0.0, 1.0, n)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    lin = spaces.di_decode(np.stack([r, g, b], axis=-1).reshape(-1, 3))
    return spectral.upsampling_units(lin)[0]


def main(n, out_path):
    units = unit_nodes(n)
    key = np.round(units, 7)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    order = np.lexsort((uniq[:, 2], uniq[:, 1], uniq[:, 0]))
    chunks = np.array_split(uniq[order], mp.cpu_count() * 4)
    t = time.time()
    with mp.Pool() as pool:
        parts = pool.map(solve_chunk, chunks)
    coef_sorted = np.concatenate([p[0] for p in parts])
    err_sorted = np.concatenate([p[1] for p in parts])
    coef = np.empty_like(coef_sorted)
    coef[order] = coef_sorted
    err = np.empty_like(err_sorted)
    err[order] = err_sorted
    np.savez_compressed(out_path, size=n, units=uniq, coefficients=coef, error=err, inverse=inverse.reshape(-1))
    print(f"nodes={len(units)} unique={len(uniq)} sec={time.time() - t:.0f} err_max={err.max():.2e} err_p99={np.quantile(err, 0.99):.2e}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
