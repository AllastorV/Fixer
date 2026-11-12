import numpy as np
import colour

DI_A, DI_B, DI_C, DI_M = 0.0075, 7.0, 0.07329248, 10.44426855
DI_LIN_CUT, DI_LOG_CUT = 0.00262409, 0.02740668

CCT_LIN_CUT, CCT_LOG_CUT = 0.0078125, 0.155251141552511
CCT_A, CCT_B = 10.5402377416545, 0.0729055341958355

F_MAX = 65504.0


def di_encode(x):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x <= DI_LIN_CUT, DI_M * x, DI_C * (np.log2(np.maximum(x, DI_LIN_CUT) + DI_A) + DI_B))


def di_decode(y):
    y = np.asarray(y, dtype=np.float64)
    return np.where(y <= DI_LOG_CUT, y / DI_M, np.exp2(np.maximum(y, DI_LOG_CUT) / DI_C - DI_B) - DI_A)


def acescct_encode(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= CCT_LIN_CUT, CCT_A * x + CCT_B, (np.log2(np.maximum(x, CCT_LIN_CUT)) + 9.72) / 17.52)


def acescct_decode(y):
    y = np.asarray(y, dtype=np.float64)
    return np.where(y <= CCT_LOG_CUT, (y - CCT_B) / CCT_A, np.exp2(np.maximum(y, CCT_LOG_CUT) * 17.52 - 9.72))


def g24_encode(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.abs(x) ** (1 / 2.4)


def g24_decode(y):
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.abs(y) ** 2.4


def _identity(x):
    return np.asarray(x, dtype=np.float64)


def _matrix(src, dst):
    if src == dst:
        return np.eye(3)
    return colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES[src], colour.RGB_COLOURSPACES[dst], chromatic_adaptation_transform="Bradford"
    )


SPACES = {
    "dwg_di": ("DaVinci Wide Gamut", di_encode, di_decode),
    "dwg_linear": ("DaVinci Wide Gamut", _identity, _identity),
    "acescct": ("ACEScct", acescct_encode, acescct_decode),
    "rec709_g24": ("ITU-R BT.709", g24_encode, g24_decode),
}

WORKING = "DaVinci Wide Gamut"


def to_working(rgb, space):
    gamut, _, decode = SPACES[space]
    return sanitize(decode(rgb)) @ _matrix(gamut, WORKING).T


def from_working(rgb, space):
    gamut, encode, _ = SPACES[space]
    return encode(np.asarray(rgb, dtype=np.float64) @ _matrix(WORKING, gamut).T)


def sanitize(rgb):
    return np.nan_to_num(np.asarray(rgb, dtype=np.float64), nan=0.0, posinf=F_MAX, neginf=-F_MAX)
