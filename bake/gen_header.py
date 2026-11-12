import pathlib
import sys
import warnings

warnings.simplefilter("ignore")

from filmemu import spaces

ORDER = ["dwg_di", "dwg_linear", "acescct", "rec709_g24"]


def fmt(m):
    return "{" + ", ".join(f"{float(v):.9e}f" for v in m.reshape(-1)) + "}"


def main(out):
    to_w = [spaces._matrix(spaces.SPACES[s][0], spaces.WORKING) for s in ORDER]
    from_w = [spaces._matrix(spaces.WORKING, spaces.SPACES[s][0]) for s in ORDER]
    lines = [
        "#pragma once",
        "namespace fe {",
        "enum Space : int { " + ", ".join(f"kSpace_{s} = {i}" for i, s in enumerate(ORDER)) + ", kSpaceCount };",
        "constexpr float kToWorking[kSpaceCount][9] = {" + ", ".join(fmt(m) for m in to_w) + "};",
        "constexpr float kFromWorking[kSpaceCount][9] = {" + ", ".join(fmt(m) for m in from_w) + "};",
        "}",
    ]
    pathlib.Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
