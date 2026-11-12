# FIXER

![FIXER](docs/brand/fixer-banner-photo@2x.png)

FIXER is a free film emulation OFX plugin for DaVinci Resolve (Windows, NVIDIA CUDA). 36 film stocks with their print or paper, built from manufacturers' published technical data: Kodak Vision3, Portra, Ektar, Gold, UltraMax, Tri-X, T-Max, Double-X, Plus-X, Ektachrome, Kodachrome; Fujifilm Velvia, Provia, C200, Superia, Pro 400H, Eterna Vivid, Acros II; Ilford HP5, FP4, Delta, Pan F, XP2.

User guide: [docs/USER-GUIDE.md](docs/USER-GUIDE.md).

## Build

Requirements: VS 2022 Build Tools (C++), CUDA 13.5, CMake ≥ 3.26, Ninja, Python 3.12 with `numpy`, `scipy`, `colour-science`.

```bat
build.bat
```

Output: `build\release\Fixer.ofx.bundle` (plugin + baked LUTs in `Contents\Resources`).

## Install

`install.bat` (asks for administrator) copies the bundle to `C:\Program Files\Common Files\OFX\Plugins`. Restart Resolve. Resolve must use CUDA as GPU processing mode.

## Tests

```bat
cd bake
py -3.12 -m unittest discover -s tests -p "test_*.py"
```

Tools in `bake/tools`: `resolve_check.py` / `resolve_run.py` (pixel match of Resolve renders against the Python reference, every stock), `resolve_params.py` (the same for exposure, push/pull, impact, glow, printer lights, colour spaces and grain), `mode_check.py` (each stock in datasheet, Lab scan and grey-balance modes with grain and halation), `scan_fit.py` (Lab scan fit).

Numeric rules: [docs/W01-numeric-contract.md](docs/W01-numeric-contract.md).

## Roadmap

- OpenCL support for AMD and Intel GPUs on Windows.
- Metal support for macOS.
- 14 more film stocks, 50 in total.

## License

Apache-2.0. Not affiliated with any film manufacturer or other film-emulation product; see `NOTICE`.
