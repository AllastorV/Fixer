# FIXER — user guide

Free OFX plugin for DaVinci Resolve on Windows (NVIDIA, CUDA). Each profile is a film stock with its print or paper, built from the manufacturer's published technical data.

## Install

1. Close Resolve.
2. Run `install.bat` and accept the administrator prompt. It copies `Fixer.ofx.bundle` to `C:\Program Files\Common Files\OFX\Plugins`.
3. In Resolve: Preferences → Memory and GPU → GPU processing mode **CUDA**. There is no CPU or OpenCL path; with OpenCL Resolve shows a generic "Open FX Error".
4. Color page → Effects → OpenFX → FIXER → **FIXER**. A new node starts on Kodak Vision3 250D with its 2383 print, Lab scan on, grain and halation at their defaults.

## The one rule: the plugin is the display transform

FIXER ends at a finished picture: the print (or paper, or slide) as it looks on screen, print white = display 1.0. Nothing after it may tone-map again. So:

- put it on the **last** node of the clip, after all scene grading;
- make sure Resolve adds **no second tone mapping** (Output DRT / ODT off, see below).

If the picture looks flat, dim or double-contrasted, a second tone mapping is on.

## Node trees

### A. DaVinci YRGB (not colour managed) — recommended

| Node | What | Settings |
|---|---|---|
| 1 | Color Space Transform | Input: your camera's gamut/gamma. Output: DaVinci Wide Gamut / DaVinci Intermediate. Tone mapping: None. |
| 2 … n | Your scene grade | Exposure, white balance, contrast in DWG/DI. |
| last | FIXER | Colour spaces → Input: DWG / Intermediate. Output: Rec.709 / Gamma 2.4. |

Deliver as Rec.709 Gamma 2.4. Footage that is already DWG/DI needs no node 1.

### B. DaVinci YRGB Color Managed (RCM)

Project Settings → Color Management: timeline colour space DaVinci WG/Intermediate, output colour space Rec.709 Gamma 2.4, **output tone mapping / DRT: None**. Input colour space per clip as usual.

FIXER on the last node: Input DWG / Intermediate, Output DWG / Intermediate. Resolve then converts the finished picture to the output space without tone mapping.

### C. ACES (ACEScct)

Project Settings: ACEScct, ACES Output Transform **No Output Transform**. FIXER on the last node: Input ACEScct, Output Rec.709 / Gamma 2.4. With an ACES output transform on, the picture is tone-mapped twice.

## Controls

| Control | Meaning |
|---|---|
| Film stock | Stock with its print/paper (cinema stocks → Kodak 2383, colour still negatives → Kodak Endura paper, black-and-white → Ilford Multigrade IV RC, slides → viewed directly). |
| Grey balance | Off: the stock's own colour cast, as in the data. On: neutral greys at every exposure. |
| Lab scan look | On (default): colours of real scans of this film (fitted to community scans). Off: the manufacturer data only. Grain, halation, exposure and printer lights are the same either way. |
| Exposure (stops) | Film exposure. Use this, not node gain, to over/underexpose the film: the stock answers with its own toe and shoulder. |
| Push / Pull | Development push or pull, in stops. |
| Impact | Mix with the original image (0 = plugin off). |
| Grain: Amount, Size, Rate, Seed | Grain from the stock's published granularity. Size is relative to a 1080-line frame, so it looks the same at any resolution and with proxies. Rate: how often the pattern changes (every frame = like real film). |
| Halation, radius | Red glow around bright highlights (from light reflected off the film base). |
| Bloom | Soft glow over bright areas. |
| Glow quality | Full: halation and bloom at full resolution. Fast: at half resolution, for smoother playback and about half the GPU memory; grain and the film response stay full resolution. Use Full for the final render. |
| Printer lights | Print exposure and R/G/B in printer points, lab convention: +R points → denser red → more cyan on screen. |
| Colour spaces | Input and output encoding of this node (see the node trees above). |

## Good to know

- Grade exposure and white balance **before** the plugin, in scene terms. Use printer lights for print-stage colour.
- Grain and halation are computed per frame and are tile-safe; renders match playback.
- Each profile variant in use (stock × grey balance × Lab scan) keeps four 65³ LUTs on the GPU, about 13 MB, plus working buffers for halation and bloom at the frame size. If the GPU runs out of memory the node reports it in Resolve's message area.
- Profiles are approximate digitizations of published data; they are not endorsed by Kodak, Fujifilm, HARMAN or anyone else.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Open FX Error … could not be run" | GPU processing mode is not CUDA. Set CUDA and restart Resolve. |
| Flat, dim or doubly contrasty picture | A second tone mapping is on (Output DRT / ACES ODT). Turn it off, see the node trees. |
| Clipped or odd colours | Input colour space of the node does not match the incoming image. |
| Grain too coarse in a proxy/timeline at another resolution | Grain size follows frame height; check that the clip is not scaled after the plugin. |
| "not enough GPU memory for this frame" | Halation and bloom need working buffers of the frame size (about 530 MB at 4K on Full, half that on Fast). Set Glow quality to Fast, lower the timeline resolution, or close other GPU programs. |
| Playback stutters with halation/bloom | Set Glow quality to Fast while grading, back to Full before rendering. |
