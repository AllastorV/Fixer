# W01 — Numeric contract (v0.1, 2026-11-10)

This contract is shared by the Python reference (`bake/filmemu`) and the future CUDA/OFX code.
Every number below is our own design choice. Nothing here is taken from another product.

## 1. Pixel format and alpha

- Host image: float32 RGBA, straight (unpremultiplied) alpha.
- If the host reports premultiplied input: RGB is divided by alpha when alpha > 0; alpha = 0 leaves RGB as is.
- Alpha is never modified. Output is re-premultiplied only when the input was premultiplied.
- `impact = 0` short-circuits: the output buffer is a bitwise copy of the input (NaN/Inf included).

## 2. Spaces

| Id | Gamut | Transfer |
|---|---|---|
| `dwg_di` (default) | DaVinci Wide Gamut | DaVinci Intermediate |
| `dwg_linear` | DaVinci Wide Gamut | linear |
| `acescct` | ACES AP1 | ACEScct |
| `rec709_g24` | Rec.709 | BT.1886 pure 2.4 power |

Working space: **scene-linear DWG, D65**. Gamut changes are 3×3 matrices, derived once from primaries and stored as constants.

DaVinci Intermediate (Blackmagic definition):

```
A = 0.0075, B = 7.0, C = 0.07329248, M = 10.44426855
LIN_CUT = 0.00262409, LOG_CUT = 0.02740668
encode(x) = M·x                  if x <= LIN_CUT  else C·(log2(x + A) + B)
decode(y) = y / M                if y <= LOG_CUT  else 2^(y/C − B) − A
```

No clamp at decode or encode. Negative and superwhite values pass through the transfer formulas.
`rec709_g24` decode/encode uses `sign(x)·|x|^γ` so negative values round-trip.

## 3. Non-finite input

With `impact > 0`: any NaN channel becomes 0, +Inf becomes `F_MAX = 65504`, −Inf becomes `−F_MAX`, before decode. The same mapping is applied to the decoded values before the gamut matrix (a huge log input decodes to Inf, and Inf·0 inside the matrix would give NaN), and again to the working-linear values after the matrix. Output is always finite.

Printer points follow lab convention: more light in a channel → denser print in that channel → less of that colour on screen (+R points → cyan).

## 4. Exposure

`lin' = lin · 2^EV` in working linear. This is exact (power-of-two scale) and applies to negative values too.

## 5. Film log-exposure domain

- Reference: scene linear 0.18 maps to `logH = 0` (log10 relative exposure).
- `logH = log10(max(lin, LIN_FLOOR) / 0.18)`, `LIN_FLOOR = 2^-20`.
  Light energy is not negative: negative and near-zero linear values land on the floor, i.e. the film's base + fog.
  This is the only clamp in the colour chain and it exists only inside the film model.
- Printer lights: one printer point = 0.025 log10 exposure (industry convention).
  Print exposure per channel: `logH_print = −D_neg + P0 + 0.025·points_c`, where `P0` is chosen so the negative mid-grey prints to the print-curve mid-grey.

## 6. Characteristic curve (until stock data arrives, W06/W09)

Synthetic, C∞, monotonic: `D(x) = D_min + (D_max − D_min) · σ(k·(x − x0))`, `σ` the logistic function.
Profiles built with it are labelled `synthetic-test` and never carry a stock name.

## 7. 3D LUT

- Size 65³, float32 RGB, axis order: R fastest, then G, then B.
- LUT input coordinate = DI-encoded **working** RGB (DI is the shaper). Domain `[0, 1]` (≈ linear −0.0026 … 100).
- LUT values are DI-encoded working RGB (decoded after sampling).
- Exposure (§4) is applied before the lookup at render time; printer points and slot curves are baked into the LUT.
- Out-of-domain coordinates are clamped to the edge texel (`cudaAddressModeClamp`).
- Sampling convention matches CUDA normalized `tex3D` with linear filter: `u = (c·(N−1) + 0.5) / N`.
- CPU oracle: exact float trilinear. A second oracle quantizes the fractional weight to 1/256 to measure the hardware filter error separately.

## 8. Three exposure slots (under / normal / over)

Push/pull `p` in stops, clamped to `[−2, +2]`. Weights form a C1 partition of unity:

```
s = smoothstep(|p| / 2) = t²(3 − 2t)
p >= 0: w_over = s, w_normal = 1 − s, w_under = 0
p <  0: w_under = s, w_normal = 1 − s, w_over = 0
```

At `p = 0` the result is exactly the normal slot. The derivative is 0 at `p = 0` and continuous everywhere.
If any slot is missing, the render returns the input unchanged and reports the error (no partial dispatch).

## 9. Output

Working linear → output gamut matrix → output transfer. `impact` blends in **output-encoded** values:
`out = in_encoded_out + impact · (film − in_encoded_out)`.

## 10. Tolerances (first gates)

| Gate | Tolerance |
|---|---|
| Transfer round-trip (encode∘decode) on [−0.1, 100] | abs ≤ 1e-6 · max(1, |x|) |
| Our DI vs independent oracle (colour-science) | abs ≤ 1e-6 |
| Exposure 2^EV | exact (bitwise) for integer EV |
| Identity LUT, exact trilinear | abs ≤ 1e-6 inside domain |
| Slot weights | sum = 1 (abs ≤ 1e-7); p=0 → normal bitwise |
| impact = 0 | bitwise |
| CUDA vs exact FP64 oracle, output-encoded, relative to max(1, |x|) | ≤ 2.5e-4 (¼ of a 10-bit code value) |
| CUDA vs oracle with 1/256-rounded filter weights | ≤ 2e-4 |

W03 measurement (2026-11-10, RTX 3070 Ti, CUDA 13.5): max 1.4e-4 vs exact, 1.04e-4 vs rounded-weight oracle,
2.1e-4 vs truncated-weight oracle. Hardware 8-bit filter weights explain part of the gap, not all of it.
If A/B shows banding in smooth gradients, switch to manual FP32 trilinear on point-sampled texels (8 fetches).
