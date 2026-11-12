# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Colorists and independent filmmakers grading in DaVinci Resolve on Windows with NVIDIA GPUs. They know the film chain (negative, print, printer lights, density) and judge a film emulation by how faithfully it behaves, not by a preset's first impression.

## Product Purpose

FIXER is a free OFX plugin that renders a digital image as it would look shot on a real film stock and printed (or scanned), with the stock's own grain and halation. Success: colorists choose it over paid emulations because the result behaves like the film they know.

## Positioning

The most realistic film emulation: each stock is built from the manufacturer's published technical data (characteristic curves, spectral sensitivity, dye density, granularity) through a two-stage negative → print spectral model, and verified pixel-for-pixel in Resolve against a Python reference. A "Lab scan" look fitted to real community scans is on by default; turning it off leaves the datasheet result. Free, no subscription, no machine lock.

## Operating Context

Resolve Color page, last node of the clip; DaVinci YRGB, RCM or ACES projects with output tone mapping off. GPU processing mode CUDA. Grading happens before the node; printer lights, exposure and push/pull are adjusted on it.

## Capabilities and Constraints

- 36 stocks: Kodak Vision3 50D/200T/250D/500T with 2383; Portra 160/400/800, Ektar 100, Gold 200, UltraMax 400; Fuji C200, Superia 200/400/1600, Fujicolor 100, Fujifilm 400, Pro 400H, Eterna Vivid 500 (2383); colour negatives on Kodak Endura; Tri-X 400, T-Max 100/400, Double-X 5222, Plus-X, Ilford HP5/FP4/Delta 100/400/3200/Pan F/XP2, Acros II on Ilford Multigrade; slides Provia 100F, Velvia 50/100, Ektachrome E100, Kodachrome 64.
- Controls: stock, grey balance, Lab scan, exposure, push/pull, impact, grain (amount, size, rate, seed), halation, bloom, glow quality, printer lights, colour spaces.
- Windows + NVIDIA CUDA only for now; Mac/Metal later.
- Ilford stocks share a default grain level (Ilford publishes no RMS granularity).

## Brand Commitments

- Name: FIXER (the fixing bath: the last step of the darkroom, which makes the image permanent). Written in capitals. No tagline for now.
- Independent: no film-manufacturer names, marks or look-alike wording in the brand (no Kodak/Fuji, no "-chrome", "-color", "Cine-"). Stock names appear only as factual product data, with "not endorsed by the manufacturer".
- Free product.

## Evidence on Hand

- Pixel parity reports: Resolve renders vs Python reference, 71/71 variants (`build/w13`, `build/w14`).
- Three-mode acceptance report, 36/36 (`build/w17/mode_check.pdf`).
- Datasheet sources and digitization overlays (private `research/` folder, not shipped).
- No testimonials, users, press or benchmarks against competitors yet. Do not invent them.

## Product Principles

1. Data first: a look comes from the stock's published data or real scans, never from taste alone; say so when data is missing.
2. Behaves like film: exposure, push/pull and printer lights answer the way the real chain does.
3. Respect the colorist: one node, honest defaults, no hidden tone mapping.
4. Free without strings.
