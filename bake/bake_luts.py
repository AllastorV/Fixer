import json
import pathlib
import sys
import warnings

warnings.simplefilter("ignore")

import numpy as np

from filmemu import film, lut, spaces

SCHEMA_VERSION = 1
SLOTS = ("under", "normal", "over")
SLOT_CONTRAST = {"under": 0.85, "normal": 1.0, "over": 1.15}
CHUNK = 16384
PRINT_HALF_RANGE = 3.0
NEGATIVES = (("kodak-vision3-50d", "Kodak Vision3 50D"), ("kodak-vision3-200t", "Kodak Vision3 200T"),
             ("kodak-vision3-250d", "Kodak Vision3 250D"), ("kodak-vision3-500t", "Kodak Vision3 500T"),
             ("fuji-eterna-vivid500", "Fuji Eterna Vivid 500"))
STILL_NEGATIVES = (("kodak-gold200", "Kodak Gold 200"), ("kodak-ultramax400", "Kodak UltraMax 400"),
                   ("fujifilm-400", "Fujifilm 400"), ("fuji-superia-xtra400", "Fuji Superia X-TRA 400"),
                   ("fuji-c200", "Fuji C200"), ("kodak-ektar100", "Kodak Ektar 100"),
                   ("fuji-fujicolor100", "Fujicolor 100"), ("fuji-superia200", "Fuji Superia 200"),
                   ("fuji-superia1600", "Fuji Superia 1600"), ("fuji-pro400h", "Fuji Pro 400H"),
                   ("kodak-portra160", "Kodak Portra 160"), ("kodak-portra400", "Kodak Portra 400"), ("kodak-portra800", "Kodak Portra 800"),
                   ("kodak-portra160nc", "Kodak Portra 160NC"), ("kodak-portra160vc", "Kodak Portra 160VC"),
                   ("kodak-proimage100", "Kodak Pro Image 100"), ("kodak-supra100", "Kodak Supra 100"),
                   ("kodak-aerocolor4-125", "Kodak Aerocolor IV 125"))
STILL_PAPER = "kodak-endura"
BW_NEGATIVES = (("ilford-fp4plus", "Ilford FP4 Plus"), ("ilford-hp5plus", "Ilford HP5 Plus"),
                ("ilford-delta400", "Ilford Delta 400"), ("fuji-acros2-100", "Fuji Acros II"),
                ("ilford-xp2super", "Ilford XP2 Super"), ("kodak-doublex", "Kodak Double-X 5222"), ("kodak-plusx", "Kodak Plus-X"),
                ("kodak-trix400", "Kodak Tri-X 400"), ("kodak-tmax100", "Kodak T-Max 100"), ("kodak-tmax400", "Kodak T-Max 400"),
                ("ilford-delta3200", "Ilford Delta 3200"), ("ilford-delta100", "Ilford Delta 100"), ("ilford-panfplus", "Ilford Pan F Plus"))
BW_PAPER = "ilford-multigrade-iv-rc"
REVERSALS = (("fuji-provia100f", "Fuji Provia 100F"), ("fuji-velvia50", "Fuji Velvia 50"), ("fuji-velvia100", "Fuji Velvia 100"),
             ("kodak-kodachrome64", "Kodak Kodachrome 64"), ("kodak-ektachrome-e100", "Kodak Ektachrome E100"))
BAKE_ALT_LIGHT = False


def identity(x):
    return x


def single_stage_profiles():
    slots = film.synthetic_slots()
    yield "identity", "Identity (test)", {s: identity for s in SLOTS}
    yield "synthetic-test", "Synthetic test", {s: (lambda x, p=p: film.render_film(x, p)) for s, p in slots.items()}


def _grid(n):
    axis = np.linspace(0.0, 1.0, n)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(-1, 3)


def _chunked(fn, pts):
    return np.concatenate([fn(pts[i:i + CHUNK]) for i in range(0, len(pts), CHUNK)])


def bake_stage_a(system_for_slot, lo, hi, scene_light, design_light, n=lut.LUT_SIZE):
    from filmemu import spectral, stocks

    lin = spaces.di_decode(_grid(n))
    coef = spectral.node_table(n)
    out = {}
    for slot in SLOTS:
        c = stocks.make_chain(system_for_slot(slot), scene_light, design_light)
        vals = np.concatenate([c.print_log_exposure(lin[i:i + CHUNK], coef[i:i + CHUNK]) for i in range(0, len(lin), CHUNK)])
        out[slot] = np.clip(vals, lo, hi).reshape(n, n, n, 3)
    return out


def bake_two_stage(system_for_slot, n=lut.LUT_SIZE, half_range=PRINT_HALF_RANGE, grey_balance=False):
    from filmemu import spectral, stocks

    coords = _grid(n)
    lin = spaces.di_decode(coords)
    coef = spectral.node_table(n)
    chain = stocks.make_chain(system_for_slot("normal"))
    x0 = float(np.mean([c.x0 for c in chain.s.print_.curves]))
    lo, hi = x0 - half_range, x0 + half_range
    stage_a = {}
    for slot in SLOTS:
        c = stocks.make_chain(system_for_slot(slot))
        vals = np.concatenate([c.print_log_exposure(lin[i:i + CHUNK], coef[i:i + CHUNK]) for i in range(0, len(lin), CHUNK)])
        stage_a[slot] = np.clip(vals, lo, hi).reshape(n, n, n, 3)
    return stage_a, bake_stage_b(chain, lo, hi, grey_balance, n), lo, hi


def bake_stage_b(chain, lo, hi, grey_balance=False, n=lut.LUT_SIZE):
    from filmemu import stocks

    log_e = lo + _grid(n) * (hi - lo)
    display = chain.display_linear
    if grey_balance:
        balance = grey_balance if callable(grey_balance) else stocks.grey_balance(chain)
        display = lambda x: balance(chain.display_linear(x))
    return spaces.di_encode(_chunked(display, log_e)).reshape(n, n, n, 3)


def two_stage_profiles():
    from filmemu import spectral

    from filmemu import stocks

    yield ("synthetic-spectral", "Synthetic spectral (test)",
           (lambda slot, balanced=False: spectral.synthetic_system(SLOT_CONTRAST[slot])),
           "synthetic spectral curves; not a film stock", None)
    for group, paper in ((NEGATIVES, "kodak-2383"), (STILL_NEGATIVES, STILL_PAPER), (BW_NEGATIVES, BW_PAPER)):
        for neg, label in group:
            if (stocks.STOCKS / f"{neg}.json").exists() and (stocks.STOCKS / f"{paper}.json").exists():
                yield (f"{neg}-{paper.removeprefix('kodak-')}", label,
                       (lambda slot, balanced=False, neg=neg, paper=paper: stocks.system(neg, paper, SLOT_CONTRAST[slot], balanced)),
                       f"approximate digitization of manufacturer technical data ({stocks.load(neg)['source']}) and "
                       f"{stocks.load(paper)['source']}; not endorsed by the manufacturers", neg)
    for neg, label in REVERSALS:
        if (stocks.STOCKS / f"{neg}.json").exists():
            yield (neg, label, (lambda slot, balanced=False, neg=neg: stocks.system(neg, None, SLOT_CONTRAST[slot], balanced)),
                   f"approximate digitization of manufacturer technical data ({stocks.load(neg)['source']}); "
                   f"not endorsed by the manufacturer", neg)


def _check(name, data):
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name}: non-finite LUT values")
    return np.ascontiguousarray(data, np.float32)


def write_single(root, profile_id, label, fns):
    out = root / profile_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": SCHEMA_VERSION, "kind": "single", "profile_id": profile_id, "label": label,
            "lut_size": lut.LUT_SIZE, "lut_encoding": "dwg_di", "axis_order": "r_fastest", "slots": {}, "test": "true"}
    for slot, fn in fns.items():
        (out / f"{slot}.bin").write_bytes(_check(f"{profile_id}/{slot}", lut.bake(fn)).tobytes())
        meta["slots"][slot] = f"{slot}.bin"
    (out / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_two_stage(root, profile_id, label, system_for_slot, data_source, negative_id=None):
    out = root / profile_id
    out.mkdir(parents=True, exist_ok=True)
    from filmemu import stocks

    is_paper = system_for_slot("normal").print_.label == stocks.load(STILL_PAPER)["label"]
    reversal = bool(negative_id) and stocks.is_reversal(negative_id)
    colour = bool(negative_id) and not stocks.is_bw(negative_id) and not reversal
    family = stocks.family_balance(STILL_PAPER if is_paper else "kodak-2383") if colour else False
    stage_a, stage_b, lo, hi = bake_two_stage(system_for_slot, grey_balance=family)
    meta = {"schema_version": SCHEMA_VERSION, "kind": "two_stage", "profile_id": profile_id, "label": label,
            "lut_size": lut.LUT_SIZE, "lut_encoding": "print_log_exposure", "axis_order": "r_fastest",
            "print_encoding": "dwg_di", "print_lo": lo, "print_hi": hi, "slots": {}, "print": "print.bin",
            "data_source": data_source}
    if not negative_id:
        meta["test"] = "true"
    for slot in SLOTS:
        (out / f"{slot}.bin").write_bytes(_check(f"{profile_id}/{slot}", stage_a[slot]).tobytes())
        meta["slots"][slot] = f"{slot}.bin"
    (out / "print.bin").write_bytes(_check(f"{profile_id}/print", stage_b).tobytes())
    if negative_id and stocks.is_bw(negative_id):
        meta["grain_mono"] = "true"
        (out / "grain.bin").write_bytes(_check(f"{profile_id}/grain", stocks.grain_table(system_for_slot("normal"), negative_id)).tobytes())
        meta["grain"] = "grain.bin"
    elif reversal:
        chain = stocks.make_chain(system_for_slot("normal"))
        print_bal = bake_stage_b(chain, lo, hi, grey_balance=True)
        (out / "print_bal.bin").write_bytes(_check(f"{profile_id}/print_bal", print_bal).tobytes())
        meta["balanced_print"] = "print_bal.bin"
        meta["grain_mono"] = "true"
        (out / "grain.bin").write_bytes(_check(f"{profile_id}/grain", stocks.grain_table(system_for_slot("normal"), negative_id)).tobytes())
        meta["grain"] = "grain.bin"
    elif negative_id:
        from filmemu import spectral

        balanced = lambda slot: system_for_slot(slot, True)
        stage_bal = bake_stage_a(balanced, lo, hi, None, None)
        meta["balanced_slots"] = {}
        for slot in SLOTS:
            (out / f"{slot}_bal.bin").write_bytes(_check(f"{profile_id}/{slot}_bal", stage_bal[slot]).tobytes())
            meta["balanced_slots"][slot] = f"{slot}_bal.bin"
        if is_paper:
            bal_chain = spectral.Chain(balanced("normal"))
            print_bal = bake_stage_b(bal_chain, lo, hi, grey_balance=True)
            (out / "print_bal.bin").write_bytes(_check(f"{profile_id}/print_bal", print_bal).tobytes())
            meta["balanced_print"] = "print_bal.bin"
        bal = stocks.balance(negative_id)
        if bal and BAKE_ALT_LIGHT:
            alt = stocks.opposite(bal)
            stage_alt = bake_stage_a(system_for_slot, lo, hi, stocks.light(alt), stocks.light(bal))
            meta["balance"], meta["alt_light"], meta["alt_slots"] = bal, alt, {}
            for slot in SLOTS:
                (out / f"{slot}_alt.bin").write_bytes(_check(f"{profile_id}/{slot}_alt", stage_alt[slot]).tobytes())
                meta["alt_slots"][slot] = f"{slot}_alt.bin"
        meta["grain_mono"] = "true"
        table = stocks.grain_table(system_for_slot("normal"), negative_id)
        if table is not None:
            (out / "grain.bin").write_bytes(_check(f"{profile_id}/grain", table).tobytes())
            meta["grain"] = "grain.bin"
    scan = stocks.scan_look(negative_id) if negative_id else None
    if scan:
        from filmemu import spectral

        chain = stocks.make_chain(system_for_slot("normal"))
        if colour and is_paper:
            bal_chain = spectral.Chain(system_for_slot("normal", True))
            grey = stocks.grey_balance(bal_chain)
            pairs = ((chain, lambda x: scan(family(x))), (bal_chain, lambda x: scan(grey(x))))
        elif colour:
            pairs = ((chain, lambda x: scan(family(x))),) * 2
        elif reversal:
            grey = stocks.grey_balance(chain)
            pairs = ((chain, scan), (chain, lambda x: scan(grey(x))))
        else:
            pairs = ((chain, scan),) * 2
        for name, (c, fn) in zip(("print_scan.bin", "print_bal_scan.bin"), pairs):
            (out / name).write_bytes(_check(f"{profile_id}/{name}", bake_stage_b(c, lo, hi, fn)).tobytes())
        meta["scan_print"], meta["scan_balanced_print"] = "print_scan.bin", "print_bal_scan.bin"
    (out / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(out_dir):
    root = pathlib.Path(out_dir) / "profiles"
    for profile_id, label, fns in single_stage_profiles():
        write_single(root, profile_id, label, fns)
    for profile_id, label, system_for_slot, data_source, negative_id in two_stage_profiles():
        write_two_stage(root, profile_id, label, system_for_slot, data_source, negative_id)
    (pathlib.Path(out_dir) / "profiles.stamp").write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
