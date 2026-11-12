import json
import pathlib
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import tifffile

from filmemu.pipeline import Params, render
from resolve_check import PROFILES, compare, luts, to16

SPACES = ("dwg_di", "dwg_linear", "acescct", "rec709_g24")
DEFAULTS = {"grainEnable": 0, "grainAmount": 0.5, "grainSize": 1.0, "grainSeed": 0, "grainRate": 0,
            "halationAmount": 0.0, "halationRadius": 6.0, "bloomAmount": 0.0, "glowQuality": 0,
            "inSpace": 0, "outSpace": 3, "exposure": 0.0, "pushPull": 0.0, "impact": 1.0,
            "printExposure": 0.0, "printRed": 0.0, "printGreen": 0.0, "printBlue": 0.0,
            "greyBalance": 0, "labScan": 0, "shootingLight": 0}
V250, TRIX, PORTRA, VELVIA = ("kodak-vision3-250d-2383", "kodak-trix400-ilford-multigrade-iv-rc",
                              "kodak-portra400-endura", "fuji-velvia50")
GRAIN = {"grainEnable": 1, "grainAmount": 0.8, "grainSize": 1.5, "grainSeed": 7, "grainRate": 3}
CASES = [
    ("exp_p2", V250, {"exposure": 2.0}),
    ("exp_m3", V250, {"exposure": -3.0}),
    ("push_p1", V250, {"pushPull": 1.0}),
    ("push_m1_5", V250, {"pushPull": -1.5}),
    ("impact_half", V250, {"impact": 0.5}),
    ("impact_zero", V250, {"impact": 0.0}),
    ("print_lights", V250, {"printExposure": 3.0, "printRed": -4.0, "printBlue": 2.0}),
    ("halation", V250, {"halationAmount": 0.6, "halationRadius": 12.0}),
    ("bloom", V250, {"bloomAmount": 0.5}),
    ("glow_both", V250, {"halationAmount": 0.4, "bloomAmount": 0.3}),
    ("in_linear", V250, {"inSpace": 1}),
    ("in_acescct", V250, {"inSpace": 2}),
    ("in_rec709", V250, {"inSpace": 3}),
    ("out_di", V250, {"outSpace": 0}),
    ("out_linear", V250, {"outSpace": 1}),
    ("out_acescct", V250, {"outSpace": 2}),
    ("light_daylight", V250, {"shootingLight": 1}),
    ("light_tungsten", V250, {"shootingLight": 2}),
    ("grain", V250, GRAIN),
    ("trix_exp_push", TRIX, {"exposure": 1.0, "pushPull": 1.0}),
    ("trix_grain", TRIX, GRAIN),
    ("portra_scan_bal_print", PORTRA, {"labScan": 1, "greyBalance": 1, "printGreen": 3.0}),
    ("velvia_exp_halation", VELVIA, {"exposure": -1.0, "halationAmount": 0.5}),
    ("velvia_grain", VELVIA, GRAIN),
]
GRAIN_FRAMES = (0,)


def params_for(inputs):
    s = {**DEFAULTS, **inputs}
    kw = dict(in_space=SPACES[s["inSpace"]], out_space=SPACES[s["outSpace"]], exposure_ev=s["exposure"],
              push_pull=s["pushPull"], impact=s["impact"], halation_amount=s["halationAmount"],
              halation_radius_px=s["halationRadius"], bloom_amount=s["bloomAmount"],
              print_log_offset=tuple(0.025 * (s["printExposure"] + s[k]) for k in ("printRed", "printGreen", "printBlue")))
    if s["grainEnable"]:
        kw.update(grain_amount=s["grainAmount"], grain_sigma_px=s["grainSize"], grain_seed=s["grainSeed"])
    return s, kw


def profile_luts(pid, s):
    pdir = PROFILES / pid
    out = luts(pdir, bool(s["greyBalance"]), bool(s["labScan"]))
    meta = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
    if "grain" in meta:
        out["grain"] = np.fromfile(pdir / meta["grain"], np.float32).reshape(3, 64)
    out["grain_mono"] = meta.get("grain_mono") in (True, "true")
    return out


def make(out_dir, input_tif):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = tifffile.imread(input_tif)
    h, w = q.shape[:2]
    rows = []
    for name, pid, inputs in CASES:
        s, kw = params_for(inputs)
        lt = profile_luts(pid, s)
        refs = {}
        for frame in (GRAIN_FRAMES if s["grainEnable"] else (0,)):
            for flip in ((False, True) if s["grainEnable"] else (False,)):
                src = q[::-1] if flip else q
                rgba = np.concatenate([src.reshape(h, w, 3) / 65535.0, np.ones((h, w, 1))], -1).astype(np.float32)
                img = render(rgba, lt, Params(frame=frame, **kw))[..., :3]
                img = img[::-1] if flip else img
                tag = f"{name}" if not s["grainEnable"] else f"{name}__f{frame}{'_flip' if flip else ''}"
                tifffile.imwrite(out / f"{tag}.tif", to16(img), photometric="rgb")
                refs[tag] = f"{tag}.tif"
        rows.append({"name": name, "profile_id": pid, "inputs": s, "refs": refs})
        print(name, flush=True)
    (out / "cases.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def render_frame(proj, out_dir, name, frame):
    proj.DeleteAllRenderJobs()
    proj.SetCurrentRenderFormatAndCodec("tif", "RGB16")
    proj.SetRenderSettings({"SelectAllFrames": False, "MarkIn": frame, "MarkOut": frame, "TargetDir": str(out_dir),
                            "CustomName": name, "ExportVideo": True, "ExportAudio": False, "ExportAlpha": False,
                            "FormatWidth": 1920, "FormatHeight": 1080})
    job = proj.AddRenderJob()
    if not job or not proj.StartRendering(job):
        return "not started"
    while proj.IsRenderingInProgress():
        time.sleep(0.1)
    st = proj.GetRenderJobStatus(job)
    return st.get("JobStatus") + (" " + st["Error"] if st.get("Error") else "")


def run(cases_dir, export_dir, lib):
    from resolve_run import connect
    resolve = connect(lib)
    proj = resolve.GetProjectManager().GetCurrentProject()
    item = proj.GetCurrentTimeline().GetItemListInTrack("video", 1)[0]
    fe = item.GetFusionCompByIndex(1).FindTool("FE")
    out = pathlib.Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((pathlib.Path(cases_dir) / "cases.json").read_text(encoding="utf-8"))
    for r in rows:
        fe.SetInput("profileId", r["profile_id"])
        for k, v in r["inputs"].items():
            fe.SetInput(k, v)
        got = {k: fe.GetInput(k) for k in r["inputs"]}
        bad = {k: (v, got[k]) for k, v in r["inputs"].items() if got[k] is None or abs(float(got[k]) - float(v)) > 1e-9}
        print(r["name"], render_frame(proj, out, r["name"] + "__", item.GetStart()), "unset" if bad else "", bad or "", flush=True)
    fe.SetInput("profileId", V250)
    for k, v in DEFAULTS.items():
        fe.SetInput(k, v)


def compare_cases(cases_dir, export_dir):
    rows = json.loads((pathlib.Path(cases_dir) / "cases.json").read_text(encoding="utf-8"))
    fails = 0
    for r in rows:
        hits = sorted(pathlib.Path(export_dir).glob(r["name"] + "__[0-9]*.tif"))
        if not hits:
            print("MISSING", r["name"])
            fails += 1
            continue
        best = None
        for tag, f in r["refs"].items():
            a = tifffile.imread(hits[0]).astype(np.float64)[..., :3] / 65535.0
            b = tifffile.imread(pathlib.Path(cases_dir) / f).astype(np.float64) / 65535.0
            d = np.abs(a - b).max()
            if best is None or d < best[0]:
                best = (d, tag)
        print(r["name"], "ref", best[1], end=" ", flush=True)
        fails += compare(hits[0], pathlib.Path(cases_dir) / r["refs"][best[1]],
                         str(pathlib.Path(export_dir) / f"{r['name']}_diff.png")) != 0
    print(f"{len(rows) - fails}/{len(rows)} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "make":
        make(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare_cases(sys.argv[2], sys.argv[3]))
    else:
        sys.exit("usage: resolve_params.py make OUT INPUT_TIF | run CASES_DIR EXPORT_DIR FUSIONSCRIPT_DLL | compare CASES_DIR EXPORT_DIR")
