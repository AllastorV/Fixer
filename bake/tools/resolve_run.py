import json
import os
import pathlib
import sys
import time

MODULES = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules"


def connect(lib):
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", lib)
    sys.path.append(MODULES)
    import DaVinciResolveScript as dvr
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        sys.exit("Resolve not reachable")
    return resolve


def render(proj, out_dir, name, frame):
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
    return proj.GetRenderJobStatus(job).get("JobStatus")


def main(expected_dir, out_dir, lib):
    resolve = connect(lib)
    proj = resolve.GetProjectManager().GetCurrentProject()
    tl = proj.GetCurrentTimeline()
    item = tl.GetItemListInTrack("video", 1)[0]
    fe = item.GetFusionCompByIndex(1).FindTool("FE")
    frame = item.GetStart()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = fe.profile.GetAttrs().get("INPIDT_ComboControl_ID") or fe.profile.GetAttrs()
    (out / "profile_choices.json").write_text(json.dumps(labels, indent=1, default=str), encoding="utf-8")
    fe.SetInput("grainEnable", 0)
    fe.SetInput("halationAmount", 0.0)
    fe.SetInput("bloomAmount", 0.0)
    fe.SetInput("inSpace", 0)
    fe.SetInput("outSpace", 3)
    rows = json.loads((pathlib.Path(expected_dir) / "variants.json").read_text(encoding="utf-8"))
    for r in rows:
        name = f"{r['profile_id']}__{r['variant']}"
        fe.SetInput("profileId", r["profile_id"])
        fe.SetInput("greyBalance", int(r["grey_balance"]))
        fe.SetInput("labScan", int(r["lab_scan"]))
        print(name, render(proj, out, name, frame), flush=True)
    fe.SetInput("profileId", "kodak-vision3-250d-2383")
    fe.SetInput("greyBalance", 0)
    fe.SetInput("labScan", 0)
    fe.SetInput("grainEnable", 1)
    print("grain_on", render(proj, out, "grain_on", frame), flush=True)
    fe.SetInput("grainEnable", 0)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: resolve_run.py EXPECTED_DIR OUT_DIR FUSIONSCRIPT_DLL")
    main(*sys.argv[1:])
