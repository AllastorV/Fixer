import argparse
import json
import pathlib
import secrets
import sys

import numpy as np
import pymupdf
import tifffile

SIDES = ("dehancer", "ours")

PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Kör A/B</title>
<style>
:root{--bg:#111;--fg:#ddd;--mut:#888;--acc:#e8b04a}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;flex-wrap:wrap}
#lab{font-size:28px;font-weight:700;color:var(--acc);min-width:2ch}
#img{display:block;max-width:100vw;max-height:calc(100vh - 120px);margin:0 auto}
button{background:#222;color:var(--fg);border:1px solid #444;padding:8px 14px;border-radius:6px;cursor:pointer;font:inherit}
button.on{border-color:var(--acc);color:var(--acc)}
input{background:#1b1b1b;color:var(--fg);border:1px solid #444;padding:7px;border-radius:6px;width:320px;font:inherit}
.mut{color:var(--mut)}
</style></head><body>
<header>
<span id="pos" class="mut"></span><span id="lab">A</span>
<span class="mut">Boşluk: A/B geçiş · 1: A · 2: B · 3: Eşit · ←/→: çift</span>
<button data-c="A">A daha film gibi</button><button data-c="B">B daha film gibi</button><button data-c="=">Eşit</button>
<input id="note" placeholder="not (isteğe bağlı)">
<button id="dl">Sonuçları indir</button><span id="done" class="mut"></span>
</header>
<img id="img" alt="">
<script>
const PAIRS = __PAIRS__;
const NAMES = __NAMES__;
const KEY = "ab-" + __SESSION__;
let st = {i: 0, side: "A", res: {}};
try { const s = localStorage.getItem(KEY); if (s) st = Object.assign(st, JSON.parse(s)); } catch (e) {}
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
function show() {
  const p = PAIRS[st.i];
  $("img").src = "img/" + p + "_" + st.side + ".png";
  $("lab").textContent = NAMES ? st.side + " · " + NAMES[st.side] : st.side;
  $("pos").textContent = (st.i + 1) + " / " + PAIRS.length;
  const r = st.res[p] || {};
  document.querySelectorAll("button[data-c]").forEach(b => b.classList.toggle("on", b.dataset.c === r.choice));
  $("note").value = r.note || "";
  $("done").textContent = Object.keys(st.res).length + " / " + PAIRS.length + " oylandı";
}
function choose(c) { const p = PAIRS[st.i]; st.res[p] = {choice: c, note: $("note").value}; save(); if (st.i < PAIRS.length - 1) { st.i++; st.side = "A"; } show(); }
document.querySelectorAll("button[data-c]").forEach(b => b.onclick = () => choose(b.dataset.c));
$("note").oninput = () => { const p = PAIRS[st.i]; if (st.res[p]) { st.res[p].note = $("note").value; save(); } };
addEventListener("keydown", e => {
  if (e.target === $("note")) return;
  if (e.code === "Space") { e.preventDefault(); st.side = st.side === "A" ? "B" : "A"; show(); }
  else if (e.key === "1") choose("A"); else if (e.key === "2") choose("B"); else if (e.key === "3") choose("=");
  else if (e.key === "ArrowRight" && st.i < PAIRS.length - 1) { st.i++; st.side = "A"; show(); }
  else if (e.key === "ArrowLeft" && st.i > 0) { st.i--; st.side = "A"; show(); }
  save();
});
$("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({session: __SESSION__, results: st.res}, null, 1)], {type: "application/json"}));
  a.download = "ab_results.json"; a.click();
};
PAIRS.forEach(p => ["A", "B"].forEach(s => { new Image().src = "img/" + p + "_" + s + ".png"; }));
show();
</script></body></html>
"""


def load8(path, crop, masks=()):
    a = tifffile.imread(path)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    a = a[..., :3].astype(np.float64) / (65535.0 if a.dtype == np.uint16 else 255.0 if a.dtype == np.uint8 else 1.0)
    for x0, y0, x1, y1 in masks:
        h, w = a.shape[:2]
        a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0.5
    if crop:
        h, w = a.shape[:2]
        x0, y0, x1, y1 = crop
        a = a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return np.ascontiguousarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def save_png(a, path):
    pymupdf.Pixmap(pymupdf.csRGB, a.shape[1], a.shape[0], a.tobytes(), False).save(str(path))


def find_pairs(src):
    pairs = {}
    for f in sorted(pathlib.Path(src).glob("*.tif*")):
        stem = f.stem.lower()
        for side in SIDES:
            if stem.endswith("_" + side):
                pairs.setdefault(f.stem[:-len(side) - 1], {})[side] = f
    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    missing = sorted(set(pairs) - set(complete))
    return complete, missing


def prepare(src, out, crop, masks=(), labelled=False):
    pairs, missing = find_pairs(src)
    if missing:
        print("eksik çift (atlandı):", ", ".join(missing))
    if not pairs:
        print("çift bulunamadı: <sahne>_dehancer.tif ve <sahne>_ours.tif bekleniyor")
        return 2
    out = pathlib.Path(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    keydir = out.parent / (out.name + "_KEY_ACMA")
    keydir.mkdir(exist_ok=True)
    rng = secrets.SystemRandom()
    scenes = sorted(pairs)
    if not labelled:
        rng.shuffle(scenes)
    session = secrets.token_hex(4)
    key, ids = {"session": session, "pairs": {}}, []
    for n, scene in enumerate(scenes, 1):
        pid = f"p{n:02d}"
        a_side = SIDES[0] if labelled else rng.choice(SIDES)
        b_side = SIDES[1] if a_side == SIDES[0] else SIDES[0]
        imgs = {s: load8(pairs[scene][s], crop, masks) for s in SIDES}
        if imgs[a_side].shape != imgs[b_side].shape:
            print(f"boyut farkı, atlandı: {scene} {imgs[a_side].shape} vs {imgs[b_side].shape}")
            continue
        save_png(imgs[a_side], out / "img" / f"{pid}_A.png")
        save_png(imgs[b_side], out / "img" / f"{pid}_B.png")
        key["pairs"][pid] = {"scene": scene, "A": a_side, "B": b_side}
        ids.append(pid)
    (keydir / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    names = json.dumps({"A": "Dehancer", "B": "bizimki"}) if labelled else "null"
    html = PAGE.replace("__PAIRS__", json.dumps(ids)).replace("__SESSION__", json.dumps(session)).replace("__NAMES__", names)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(ids)} çift hazır: {out / 'index.html'}  (anahtar: {keydir / 'key.json'} — oylama bitene kadar açma)")
    return 0


def reveal(results, key):
    res = json.loads(pathlib.Path(results).read_text(encoding="utf-8"))
    k = json.loads(pathlib.Path(key).read_text(encoding="utf-8"))
    if res.get("session") != k["session"]:
        print(f"UYARI: oturum kimliği eşleşmiyor (sonuç {res.get('session')}, anahtar {k['session']})")
    tally = {"dehancer": 0, "ours": 0, "eşit": 0}
    for pid, info in sorted(k["pairs"].items()):
        r = res["results"].get(pid)
        if not r:
            print(f"{pid} {info['scene']}: oylanmadı")
            continue
        win = "eşit" if r["choice"] == "=" else info[r["choice"]]
        tally[win] += 1
        print(f"{pid} {info['scene']}: {win}" + (f"  — {r['note']}" if r.get("note") else ""))
    voted = sum(tally.values())
    print(f"\nToplam {voted}: bizimki {tally['ours']}, Dehancer {tally['dehancer']}, eşit {tally['eşit']}")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="ab_blind.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="0..1 kesirleri; filigranı dışarıda bırak")
    p.add_argument("--mask", type=float, nargs=4, action="append", default=[], metavar=("X0", "Y0", "X1", "Y1"),
                   help="0..1 kesirleri; bu bölge iki görüntüde de griyle örtülür (tekrarlanabilir)")
    p.add_argument("--labelled", action="store_true", help="kör değil: A=Dehancer, B=bizimki, sahne sırasıyla")
    r = sub.add_parser("reveal")
    r.add_argument("results")
    r.add_argument("key")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        for rect in ([a.crop] if a.crop else []) + a.mask:
            if not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1):
                ap.error("dikdörtgen 0 <= X0 < X1 <= 1 ve 0 <= Y0 < Y1 <= 1 olmalı")
        return prepare(a.src, a.out, a.crop, a.mask, a.labelled)
    return reveal(a.results, a.key)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
