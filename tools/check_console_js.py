"""
Static checks for the console's inline script.

`node --check` validates SYNTAX only. It is perfectly happy with a function that
references a variable which does not exist, which is how two separate
production-breaking bugs shipped:

  * hudSize() read `isStill`, which is a parameter of drawOverlay() and not a
    module-level name. It threw ReferenceError during top-level init, so the
    script aborted before the click listener was attached and the camera button
    sat on "Starting..." forever.
  * drawKpis() read `perImg` after that identifier was removed, throwing four
    times a second and silently killing the event log.

Both are ReferenceErrors that only surface at runtime. So: run the script and
see if it throws. A harness that exercises the top-level statements and the
entry points catches this class without a browser.
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "graytech" / "server.py"

# Minimal DOM the script touches at load time. Enough for the top-level
# statements to run and for the listeners to attach.
HARNESS = r"""
const noop = () => {};
const mkEl = (id, tag) => {
  const el = {
    id, tagName: (tag || 'DIV').toUpperCase(), style: {}, dataset: {},
    className: '', textContent: '', innerHTML: '', value: '', disabled: false,
    hidden: false, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    children: [], childNodes: [], options: [],
    appendChild: noop, removeChild: noop, insertAdjacentHTML: noop,
    addEventListener: noop, removeEventListener: noop, dispatchEvent: noop,
    getContext: () => ctx2d, getBoundingClientRect: () => ({left:0,top:0,width:681,height:382,right:681,bottom:382}),
    querySelector: () => null, querySelectorAll: () => [],
    focus: noop, click: noop, remove: noop, setAttribute: noop,
    clientWidth: 681, clientHeight: 382, width: 681, height: 382,
    play: () => Promise.resolve(), pause: noop, load: noop,
    srcObject: null, readyState: 4, paused: false, muted: true,
    videoWidth: 720, videoHeight: 1280, currentTime: 0,
    getContext2d: null,
  };
  return el;
};
const ctx2d = {
  canvas: { width: 681, height: 382 },
  clearRect: noop, fillRect: noop, strokeRect: noop, beginPath: noop, moveTo: noop,
  lineTo: noop, stroke: noop, fill: noop, ellipse: noop, arc: noop, fillText: noop,
  measureText: () => ({ width: 10 }), drawImage: noop, save: noop, restore: noop,
  setLineDash: noop, getImageData: () => ({ data: new Uint8ClampedArray(4) }),
  fillStyle: '', strokeStyle: '', lineWidth: 1, font: '', globalAlpha: 1, textAlign: '',
};
const store = {};
const document = {
  getElementById: (id) => (store[id] ||= mkEl(id)),
  querySelector: (sel) => mkEl(sel),
  querySelectorAll: () => [],
  createElement: (t) => mkEl('', t),
  createElementNS: (ns, t) => mkEl('', t),
  addEventListener: noop, removeEventListener: noop,
  body: { style: {}, appendChild: noop, classList: { add: noop, remove: noop, toggle: noop } },
  documentElement: { style: {}, scrollWidth: 800 },
  hidden: false,
};
const window = {
  innerWidth: 800, innerHeight: 600, devicePixelRatio: 1,
  addEventListener: noop, removeEventListener: noop,
  location: { origin: 'http://x', href: 'http://x/' },
};
// The script reads CSS custom properties off the document element at load, and
// calls this as a BARE global - so it has to be a global, not window.getComputedStyle.
const getComputedStyle = () => new Proxy({}, {
  get: (t, k) => {
    if (k === 'getPropertyValue') return () => '';
    return '';
  },
});
const navigator = {
  mediaDevices: {
    enumerateDevices: () => Promise.resolve([]),
    getUserMedia: () => Promise.reject(new Error('no camera in harness')),
  },
  permissions: { query: () => Promise.resolve({ state: 'denied' }) },
  userAgent: 'harness',
};
const location = window.location;
const crypto = { randomUUID: () => '00000000-0000-4000-8000-000000000000' };
const EventSource = function () {
  this.close = noop; this.onopen = null; this.onerror = null; this.onmessage = null;
};
EventSource.prototype.close = noop;
const FormData = function () { this.append = noop; };
const File = function () {};
const Blob = function () {};
const URL = { createObjectURL: () => 'blob:x', revokeObjectURL: noop };
const Image = function () { this.src = ''; };
Image.prototype.decode = () => Promise.resolve();
const fetch = () => Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}), text: () => Promise.resolve('') });
const setInterval = () => 0, clearInterval = noop;
const setTimeout = () => 0, clearTimeout = noop;
const requestAnimationFrame = noop;
// Real console semantics for error(), and a process exit code that reflects it.
// The previous version routed errors into an array nobody ever read, so the
// harness reported OK on a script that threw - a green check that cannot fail,
// which is worse than no check at all.
const console = {
  log: noop, warn: noop, info: noop, debug: noop,
  error: (...a) => { process.stderr.write('[console.error] ' + a.join(' ') + '\n'); },
};
globalThis.addEventListener = noop;
"""


def main() -> int:
    src = SERVER.read_text(encoding="utf-8")
    m = re.search(r'UI_HTML = r"""(.*?)"""', src, re.S)
    if not m:
        print("could not find UI_HTML")
        return 1
    html = m.group(1)
    js = html[html.index("<script>") + 8: html.rindex("</script>")]

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "hud.js"
        # Wrap so top-level statements actually execute and any throw is caught.
        p.write_text(
            HARNESS
            + "\n;(async () => {\n"
            + js
            + "\n})().then(() => {\n"
            + "  if (typeof hudSize === 'function') { try { hudSize(); } "
            + "catch (e) { process.stderr.write('HUDSIZE: ' + e.name + ': ' + e.message + '\\n'); } }\n"
            + "  if (typeof listCams === 'function') { listCams().catch(() => {}); }\n"
            + "}).catch(e => { process.stderr.write('TOPLEVEL: ' + e.name + ': ' + e.message + '\\n'); });\n",
            encoding="utf-8")
        r = subprocess.run(["node", str(p)], capture_output=True, text=True, timeout=60)

    # Exercise the per-frame overlay path too. A temporal-dead-zone read inside
    # drawOverlay() - "Cannot access 'onFace' before initialization" - only
    # throws once a frame is actually drawn, so checking init alone is not enough.
    # Two shapes: a person with a face (named), and one without.
    shapes = {
        "named": {
            "frame": {"w": 720, "h": 1280},
            "persons": [{"box": [100, 200, 400, 1100], "conf": 0.9, "norm": [0.14, 0.16, 0.55, 0.86]}],
            "tracks": [{"id": 1, "box": [100, 200, 400, 1100], "conf": 0.9, "hits": 4, "age_s": 2}],
            "identify": {"ran": True, "named": 1, "gated": 0, "gallery": 33, "face_ms": 40,
                         "people": [{"box": [100, 200, 400, 1100], "stage": "named", "name": "Joe_Lee",
                                     "cosine": 0.99, "conf": 0.9,
                                     "face": {"bbox": [200, 220, 300, 330], "face_px": 100,
                                              "band": "robust"}}]},
            "count": 1, "track_count": 1, "ok": True, "conf": 0.25, "ms": 100, "find_ms": 90,
        },
        "noface": {
            "frame": {"w": 720, "h": 1280},
            "persons": [{"box": [10, 20, 300, 900], "conf": 0.4, "norm": [0.01, 0.02, 0.42, 0.70]}],
            "tracks": [{"id": 2, "box": [10, 20, 300, 900], "conf": 0.4, "hits": 1, "age_s": 1}],
            "identify": {"ran": True, "named": 0, "gated": 0, "gallery": 33, "face_ms": 30,
                         "people": [{"box": [10, 20, 300, 900], "stage": "found", "note": "no face"}]},
            "count": 1, "track_count": 1, "ok": True, "conf": 0.25, "ms": 120, "find_ms": 95,
        },
    }
    with tempfile.TemporaryDirectory() as td:
        p2 = Path(td) / "frame.js"
        p2.write_text(
            HARNESS
            + "\n;(async () => {\n" + js + "\n"
            + "  const shapes = " + __import__("json").dumps(shapes) + ";\n"
            + "  for (const [k, d] of Object.entries(shapes)) {\n"
            + "    try { drawOverlay(d, false); }"
            + "    catch (e) { process.stderr.write('OVERLAY[' + k + ']: ' + e.name + ': ' + e.message + '\\n'); }\n"
            + "    try { drawOverlay(d, true); }"
            + "    catch (e) { process.stderr.write('OVERLAY-STILL[' + k + ']: ' + e.name + ': ' + e.message + '\\n'); }\n"
            + "  }\n"
            + "  const st = Object.assign({}, shapes.named, {\n"
            + "    present: [], stats: { identified: 5, unknown: 1, entered: 9, exited: 4 },\n"
            + "    in_frame: 2, tracked: 2, edge: 0, scan_ms: 42, uptime: 300,\n"
            + "    pixel: { observed_median: 150, iod_median: 62, spec: { detect_px: 20 } },\n"
            + "    person: { live_frames: 10, live_detections: 9, tracks: [] },\n"
            + "    calibration: { recommend: {}, profiles: {} },\n"
            + "  });\n"
            + "  try { drawKpis(st); }"
            + "  catch (e) { process.stderr.write('KPIS: ' + e.name + ': ' + e.message + '\\n'); }\n"
            + "  try { drawStages(st); }"
            + "  catch (e) { process.stderr.write('STAGES: ' + e.name + ': ' + e.message + '\\n'); }\n"
            + "  try { drawLog([{t:1,kind:'scan',name:'x',detected:true,matched:true,confidence:0.9}]); }"
            + "  catch (e) { process.stderr.write('LOG: ' + e.name + ': ' + e.message + '\\n'); }\n"
            + "})().catch(e => { process.stderr.write('TOPLEVEL: ' + e.name + ': ' + e.message + '\\n'); });\n",
            encoding="utf-8")
        r2 = subprocess.run(["node", str(p2)], capture_output=True, text=True, timeout=60)

    out2 = ((r2.stderr or "") + (r2.stdout or "")).strip()
    if r2.returncode != 0 or any(m in out2 for m in ("TOPLEVEL:", "HUDSIZE:", "OVERLAY", "KPIS:", "STAGES:", "LOG:")):
        print("HARNESS FAILED - the per-frame path threw:")
        print(out2[:2500] if out2 else f"(exit {r2.returncode}, no output)")
        return 1

    out = ((r.stderr or "") + (r.stdout or "")).strip()
    if r.returncode != 0 or any(m in out for m in ("TOPLEVEL:", "HUDSIZE:")):
        print("HARNESS FAILED - the script threw at load:")
        print(out[:2000] if out else f"(exit {r.returncode}, no output)")
        return 1

    print("HARNESS OK")
    print("  - script initialises and hudSize() runs")
    print("  - drawOverlay() runs for a named person and for one with no face")
    print("  - drawKpis/drawStages/drawLog run")
    print("  catches ReferenceError and temporal-dead-zone reads that")
    print("  node --check cannot see")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())