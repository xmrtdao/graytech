"""
GrayTech Security — live scene visualisation over the face-service pipeline.

WHAT THIS IS, precisely, because the distinction matters:

This is a SIMULATION driving the REAL recognition pipeline. There is no camera.
People "walk" across a synthetic scene because a scheduler moves their
photographs across a canvas; when a person reaches the scan zone the frame is
pushed through the same insightface detection + cosine matching that
face-service/app uses. The detection, the embeddings, the identity decisions
and the confidence scores are all real. The walking is not.

The source photographs are the Faces dataset, which is celebrity portraits. That
makes them harmless for a demo and useless as evidence about real-world
performance: studio lighting, cooperative subjects, one person centred in frame.
Anything this dashboard says about accuracy applies to that set and no wider.

Architecture:
  - ONE process. The recognition model is loaded once here rather than by
    calling the face-service HTTP API, because a frame crosses the scan line
    roughly every few seconds and a round trip plus a second model load per
    frame is wasteful. Same code path, same model, imported not reimplemented.
  - Simulation runs on a background thread; the browser gets events over SSE.
    SSE rather than WebSocket because this is one-directional (server pushes,
    browser never sends), and SSE reconnects on its own.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

# Reuse the service that already exists. Same model, same matcher, same
# threshold - importing rather than reimplementing is the point, so a change to
# app/ changes this dashboard too.
import app as faceapp
from app import IDENTITY_DIR, face as recogniser

DATASET_DEFAULT = r"C:\Users\PureTrek\Desktop\Faces\Faces"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCENE_W, SCENE_H = 960, 540
SCAN_X = SCENE_W * 0.5          # the scan line, people walk through here
HEAD_R = 26                      # head radius on the stick figure

# Face thumbnails. Cached because the same 31 portraits are re-used constantly
# and decoding a JPEG per visitor per frame at 20Hz would dominate the sim loop.
#
# CROP, NOT SQUASH: insightface returns a bbox in source-image coordinates, so we
# crop to it first. Resizing the whole portrait into a circle instead would show
# shoulders and background, not the face, and the face is the entire point.
_thumb_cache: dict[str, Optional[str]] = {}
_thumb_lock = threading.Lock()


def thumb_data_uri(path_str: str, max_side: int = 128) -> Optional[str]:
    """Crop the detected face out of a photo, return it as a data URI."""
    with _thumb_lock:
        if path_str in _thumb_cache:
            return _thumb_cache[path_str]
    uri = None
    try:
        import cv2
        img = cv2.imread(path_str)
        if img is None:
            raise ValueError("unreadable")
        faces, _ = recogniser.embed(Path(path_str).read_bytes())
        if faces:
            # square the bbox around its centre and pad a little, so the crop is
            # not a hard rectangle of hairline
            x1, y1, x2, y2 = faces[0]["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1) * 0.62
            h, w = img.shape[:2]
            xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
            xb, yb = int(min(w, cx + half)), int(min(h, cy + half))
            crop = img[ya:yb, xa:xb]
            if crop.size:
                # Circular mask so the portrait fades into the head outline
                # instead of reading as a hard-edged square pasted on top.
                #
                # This was originally cv2.bitwise_join(blurred, mask), which
                # does NOT exist in OpenCV 5 - no bitwise_join at all. It threw
                # AttributeError, and because the whole body sat under a bare
                # `except Exception: uri = None`, every thumbnail silently came
                # back None. The symptom was heads rendering as plain circles
                # with withFaceImage:0, which reads like a detection failure
                # rather than a missing function. Blurred crop is masked with
                # the circle itself, which is what bitwise_join was reaching for.
                ch, cw = crop.shape[:2]
                mask = np.zeros((ch, cw), np.uint8)
                cv2.circle(mask, (cw // 2, ch // 2), min(cw, ch) // 2, 255, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), 1.0)[..., None].astype(np.float32) / 255.0
                soft = cv2.GaussianBlur(crop, (0, 0), 1.2).astype(np.float32)
                crop = (soft * mask + crop.astype(np.float32) * (1 - mask))
                crop = np.clip(crop, 0, 255).astype(np.uint8)
                crop = cv2.resize(crop, (max_side, max_side), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception as exc:                                   # noqa: BLE001
        # Logged, not swallowed. A bare `except: uri = None` is what let a
        # missing OpenCV function present as "no faces detected".
        log.warning("thumbnail failed for %s: %s: %s", path_str, type(exc).__name__, exc)
        uri = None
    with _thumb_lock:
        _thumb_cache[path_str] = uri
    return uri

MATCH_THRESHOLD = faceapp.MATCH_THRESHOLD


@dataclass
class Visitor:
    """One person in the scene. Source is either a dataset photo or the webcam."""
    name: str
    file: str
    x: float
    y: float
    vx: float
    vy: float
    confidence: Optional[float] = None
    matched: bool = False
    scanned: bool = False
    entering: bool = True
    live: bool = False          # True when this visitor is a live webcam frame
    frame: Optional[bytes] = None
    born: float = field(default_factory=time.time)

    def tick(self, dt: float) -> bool:
        """Advance position. Returns False when they have left the scene."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        # gentle vertical drift so the walk is not a perfectly straight line
        self.y += np.sin((time.time() - self.born) * 1.4) * 0.06
        if self.x < -140 or self.x > SCENE_W + 140:
            return False
        return True


class Scene:
    """The simulation. Owns the visitor list and the event stream."""

    def __init__(self, dataset: Path, speed: float = 34.0, max_present: int = 7):
        self.dataset = dataset
        self.speed = speed
        self.max_present = max_present
        self.visitors: list[Visitor] = []
        self.events: deque[dict] = deque(maxlen=400)
        self.loop = asyncio.get_event_loop()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = {
            "entered": 0, "exited": 0, "identified": 0,
            "unknown": 0, "started_at": time.time(),
        }
        self.scan_ms = 0.0
        # Live camera mode. OFF by default so a demo never opens a stranger's
        # webcam by accident - it has to be switched on deliberately.
        self.live_camera = False
        self.cam_name = os.environ.get("GRAYTECH_CAMERA", "HP TrueVision HD Camera")
        self.ffmpeg = os.environ.get("GRAYTECH_FFMPEG", r"C:\tools\ffmpeg\ffmpeg.exe")
        self.cam_status = "off"
        self.cam_frame: Optional[bytes] = None
        self.cam_at = 0.0

    # -- helpers ----------------------------------------------------------
    def log(self, kind: str, **kw) -> None:
        ev = {"kind": kind, "t": time.time(), **kw}
        self.events.append(ev)

    def candidates(self) -> list[Path]:
        if not self.dataset.is_dir():
            return []
        return sorted(p for p in self.dataset.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    # -- spawning ---------------------------------------------------------
    def grab_webcam(self) -> Optional[bytes]:
        """
        One frame from the laptop webcam.

        Deliberately the SAME capture path as the relay's `vex-vision` handler:
        ffmpeg DirectShow against the named device. cv2.VideoCapture on Windows
        builds its own capture graph and ignores DirectShow device naming, and a
        second backend would be a second competing handle on one camera.
        """
        if not Path(self.ffmpeg).exists():
            self.cam_status = f"ffmpeg missing at {self.ffmpeg}"
            return None
        tmp = Path(tempfile.gettempdir()) / "graytech-cam.jpg"
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "dshow", "-i", f"video={self.cam_name}",
               "-frames:v", "1", "-q:v", "2", "-update", "1", str(tmp), "-y"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=12)
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or b"").decode(errors="replace").strip()
            self.cam_status = f"capture failed: {detail[:120] or 'camera busy or not found'}"
            return None
        except subprocess.TimeoutExpired:
            self.cam_status = "capture timed out - camera held by another app"
            return None
        try:
            data = tmp.read_bytes()
        except OSError as e:
            self.cam_status = f"read failed: {e}"
            return None
        self.cam_status = f"live - {self.cam_name}"
        self.cam_frame = data
        self.cam_at = time.time()
        return data

    def spawn(self) -> Optional[Visitor]:
        from_left = random.random() < 0.5
        y = random.uniform(SCENE_H * 0.22, SCENE_H * 0.78)

        # Live camera mode: whoever is standing in front of the laptop walks in.
        if self.live_camera:
            data = self.grab_webcam()
            if not data:
                return None
            v = Visitor(
                name="live camera", file="(webcam)",
                x=-120 if from_left else SCENE_W + 120, y=y,
                vx=self.speed * 0.9 * (1 if from_left else -1),
                vy=random.uniform(-4, 4),
                entering=from_left, live=True, frame=data,
            )
            self.visitors.append(v)
            self.stats["entered"] += 1
            self.log("entry", name="live camera", file="webcam", live=True)
            return v

        files = self.candidates()
        if not files:
            return None
        # avoid re-using the very same photo immediately; feels less mechanical
        recent = {v.file for v in self.visitors}
        pool = [f for f in files if str(f) not in recent] or files
        p = random.choice(pool)

        v = Visitor(
            name=faceapp_synth_name(p.stem),
            file=str(p),
            x=-120 if from_left else SCENE_W + 120,
            y=y,
            vx=self.speed * random.uniform(0.85, 1.15) * (1 if from_left else -1),
            vy=random.uniform(-6, 6),
            entering=from_left,
        )
        self.visitors.append(v)
        self.stats["entered"] += 1
        self.log("entry", name=v.name, file=p.name)
        return v

    # -- the recognition step --------------------------------------------
    def scan(self, v: Visitor) -> None:
        """
        Run a visitor's photo through the real pipeline.

        Blocking CPU work, so it runs on the simulation thread - which is
        already a background thread, separate from the event loop. That keeps
        the ~55ms inference off the asyncio loop and the UI stays responsive.
        """
        if v.live:
            data = v.frame
            if not data:
                self.log("error", name=v.name, detail="no webcam frame")
                return
        else:
            try:
                data = Path(v.file).read_bytes()
            except OSError as e:
                self.log("error", name=v.name, detail=str(e))
                return
        t0 = time.perf_counter()
        try:
            faces, matrix = recogniser.embed(data)
        except Exception as e:                                  # noqa: BLE001
            self.log("error", name=v.name, detail=str(e))
            return
        self.scan_ms = (time.perf_counter() - t0) * 1000

        if matrix.size == 0:
            v.scanned = True
            v.matched = False
            self.stats["unknown"] += 1
            self.log("scan", name=v.name, detected=False, matched=False)
            return

        m = recogniser.match(matrix, k=1)
        top = m[0][0] if (m and m[0]) else None
        v.scanned = True
        if top and top["matched"]:
            v.matched = True
            v.confidence = top["cosine"]
            v.name = top["name"]
            self.stats["identified"] += 1
        else:
            v.matched = False
            v.confidence = top["cosine"] if top else 0.0
            v.name = f"UNKNOWN ({v.name})"
            self.stats["unknown"] += 1
        self.log("scan", name=v.name, detected=True,
                 matched=v.matched, confidence=v.confidence,
                 ms=round(self.scan_ms, 1))

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        last = time.perf_counter()
        spawn_at = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            dt = min(now - last, 0.25)      # clamp: a long stall must not teleport anyone
            last = now

            # scan anyone who has just reached the line and not been scanned
            for v in self.visitors:
                if not v.scanned and ((v.x <= SCAN_X < v.x + v.vx * dt) if v.vx > 0
                                      else (v.x >= SCAN_X > v.x + v.vx * dt)):
                    self.scan(v)

            survivors = []
            for v in self.visitors:
                if v.tick(dt):
                    survivors.append(v)
                else:
                    self.stats["exited"] += 1
                    self.log("exit", name=v.name, matched=v.matched,
                             confidence=v.confidence)
            self.visitors = survivors

            # spawn cadence: never above max_present, and not faster than the UI can use
            if now >= spawn_at:
                if len(self.visitors) < self.max_present:
                    self.spawn()
                spawn_at = now + random.uniform(1.6, 4.2)

            time.sleep(0.05)                 # ~20Hz: smooth enough, cheap

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, daemon=True, name="graytech-sim")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- snapshot for the UI ---------------------------------------------
    def snapshot(self) -> dict:
        return {
            "present": [{
                "name": v.name,
                "x": round(v.x, 1),
                "y": round(v.y, 1),
                "scanned": v.scanned,
                "matched": v.matched,
                "confidence": (round(v.confidence, 3)
                               if v.confidence is not None else None),
                "entering": v.entering,
                "live": v.live,
                # Face portrait for the head of the stick figure. Sent only for
                # visitors who exist in a known file; live webcam frames carry
                # their own image instead.
                "face": None if v.live else thumb_data_uri(v.file),
                # A webcam visitor carries its own frame, shown inside the head so
                # the client can see it is live rather than a stored portrait.
                "liveFrame": ("data:image/jpeg;base64," +
                              base64.b64encode(v.frame).decode()) if (v.live and v.frame) else None,
            } for v in self.visitors],
            "stats": dict(self.stats),
            "scan_ms": round(self.scan_ms, 1),
            "scan_x": SCAN_X,
            "scene": {"w": SCENE_W, "h": SCENE_H},
            "uptime": round(time.time() - self.stats["started_at"], 1),
        }


def faceapp_synth_name(stem: str) -> str:
    """'Akshay Kumar_0' -> 'Akshay Kumar'. Matches the dataset labelling."""
    import re
    return re.sub(r"_\d+$", "", stem).strip()


scene = Scene(Path(DATASET_DEFAULT))


@asynccontextmanager
async def lifespan(_: FastAPI):
    recogniser.load()
    recogniser.reload_identities()
    scene.start()
    yield
    scene.stop()


application = FastAPI(title="GrayTech Security", version="0.1.0", lifespan=lifespan)


@application.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_pack": faceapp.MODEL_PACK,
        "identities_enrolled": len(recogniser.names),
        "present_now": len(scene.visitors),
        "threshold": MATCH_THRESHOLD,
        "dataset": str(scene.dataset),
        "dataset_exists": scene.dataset.is_dir(),
    }


@application.get("/api/state")
def state() -> dict:
    return scene.snapshot()


@application.get("/api/events")
async def events_recent(limit: int = Query(60, ge=1, le=400)) -> dict:
    return {"events": list(scene.events)[-limit:]}


@application.get("/api/stream")
async def stream():
    """Server-sent events. One-directional, so SSE beats WebSocket here."""
    async def gen():
        last_sent = 0.0
        while True:
            snap = scene.snapshot()
            recent = list(scene.events)[-40:]
            yield f"data: {json.dumps({'state': snap, 'events': recent})}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@application.post("/api/pause")
def pause() -> dict:
    if scene._stop.is_set():
        scene.start()
        return {"running": True}
    scene.stop()
    return {"running": False}


@application.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


UI_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GrayTech Security</title>
<style>
:root{--bg:#07090c;--panel:#0d1117;--line:#1e2530;--txt:#dfe6ee;--mut:#8b97a6;
      --grn:#3ddc84;--amb:#ffb020;--red:#ff5c5c;--cyn:#39d0ff;--pur:#a77bff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;padding:16px}
h1{font-size:19px;font-weight:700;letter-spacing:.02em;display:flex;align-items:center;gap:10px}
h1 small{font-size:12px;color:var(--mut);font-weight:400}
.badge{font-size:10px;padding:2px 7px;border-radius:999px;border:1px solid var(--line);color:var(--mut)}
.badge.live{border-color:var(--grn);color:var(--grn)}
.badge.sim{border-color:var(--amb);color:var(--amb)}
.row{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.grow{flex:1 1 620px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:14px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi .lab{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}
.kpi .val{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px}
canvas{width:100%;display:block;border-radius:8px;background:#05070a;border:1px solid var(--line)}
#log{max-height:330px;overflow-y:auto;font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
.ev{padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);display:flex;gap:8px}
.ev .t{color:var(--mut);flex-shrink:0}
.ev .k{width:52px;flex-shrink:0;font-weight:600}
.k.entry{color:var(--cyn)}.k.exit{color:var(--pur)}
.k.scan{color:var(--grn)}.k.error{color:var(--red)}
.ok{color:var(--grn)}.no{color:var(--red)}.mid{color:var(--amb)}
button{background:#151b24;color:var(--txt);border:1px solid var(--line);border-radius:6px;
       padding:6px 12px;cursor:pointer;font:inherit;font-size:12px}
button:hover{border-color:var(--cyn)}
.note{color:var(--mut);font-size:11.5px;margin-top:10px;line-height:1.5}
</style></head><body>

<h1>GrayTech Security
    <span class="badge live" id="conn">connecting</span></h1>
<small style="color:var(--mut);font-size:12px;display:block;margin-top:4px">
  Face detection &amp; recognition over a monitored scene.</small>

<div class="kpis" id="kpis"></div>

<div class="row">
  <div class="card grow">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <strong>Scene</strong>
      <button onclick="togglePause()">Pause / Resume</button>
    </div>
    <canvas id="scene" width="960" height="540"></canvas>
    <div class="note" id="scaninfo"></div>
  </div>
  <div class="card" style="flex:1 1 330px;min-width:300px">
    <strong style="display:block;margin-bottom:8px">Event log</strong>
    <div id="log"></div>
  </div>
</div>

<div class="card" style="margin-top:14px">
  <strong style="display:block;margin-bottom:6px">About this demo</strong>
  <p class="note" style="margin-top:0">
    <b>What is real:</b> face detection, the 512-d embeddings, the identity
    matches and every confidence score. These come from insightface
    (buffalo_s, detection + recognition heads only) via onnxruntime, run once
    per person as they cross the scan line. Identities are matched by cosine
    similarity against enrolled reference embeddings.
  </p>
  <p class="note">
    <b>What is simulated:</b> the movement. People walk across this scene
    because a scheduler moves their images along a path — there is no camera
    tracking real motion unless live camera mode is switched on. The frame that
    gets analysed is a still photograph, not live video.
  </p>
  <p class="note">
    <b>About the accuracy figure:</b> source images are the Faces dataset of
    studio portraits — controlled lighting, cooperative subjects, one face
    centred in frame. The percentage shown describes that set only. It is not
    evidence about performance on real people, in real conditions, off-angle or
    in poor light, where accuracy is typically materially lower.
  </p>
</div>

<script>
const cv = document.getElementById('scene'), cx = cv.getContext('2d');
let lastEvents = [], paused = false;

const fmtT = t => new Date(t*1000).toLocaleTimeString();

function draw(s){
  // decode each visitor's face ONCE, cache the Image on the state object.
  // Creating an Image per visitor per frame would decode the same 31 portraits
  // 20 times a second and stall the render.
  s.present.forEach(p=>{
    if(p.face && (!p._img || p._img.src!==p.face)){
      const im=new Image(); im.src=p.face; p._img=im;
    }
    if(p.liveFrame){
      if(!p._live || p._live.src!==p.liveFrame){
        const im=new Image(); im.src=p.liveFrame; p._live=im;
      }
    }
  });
  const W=s.scene.w, H=s.scene.h;
  cx.clearRect(0,0,W,H);
  // floor / zone
  cx.fillStyle='#080b10'; cx.fillRect(0,0,W,H);
  cx.strokeStyle='rgba(57,208,255,.13)';
  for(let gx=0; gx<W; gx+=60){ cx.beginPath(); cx.moveTo(gx,0); cx.lineTo(gx,H); cx.stroke(); }
  for(let gy=0; gy<H; gy+=60){ cx.beginPath(); cx.moveTo(0,gy); cx.lineTo(W,gy); cx.stroke(); }
  // scan line
  const sx=s.scan_x;
  const grd=cx.createLinearGradient(sx-26,0,sx+26,0);
  grd.addColorStop(0,'rgba(57,208,255,0)');
  grd.addColorStop(.5,'rgba(57,208,255,.20)');
  grd.addColorStop(1,'rgba(57,208,255,0)');
  cx.fillStyle=grd; cx.fillRect(sx-26,0,52,H);
  cx.strokeStyle='rgba(57,208,255,.75)'; cx.beginPath();
  cx.moveTo(sx+.5,0); cx.lineTo(sx+.5,H); cx.stroke();
  cx.fillStyle='rgba(57,208,255,.85)'; cx.font='11px ui-monospace,monospace';
  cx.fillText('SCAN ZONE', sx-38, 16);

  // people
  s.present.forEach(p=>{
    const x=p.x, y=p.y, R=26;
    const col = !p.scanned ? '#8b97a6' : (p.matched ? '#3ddc84' : '#ff5c5c');

    // body first, so the head sits on top of it
    cx.strokeStyle=col; cx.lineWidth=2;
    cx.beginPath(); cx.moveTo(x,y+R); cx.lineTo(x,y+64); cx.stroke();
    cx.beginPath(); cx.moveTo(x-20,y+42); cx.lineTo(x+20,y+42); cx.stroke();
    cx.beginPath(); cx.moveTo(x,y+64); cx.lineTo(x-16,y+96); cx.stroke();
    cx.beginPath(); cx.moveTo(x,y+64); cx.lineTo(x+16,y+96); cx.stroke();

    // head: the cropped face if we have one, otherwise the plain circle
    cx.save();
    cx.beginPath(); cx.arc(x,y,R,0,Math.PI*2); cx.clip();
    if(p.face){
      // cover-fit the square crop into the circle
      const img = p._img; const d=R*2;
      if(img.complete && img.naturalWidth){
        const sc = Math.max(d/img.naturalWidth, d/img.naturalHeight);
        cx.drawImage(img, x-d/2, y-d/2, img.naturalWidth*sc, img.naturalHeight*sc);
      } else { cx.fillStyle='#11161d'; cx.fillRect(x-R,y-R,d,d); }
    } else if(p.live && p._live){
      // webcam visitor: draw the live frame inside the head
      const img = p._live; const d=R*2;
      if(img.complete && img.naturalWidth){
        const sc = Math.max(d/img.naturalWidth, d/img.naturalHeight);
        cx.drawImage(img, x-d/2, y-d/2, img.naturalWidth*sc, img.naturalHeight*sc);
      } else { cx.fillStyle='#11161d'; cx.fillRect(x-R,y-R,d,d); }
    } else {
      cx.fillStyle = p.scanned ? 'rgba(20,32,26,.9)' : 'rgba(24,29,36,.9)';
      cx.fillRect(x-R,y-R,R*2,R*2);
    }
    cx.restore();

    // ring: colour carries the recognition state, and stays visible over a photo
    cx.strokeStyle=col; cx.lineWidth= p.scanned ? 3 : 1.5;
    cx.beginPath(); cx.arc(x,y,R,0,Math.PI*2); cx.stroke();

    if(p.scanned){
      cx.fillStyle=col; cx.font='bold 12px ui-sans-serif,system-ui';
      cx.textAlign='center'; cx.fillText(p.name, x, y-34);
      if(p.confidence!=null){
        cx.font='11px ui-monospace,monospace';
        cx.fillStyle='rgba(223,230,238,.85)';
        cx.fillText(p.confidence.toFixed(3), x, y-20);
      }
    } else {
      cx.fillStyle='rgba(139,151,166,.9)'; cx.font='11px ui-monospace,monospace';
      cx.textAlign='center'; cx.fillText('scanning...', x, y-34);
    }
    cx.textAlign='left';
  });
}

function drawKpis(s){
  const st=s.stats;
  const items=[
    ['Present now', s.present.length, ''],
    ['Entered', st.entered, ''],
    ['Exited', st.exited, ''],
    ['Identified', st.identified, 'var(--grn)'],
    ['Unknown', st.unknown, st.unknown?'var(--amb)':''],
    ['Scan time', (s.scan_ms||0).toFixed(0)+' ms', ''],
    ['Uptime', Math.floor(s.uptime/60)+'m '+Math.floor(s.uptime%60)+'s', ''],
  ];
  document.getElementById('kpis').innerHTML = items.map(([l,v,c])=>
    `<div class="kpi"><div class="lab">${l}</div><div class="val" style="color:${c||'var(--txt)'}">${v}</div></div>`
  ).join('');
  const ident = st.identified + st.unknown;
  document.getElementById('scaninfo').innerHTML =
    `Identified <b>${st.identified}</b> of <b>${ident}</b> scans `+
    `(${ident?((st.identified/ident)*100).toFixed(1):'0.0'}%) &middot; `+
    `model ${paused?'<span class="mid">paused</span>':'<span class="ok">running</span>'}`;
}

function drawLog(evs){
  const box=document.getElementById('log');
  const changed = evs.length!==lastEvents.length ||
                  (evs.length && evs[evs.length-1].t!==lastEvents[lastEvents.length-1]?.t);
  if(!changed) return;
  lastEvents=evs;
  box.innerHTML = evs.slice(-70).reverse().map(e=>{
    let detail='';
    if(e.kind==='scan'){
      detail = e.detected
        ? (e.matched? `matched ${e.name} ${e.confidence}` : `no match (best ${e.confidence})`)
        : 'no face detected';
    }
    return `<div class="ev"><span class="t">${fmtT(e.t)}</span>`+
           `<span class="k ${e.kind}">${e.kind.toUpperCase()}</span>`+
           `<span>${e.name||''} ${detail}</span></div>`;
  }).join('');
}

const es=new EventSource('/api/stream');
es.onopen =()=>{document.getElementById('conn').textContent='live';
                document.getElementById('conn').className='badge live';};
es.onerror=()=>{document.getElementById('conn').textContent='reconnecting';
                document.getElementById('conn').className='badge';};
es.onmessage=(m)=>{
  const d=JSON.parse(m.data);
  draw(d.state); drawKpis(d.state); drawLog(d.events);
};

async function togglePause(){
  const r=await fetch('/api/pause',{method:'POST'});
  const d=await r.json(); paused=!d.running;
}
</script></body></html>"""

app = application