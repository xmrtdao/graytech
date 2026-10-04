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

        # RECORD every outcome for calibration. match_recorded takes the name we
        # believe is correct, which is what turns a raw score into a labelled
        # observation. Without it the panel has nothing to show and the
        # threshold can never be revised by evidence - which is exactly the
        # hardcoded-0.45 problem this is meant to fix.
        truth = safe_identity(v.name)
        m = recogniser.match_recorded(matrix, truth=truth, k=1)
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


def safe_identity(name: str) -> str:
    """
    'Akshay Kumar' -> 'Akshay_Kumar', matching how identities are stored on disk.

    The scene knows people by their readable dataset name, the gallery knows them
    by the filesystem-safe key. Scoring those against each other naively would
    mark every correct match as incorrect and poison the calibration data with
    false negatives that never happened.
    """
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


scene = Scene(Path(DATASET_DEFAULT))


@asynccontextmanager
async def lifespan(_: FastAPI):
    recogniser.load()
    recogniser.reload_identities()
    scene.start()
    yield
    scene.stop()


application = FastAPI(title="GrayTech Security", version="0.1.0", lifespan=lifespan)


@application.get("/calibration")
def calibration() -> dict:
    """
    Threshold calibration panel.

    The matching threshold was a hardcoded 0.45 - insightface's general default.
    This reports where the errors actually are in observed traffic, and what the
    threshold should be given them, so the number is measured rather than
    inherited.
    """
    from app.calibration import calibrator
    snap = calibrator.snapshot()
    snap["recommend"] = calibrator.recommend()
    snap["profiles"] = {k: v for k, v in snap.get("profiles", {}).items()
                        if v.get("corrections")}
    return snap


@application.post("/calibration/apply")
async def calibration_apply(req: dict) -> dict:
    from app.calibration import calibrator
    return calibrator.apply(req.get("threshold", 0.45))


@application.get("/api/profiles")
def profiles() -> dict:
    from app.calibration import calibrator
    return {"profiles": calibrator.profile_meta,
            "note": "enrolled identities that have absorbed corrections"}


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
    s = scene.snapshot()
    from app.calibration import calibrator
    s["calibration"] = calibrator.snapshot()
    s["calibration"]["recommend"] = calibrator.recommend()
    return s


@application.get("/api/events")
async def events_recent(limit: int = Query(60, ge=1, le=400)) -> dict:
    return {"events": list(scene.events)[-limit:]}


@application.get("/api/stream")
async def stream():
    """Server-sent events. One-directional, so SSE beats WebSocket here."""
    async def gen():
        while True:
            snap = scene.snapshot()
            # Calibration rides along on the stream, not just on /api/state.
            #
            # This called scene.snapshot() directly, so the SSE payload had no
            # 'calibration' key at all: /api/state returned 126 observations
            # while the live stream carried none, and the panel stayed empty
            # because drawCalibration(d.state.calibration) was being handed
            # undefined. Enriching here rather than duplicating the enrichment
            # means the stream and the REST endpoint can never disagree.
            try:
                from app.calibration import calibrator
                snap["calibration"] = calibrator.snapshot()
                snap["calibration"]["recommend"] = calibrator.recommend()
            except Exception as exc:                      # noqa: BLE001
                snap["calibration"] = {"error": str(exc)}
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
<meta name="description" content="Face detection and recognition over a monitored scene, with a self-calibrating match threshold.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' fill='%230D0F0C'/%3E%3Cpolygon points='32,7 57,32 32,57 7,32' fill='none' stroke='%23C9A227' stroke-width='5'/%3E%3Cpolygon points='32,20 44,32 32,44 20,32' fill='%23C9A227'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@400;500;600;700;800&family=Barlow:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
/* Palette and type scale lifted from graytechsolutions.dev so the operator
   console and the public site read as one product: olive-ink surfaces, 1px
   rules, square corners, gold accent, Saira Condensed / Barlow / Space Mono.

   The five state colours are NOT from that palette and are deliberately kept.
   Here colour carries meaning -- green matched, red unknown, amber calibration
   warning, cyan entry, violet exit -- so they are retuned warm enough to sit
   on olive without collapsing into the gold. Everything else is the site. */
:root{
  --ink:#0D0F0C; --ink-2:#111310; --ink-3:#171A15;
  --line:#2A2E25; --line-2:#3A3F31;
  --gold:#C9A227; --gold-bright:#E4BB3A;
  --paper:#ECEDE6; --mute:#9AA091; --mute-2:#6C7263;
  --grn:#8FC97A; --amb:#DC8A1E; --red:#E4695A; --cyn:#63C7D4; --pur:#A98AD6;
  --maxw:1180px;
  /* aliases -- the console script addresses these names directly */
  --bg:var(--ink); --panel:var(--ink-3); --txt:var(--paper); --mut:var(--mute);
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ink);color:var(--paper);max-width:var(--maxw);margin:0 auto;
  padding:0 28px 72px;font-family:'Barlow',system-ui,sans-serif;font-weight:400;
  line-height:1.6;-webkit-font-smoothing:antialiased}
::selection{background:var(--gold);color:var(--ink)}
.mono{font-family:'Space Mono',monospace}
.eyebrow{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.26em;
  text-transform:uppercase;color:var(--gold);display:block}

/* ---------- masthead ---------- */
header.top{border-bottom:1px solid var(--line);margin:0 -28px 24px;padding:30px 28px 24px;
  background:radial-gradient(120% 90% at 78% 0%, rgba(201,162,39,.10), transparent 55%),var(--ink)}
h1{font-family:'Saira Condensed',sans-serif;font-weight:700;font-size:clamp(30px,4.2vw,46px);
  line-height:1;text-transform:uppercase;letter-spacing:.01em;margin-top:12px;
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1 .g{color:var(--gold)}
h1 .mark{width:26px;height:26px;flex:none}
.tagline{color:var(--mute);font-weight:300;font-size:15.5px;margin-top:10px;max-width:640px}
.badge{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.16em;
  text-transform:uppercase;padding:4px 10px;border:1px solid var(--line-2);color:var(--mute);
  align-self:center}
.badge.live{border-color:var(--grn);color:var(--grn)}
.badge.sim{border-color:var(--amb);color:var(--amb)}

/* ---------- console surfaces ---------- */
.row{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px}
.card{background:var(--ink-3);border:1px solid var(--line);padding:20px 22px}
.grow{flex:1 1 620px;min-width:0}
.side{flex:1 1 340px;min-width:300px}
.cardhead{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;
  flex-wrap:wrap;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--line)}
.cardhead h2{font-family:'Saira Condensed',sans-serif;font-weight:600;
  font-size:clamp(20px,2.4vw,26px);text-transform:uppercase;letter-spacing:.01em;
  line-height:1.05;margin-top:6px}

.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:14px}
.kpi{background:var(--ink-3);border:1px solid var(--line);padding:14px 16px}
.kpi .lab{font-family:'Space Mono',monospace;font-size:10.5px;letter-spacing:.2em;
  color:var(--gold);text-transform:uppercase}
.kpi .val{font-family:'Saira Condensed',sans-serif;font-weight:700;font-size:30px;line-height:1;
  margin-top:8px;font-variant-numeric:tabular-nums;color:var(--paper)}

canvas{width:100%;display:block;background:var(--ink);border:1px solid var(--line-2)}
#log{max-height:330px;overflow-y:auto;font-family:'Space Mono',monospace;font-size:12px;line-height:1.7}
.ev{padding:4px 0;border-bottom:1px solid rgba(236,237,230,.05);display:flex;gap:10px}
.ev .t{color:var(--mute-2);flex-shrink:0}
.ev .k{width:60px;flex-shrink:0;font-weight:700;letter-spacing:.08em}
.k.entry{color:var(--cyn)}.k.exit{color:var(--pur)}
.k.scan{color:var(--grn)}.k.error{color:var(--red)}
.ok{color:var(--grn)}.no{color:var(--red)}.mid{color:var(--amb)}

button{background:transparent;color:var(--paper);border:1px solid var(--line-2);
  font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  padding:9px 16px;cursor:pointer;transition:border-color .2s,color .2s}
button:hover{border-color:var(--gold);color:var(--gold)}
button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}

.note{color:var(--mute);font-weight:300;font-size:14px;line-height:1.6;margin-top:12px}
.note b{color:var(--paper);font-weight:500}

/* ---------- workup: catalog pattern from the public site ---------- */
#workup{margin-top:44px}
.wtitle{font-family:'Saira Condensed',sans-serif;font-weight:700;
  font-size:clamp(28px,4.2vw,48px);text-transform:uppercase;line-height:1;
  letter-spacing:-.005em;margin-top:12px}
.wtitle .g{color:var(--gold)}
.wlede{color:var(--mute);font-weight:300;font-size:16.5px;line-height:1.6;max-width:780px;margin-top:16px}
.wlede b{color:var(--paper);font-weight:500}
.subhead{font-family:'Space Mono',monospace;font-size:11.5px;letter-spacing:.24em;
  color:var(--gold);text-transform:uppercase;margin:34px 0 12px;padding-top:24px;
  border-top:1px solid var(--line)}
.cards{display:grid;gap:14px}
details.card{padding:0;transition:border-color .25s,background .25s}
details.card:hover{border-color:var(--gold)}
details.card[open]{border-color:var(--line-2);background:var(--ink-3)}
summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:20px;
  padding:20px 24px;outline:none}
summary::-webkit-details-marker{display:none}
summary:focus-visible{outline:2px solid var(--gold);outline-offset:-2px}
.c-num{font-family:'Space Mono',monospace;font-size:11.5px;color:var(--gold);
  letter-spacing:.1em;flex:none;width:46px}
.c-main{flex:1;min-width:0}
.c-title{font-family:'Saira Condensed',sans-serif;font-weight:600;font-size:22px;
  text-transform:uppercase;letter-spacing:.01em;line-height:1.05;display:block}
.c-sub{font-size:14.5px;color:var(--mute);font-weight:300;margin-top:4px;display:block}
.c-toggle{flex:none;width:22px;height:22px;position:relative;color:var(--mute-2)}
.c-toggle::before,.c-toggle::after{content:"";position:absolute;background:currentColor;
  transition:transform .25s}
.c-toggle::before{top:10px;left:3px;right:3px;height:2px}
.c-toggle::after{left:10px;top:3px;bottom:3px;width:2px}
details[open] .c-toggle::after{transform:scaleY(0)}
details[open] .c-toggle{color:var(--gold)}
.c-body{padding:0 24px 26px 70px;color:var(--mute);font-weight:300;font-size:15px;line-height:1.65}
.c-body p{margin-bottom:12px}
.c-body b{color:var(--paper);font-weight:500}
.c-body a{color:var(--gold)}
@media(max-width:640px){.c-body{padding-left:24px}}
.c-body .who{font-family:'Space Mono',monospace;font-size:11.5px;letter-spacing:.06em;
  color:var(--paper);text-transform:uppercase;display:block;margin:16px 0 10px}
.c-body .who span{color:var(--gold)}
.feats{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.feat{font-family:'Space Mono',monospace;font-size:10.5px;letter-spacing:.05em;color:var(--paper);
  border:1px solid var(--line-2);padding:6px 11px;background:rgba(201,162,39,.04)}
.feat.spec{color:var(--gold);border-color:var(--gold)}

.wt{width:100%;border-collapse:collapse;margin:14px 0 18px;font-size:13.5px;
  font-family:'Barlow',system-ui,sans-serif}
.wt th{font-family:'Space Mono',monospace;font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--gold);text-align:left;font-weight:400;
  padding:8px 10px;border-bottom:1px solid var(--line-2)}
.wt td{padding:8px 10px;border-bottom:1px solid rgba(236,237,230,.06);vertical-align:top}
.wt td:first-child{color:var(--paper)}
.wt .src{display:block;font-family:'Space Mono',monospace;font-size:10.5px;
  letter-spacing:.08em;color:var(--mute-2);margin-top:10px}
.pipe{font-family:'Space Mono',monospace;font-size:12.5px;line-height:1.85;color:var(--mute);
  background:var(--ink);border:1px solid var(--line);padding:14px 16px;margin:14px 0 18px;
  white-space:pre-wrap;overflow-x:auto}
.pipe b{color:var(--gold);font-weight:400}
.callout{border-left:2px solid var(--gold);background:rgba(201,162,39,.05);padding:14px 18px;margin:16px 0}
.callout.warn{border-left-color:var(--red);background:rgba(228,105,90,.06)}
.callout .h{font-family:'Space Mono',monospace;font-size:10.5px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--gold);display:block;margin-bottom:6px}
.callout.warn .h{color:var(--red)}
.callout p{margin-bottom:10px}
.callout p:last-child{margin-bottom:0}

footer{border-top:1px solid var(--line);margin-top:40px;padding-top:22px;
  font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.12em;color:var(--mute-2);
  text-transform:uppercase;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}

@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body>

<header class="top">
  <span class="eyebrow">Gray Tech Solutions &middot; Situational Awareness</span>
  <h1><svg class="mark" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><polygon points="32,7 57,32 32,57 7,32" fill="none" stroke="#C9A227" stroke-width="5"/><polygon points="32,20 44,32 32,44 20,32" fill="#C9A227"/><polygon points="32,27 37,32 32,37 27,32" fill="#0D0F0C"/></svg>GrayTech <span class="g">Security</span>
      <span class="badge live" id="conn">connecting</span></h1>
  <p class="tagline">Face detection &amp; recognition over a monitored scene, with a match
    threshold derived from observed traffic rather than a generic default.</p>
</header>

<div class="kpis" id="kpis"></div>

<div class="row">
  <div class="card grow">
    <div class="cardhead">
      <div><span class="eyebrow">Live</span><h2>Scene</h2></div>
      <button onclick="togglePause()">Pause / Resume</button>
    </div>
    <canvas id="scene" width="960" height="540"></canvas>
    <div class="note" id="scaninfo"></div>
  </div>
  <div class="card side">
    <div class="cardhead"><div><span class="eyebrow">Stream</span><h2>Event log</h2></div></div>
    <div id="log"></div>
  </div>
</div>

<div class="card" style="margin-top:14px">
  <div class="cardhead">
    <div><span class="eyebrow">Calibration</span><h2>Recognition threshold &mdash; self-calibrating</h2></div>
    <span id="calBadge" class="badge sim" style="display:none"></span>
  </div>
  <div class="note" style="margin-top:6px">
    The match threshold decides when a face counts as an enrolled person. It is
    not a guess: every score the system produces is logged, and the operating
    point below is derived from where errors actually occur in this traffic.
  </div>
  <div class="kpis" style="margin-top:10px" id="calKpis"></div>
  <div id="calRec" class="note" style="margin-top:8px"></div>
  <div id="calProfiles" class="note" style="margin-top:6px"></div>
</div>

<div class="card" style="margin-top:14px">
  <div class="cardhead"><div><span class="eyebrow">Scope</span><h2>About this demo</h2></div></div>
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

<!-- ============ PRESENTATION WORKSPACE WORKUP ============ -->
<section id="workup">
  <span class="eyebrow">Field Workup &middot; Aerial Biometrics</span>
  <h2 class="wtitle">The recognition <span class="g">stack that actually holds up</span></h2>
  <p class="wlede">There is no single open-source &ldquo;drone facial-recognition product.&rdquo;
    The stack actually in use in 2026 is a <b>detect &rarr; optically zoom &rarr; align &rarr;
    embed &rarr; search</b> pipeline, not a Haar-cascade Tello demo. Digital zoom does not
    substitute for optical zoom: it interpolates pixels that were never captured. Identity
    matching from the air is a pixel-budget problem first and a model problem second.</p>

  <div class="callout warn">
    <span class="h">Read this first &mdash; scope</span>
    <p>This workup describes a <b>different airframe</b>: a 7&ndash;10&Prime; quad carrying a
      10&ndash;30&times; optical gimbal and a Jetson Orin NX. It is <b>not</b> a measurement of
      the console above, which is a fixed ground camera running <code>buffalo_s</code> on CPU
      via onnxruntime. Nothing here raises the accuracy figure shown in this console, and the
      figures quoted below come from cited third-party UAV studies, not from this system.</p>
    <p>It is included because it is the honest answer to &ldquo;how far can this go, and what
      would it actually take?&rdquo; &mdash; and because its licensing and legal limits are
      load-bearing for anyone planning to build on it.</p>
  </div>

  <div class="subhead">01 / What actually works</div>
  <div class="cards">
    <details class="card" open>
      <summary>
        <span class="c-num">01</span>
        <span class="c-main">
          <span class="c-title">The pixel budget</span>
          <span class="c-sub">Most drone + face GitHub repos are face <em>tracking</em>, not recognition.</span>
        </span>
        <span class="c-toggle"></span>
      </summary>
      <div class="c-body">
        <p>Most drone + face repos (Tello-Face-Recognition, Haar cascades,
          <code>face_recognition</code>/dlib) keep a blob in frame. They cannot identify a
          person at range. Real identification needs roughly:</p>
        <table class="wt">
          <tr><th>Task</th><th>Face width in the image</th><th>Typical IOD (eye-to-eye)</th></tr>
          <tr><td>Detect a face</td><td>~20 px</td><td>~8&ndash;12 px</td></tr>
          <tr><td>Recognize / verify</td><td>~40&ndash;80 px (IEC 62676-4: ~40 px across the face for ID)</td><td>~50&ndash;90 px historically</td></tr>
          <tr><td>Robust ID (pose, motion, backlight)</td><td>80&ndash;120+ px</td><td>70+ px</td></tr>
        </table>
        <p>A DJI Mini-class 4K wide camera with <b>no optical zoom</b> produced these ArcFace results:</p>
        <table class="wt">
          <tr><th>Distance</th><th>Face size</th><th>ArcFace accuracy (cosine, dynamic threshold)</th></tr>
          <tr><td>2 m</td><td>125&times;170 px</td><td>~92%</td></tr>
          <tr><td>5 m</td><td>80&times;98 px</td><td>~92%</td></tr>
          <tr><td>10 m</td><td>38&times;44 px</td><td>~81%</td></tr>
          <tr><td>15 m</td><td>25&times;31 px</td><td>~72%</td></tr>
          <tr><td>30 m</td><td>14&times;19 px</td><td>~56% &mdash; near chance at gallery scale</td></tr>
          <tr><td colspan="3"><span class="src">Source: Sensors 2023 UAV study, wide 4K camera, no optical zoom.</span></td></tr>
        </table>
        <p><b>Depression angle is the other killer.</b> A drone looking 40&ndash;60&deg; down sees
          forehead and hair, not a frontal face. You want a standoff orbit, not a hover-over:
          20&ndash;40 m slant range, shallow look-down, optical zoom to fill the face.</p>
      </div>
    </details>

    <details class="card">
      <summary>
        <span class="c-num">02</span>
        <span class="c-main">
          <span class="c-title">Software &mdash; the stack that is SOTA</span>
          <span class="c-sub">InsightFace SCRFD + ArcFace, quality-gated, with a cheap person-find in front.</span>
        </span>
        <span class="c-toggle"></span>
      </summary>
      <div class="c-body">
        <p><b>Winner: InsightFace pipeline.</b> The de-facto open-source production stack
          (NIST-class ArcFace lineage, ONNX, TensorRT-friendly).</p>
        <table class="wt">
          <tr><th>Stage</th><th>Model</th><th>Why</th></tr>
          <tr><td>Person find (long range)</td><td>YOLOv8/v11 (person class) or YOLO-World</td><td>Faces are too small at 100 m. Find the body first, then zoom.</td></tr>
          <tr><td>Face detect</td><td>SCRFD-10GF (buffalo_l) or SCRFD-2.5GF (buffalo_m)</td><td>Best speed/accuracy for small-to-medium faces. RetinaFace is slower; Haar is obsolete.</td></tr>
          <tr><td>Landmarks / align</td><td>5-point or 106-point from the same pack</td><td>Affine-align to 112&times;112 before embedding. Do not skip this.</td></tr>
          <tr><td>Embedding</td><td>ArcFace ResNet-50 @ WebFace600K (buffalo_l)</td><td>512-D vector. LFW 99.83, IJB-C 97.25. Default production pack.</td></tr>
          <tr><td>Heavy gallery / hard pose</td><td>antelopev2 (R100 @ Glint360K)</td><td>Better on profile / age / domain shift. Heavier.</td></tr>
          <tr><td>Edge / low power</td><td>buffalo_s / buffalo_sc (MobileFaceNet)</td><td>Drop accuracy, especially on aerial/non-frontal. Only if Orin Nano / Hailo budget.</td></tr>
          <tr><td>1:N search</td><td>FAISS (GPU on Orin) or cosine on a small gallery</td><td>Sub-ms for tens of thousands of embeddings.</td></tr>
          <tr><td>Track (don&rsquo;t re-embed every frame)</td><td>ByteTrack / BoT-SORT</td><td>Embed on new tracks + every N frames / quality spike. This is how AGX Orin papers hit 200&ndash;290 FPS.</td></tr>
          <tr><td>Quality gate</td><td>Blur (Laplacian), pose (yaw/pitch from landmarks), face size, occlusion</td><td>Reject bad crops. This matters more than model choice in the air.</td></tr>
        </table>
        <p>A 2023 UAV face-verification paper compared ArcFace, FaceNet512, SFace, Dlib and
          VGG-Face. <b>ArcFace and FaceNet512 won</b> &mdash; ArcFace being the one with a
          maintained ONNX zoo and a TensorRT path.</p>
        <p class="pipe"><b>How the drone loop actually runs</b>
RTSP/H.265 from gimbal
  &rarr; NVDEC hardware decode on Jetson
  &rarr; YOLO person detect (wide / low zoom)
  &rarr; if person: command gimbal absolute zoom + track
  &rarr; when face &ge; ~80 px: SCRFD &rarr; align &rarr; ArcFace
  &rarr; cosine vs gallery (threshold by distance/zoom)
  &rarr; ByteTrack ID persists across frames
  &rarr; MAVLink / UDP: alert + snapshot + lat/lon/alt + zoom</p>
        <p>That last part is important: <b>do not run ArcFace on every 4K frame of empty
          sky.</b> Detect people cheaply, zoom, then spend GPU on the crop.</p>
        <table class="wt">
          <tr><th>Wrapper</th><th>Role</th><th>Use it?</th></tr>
          <tr><td>InsightFace</td><td>Core models + Python FaceAnalysis</td><td>Yes. This is the engine.</td></tr>
          <tr><td>CompreFace</td><td>Docker REST API over FaceNet/InsightFace</td><td>Yes if you want a service, not a library.</td></tr>
          <tr><td>DeepFace</td><td>Easy Python wrapper, many backends</td><td>Prototyping only. Too heavy/slow onboard.</td></tr>
          <tr><td>UniFace (2025, MIT)</td><td>ONNX Runtime, ArcFace + MobileFace</td><td>Good newer wrapper.</td></tr>
          <tr><td>face_recognition (dlib)</td><td>128-D HOG/CNN</td><td>No, for this job.</td></tr>
          <tr><td>YOLO + ByteTrack</td><td>Person detect / track</td><td>Yes, in front of the face models.</td></tr>
        </table>
        <div class="callout warn">
          <span class="h">Licence trap &mdash; read before you build</span>
          <p>InsightFace <b>code</b> is MIT. The <b>pretrained packs</b> (buffalo_l,
            antelopev2, etc.) are <b>non-commercial research only</b>. Commercial deployment
            needs a licence from InsightFace. For a clean commercial path, either licence
            those weights, train your own ArcFace on a dataset you hold rights to, or use a
            commercially licensed SDK.</p>
        </div>
        <p class="who">Aerial-specific details people miss</p>
        <div class="feats">
          <span class="feat spec">Dynamic thresholds &mdash; same cosine cutoff fails at 5 m and 40 m</span>
          <span class="feat">Distance/zoom-binned thresholds; treat far matches as hints, not IDs</span>
          <span class="feat">Enroll 5&ndash;15 shots: frontal, &plusmn;30&deg; yaw, slight down-pitch, sunglasses on/off</span>
          <span class="feat">One LinkedIn headshot will fail from the air</span>
          <span class="feat">Don&rsquo;t downscale 4K to 640 for tiny faces &mdash; run YOLO/SCRFD on a high-res tile or zoomed crop</span>
          <span class="feat">Super-resolution (Real-ESRGAN) after optical zoom is optional and often hurts embeddings. Quality-gate first.</span>
        </div>
      </div>
    </details>

    <details class="card">
      <summary>
        <span class="c-num">03</span>
        <span class="c-main">
          <span class="c-title">Hardware &mdash; the build that matches</span>
          <span class="c-sub">Optical zoom is the recognition sensor. Everything else is secondary.</span>
        </span>
        <span class="c-toggle"></span>
      </summary>
      <div class="c-body">
        <p><b>Do not put this on the flight controller.</b> PX4/ArduPilot flies. A companion
          computer sees.</p>
        <table class="wt">
          <tr><th>Piece</th><th>Pick</th><th>Why</th></tr>
          <tr><td>Airframe</td><td>7&ndash;10&Prime; class industrial quad or small VTOL, 2&ndash;4 kg AUW</td><td>Needs payload + 15&ndash;40 W compute + zoom gimbal. Tello/Mini cannot do this.</td></tr>
          <tr><td>Autopilot</td><td>Holybro Pixhawk 6X / 6C running PX4 or ArduPilot</td><td>MAVLink to companion, gimbal mount, Ethernet optional.</td></tr>
          <tr><td>Companion (AI)</td><td>NVIDIA Jetson Orin NX 16GB Super (~157 TOPS, 10&ndash;40 W)</td><td>TensorRT, NVDEC, CUDA FAISS, DeepStream. This is the right accelerator.</td></tr>
          <tr><td>Carrier</td><td>ARK Jetson PAB or Auvidea JNX12x PAB</td><td>Pixhawk Autopilot Bus + Jetson on one board. NDAA options exist (ARK).</td></tr>
          <tr><td>Compact alt</td><td>Seeed reComputer Mini (Orin Nano/NX)</td><td>Proven PX4 object-tracking wiki.</td></tr>
          <tr><td>Camera</td><td>SIYI ZR30 (30&times; optical / 180&times; hybrid, 4K, ~668 g)</td><td>Optical zoom is the recognition sensor. Ethernet RTSP + UDP zoom control.</td></tr>
          <tr><td>Budget camera</td><td>SIYI ZR10 (10&times; optical / 30&times; hybrid, 2K, ~381 g)</td><td>Best compact optical zoom for lighter quads.</td></tr>
          <tr><td>Pro camera</td><td>Viewpro Mini-H30T / H30T (30&times; STARVIS + thermal + LRF)</td><td>Night + ID + range-to-target. Heavier and 5&ndash;10&times; the price.</td></tr>
          <tr><td>Link</td><td>SIYI HM30 / MK32 or equivalent Ethernet video</td><td>Gimbal RTSP must land on the Jetson, not only the GCS.</td></tr>
          <tr><td>Storage</td><td>NVMe 256 GB+ on the Jetson</td><td>Embeddings, snapshots, TensorRT engines.</td></tr>
          <tr><td>Power</td><td>6S pack, UBEC 12&ndash;24 V to Jetson (XT30 on WeAct-style carriers)</td><td>Orin NX + gimbal + radio is a real power budget. Active cooling required.</td></tr>
        </table>
        <p><b>SIYI ZR30</b> (the default pick): 30&times; optical, 4.5&ndash;148 mm, 8 MP 1/2.7&Prime;
          Sony CMOS, 4K. At 30&times;, HFOV ~2.1&deg; &mdash; a face at 80&ndash;120 m can reach
          identification pixel counts that a wide 4K camera only gets at ~5 m. Ethernet RTSP +
          UART/UDP zoom/gimbal, so the Jetson can command zoom from the YOLO box. ~5&ndash;12 W,
          668 g, 3S&ndash;6S, PX4/ArduPilot compatible.</p>
        <p><b>SIYI ZR10:</b> 10&times; optical, 381 g, 2K. Sweet spot if you fly closer
          (inspection, SAR in woods, &lt;50 m).</p>
        <p><b>SIYI A8 mini:</b> 4K, 95 g, digital 6&times; only. Great FPV/inspection cam.
          Wrong as the ID sensor.</p>
        <p><b>Viewpro 30&times; STARVIS (H30T / Mini-H30T):</b> if you need night colour +
          thermal cueing + laser range. Thermal finds the person; EO zoom IDs them. This is how
          actual ISR payloads work.</p>
        <p>Gimbal stability of &plusmn;0.01&deg; is not a vanity spec. At 30&times;, 0.2&deg; of
          jitter smears the face and kills embeddings. Mechanical 3-axis is mandatory.</p>
        <table class="wt">
          <tr><th>Accelerator</th><th>TOPS</th><th>Power</th><th>Drone verdict</th></tr>
          <tr><td>Jetson Orin NX 16GB Super</td><td>~157</td><td>10&ndash;40 W</td><td>Best overall. TensorRT + NVDEC + 16 GB for gallery + DeepStream.</td></tr>
          <tr><td>Jetson Orin Nano Super</td><td>~67</td><td>7&ndash;25 W</td><td>Fine for YOLO + buffalo_s/buffalo_m. Tight for 4K + buffalo_l + FAISS.</td></tr>
          <tr><td>Jetson AGX Orin</td><td>200&ndash;275</td><td>15&ndash;60 W</td><td>Ground station or large UAV. Published 200&ndash;290 FPS face pipelines. Overkill / hot / heavy on a small quad.</td></tr>
          <tr><td>ModalAI VOXL 2</td><td>~15 TOPS, 16 g, PX4 onboard</td><td>~5&ndash;10 W</td><td>Best drone-native computer. Great for YOLO/VIO. Weak for buffalo_l at 4K. Pair with ground-side InsightFace if you must.</td></tr>
          <tr><td>Hailo-8 (M.2)</td><td>~26</td><td>~2.5 W</td><td>Needs a host. Good YOLO offload. Awkward InsightFace path. Don&rsquo;t use as the only brain.</td></tr>
          <tr><td>Google Coral</td><td>~4</td><td>&mdash;</td><td>Too weak for modern ArcFace. Dead end.</td></tr>
          <tr><td>Raspberry Pi 5</td><td>0 NPU</td><td>&mdash;</td><td>Prototyping video ingest only.</td></tr>
          <tr><td>Qualcomm QCS6490 (Advantech ASR-D501)</td><td>~12</td><td>&lt;10 W</td><td>New 2026 UAV mission computer. Person detect / VIO, not ArcFace-R50.</td></tr>
        </table>
        <p><b>Buy the Orin NX.</b> Convert InsightFace ONNX &rarr; TensorRT FP16. Run YOLO on
          DLA, ArcFace on GPU, decode on NVDEC. That split is what the 2025/2026 Orin face
          papers actually measured.</p>
        <p><b>Ground-station alternative</b> (sometimes smarter): stream H.265 to a laptop/3080
          and run antelopev2 there. Onboard you only do YOLO + zoom + snapshot. Latency and radio
          bandwidth become the limit; identity quality goes up.</p>
      </div>
    </details>

    <details class="card">
      <summary>
        <span class="c-num">04</span>
        <span class="c-main">
          <span class="c-title">Reference build</span>
          <span class="c-sub">What I would actually assemble, and what it weighs.</span>
        </span>
        <span class="c-toggle"></span>
      </summary>
      <div class="c-body">
        <p class="who">Air side</p>
        <div class="feats">
          <span class="feat">Custom 10&Prime; carbon quad or small VTOL, 6S</span>
          <span class="feat">Pixhawk 6X (PX4)</span>
          <span class="feat">ARK Jetson PAB + Orin NX 16GB + NVMe + active heatsink</span>
          <span class="feat">SIYI ZR30 on the gimbal port, Ethernet to Jetson (192.168.144.x typical SIYI subnet)</span>
          <span class="feat">SIYI or Herelink-class HD link for GCS</span>
          <span class="feat">Here3+ GNSS, rangefinder, 4G/5G backup modem for alerts</span>
        </div>
        <p class="who">Software on Jetson (JetPack 6.x)</p>
        <div class="feats">
          <span class="feat">DeepStream or GStreamer <code>nvv4l2decoder</code> on RTSP</span>
          <span class="feat">Ultralytics YOLO11n/s TensorRT &mdash; person</span>
          <span class="feat">InsightFace buffalo_l &rarr; TensorRT</span>
          <span class="feat">FAISS-GPU gallery</span>
          <span class="feat">ByteTrack</span>
          <span class="feat">mavsdk / MAVROS2: gimbal zoom, ROI, geolocation of the pixel using aircraft attitude + gimbal angles (add an LRF if you need real coordinates)</span>
          <span class="feat spec">Quality filters before match; log crop + score + GPS, don&rsquo;t just fire a name</span>
        </div>
        <p class="who">Weight / power ballpark</p>
        <table class="wt">
          <tr><th>Item</th><th>Mass</th><th>Power</th></tr>
          <tr><td>Orin NX + carrier + cooler</td><td>~150&ndash;250 g</td><td>15&ndash;25 W typical</td></tr>
          <tr><td>SIYI ZR30</td><td>~670 g</td><td>5&ndash;12 W</td></tr>
          <tr><td>With FC + radio</td><td colspan="2">Plan ~1 kg payload, ~30 W avionics. That is a real aircraft, not a toy.</td></tr>
        </table>
        <p>If the aircraft must stay light: ZR10 + Orin Nano Super, person-detect onboard,
          identity only when the face crop is large enough, or offload identity to the ground.</p>
      </div>
    </details>

    <details class="card">
      <summary>
        <span class="c-num">05</span>
        <span class="c-main">
          <span class="c-title">What will still fail</span>
          <span class="c-sub">Even with this stack. Especially with this stack.</span>
        </span>
        <span class="c-toggle"></span>
      </summary>
      <div class="c-body">
        <div class="feats">
          <span class="feat">Hovering 80 m AGL with a wide camera and &ldquo;AI digital zoom&rdquo;</span>
          <span class="feat">Hats, masks, strong backlight, motion blur at 30&times;</span>
          <span class="feat">One enrollment photo</span>
          <span class="feat">Treating a 0.55 cosine at 150 m as an identity</span>
          <span class="feat spec">Running recognition as a civil surveillance tool without a legal basis</span>
        </div>
        <div class="callout">
          <span class="h">Most missions do not need identity</span>
          <p>For search and rescue you usually do not need identity. <b>YOLO-person + thermal +
            optical zoom</b> is the useful system &mdash; TEXSAR&rsquo;s ADIAT is the open-source
            SAR image tool, and that is person detection, not FR. Use ArcFace only when you have
            a consented, small gallery (missing person, authorised crew, etc.).</p>
        </div>
      </div>
    </details>

    <details class="card">
      <summary>
        <span class="c-num">06</span>
        <span class="c-main">
          <span class="c-title">Legal &amp; model-licence notes</span>
          <span class="c-sub">Not optional, and not a lawyer.</span>
        </span>
        <span class="c-toggle"></span>
      </summary>
      <div class="c-body">
        <div class="callout warn">
          <span class="h">Licence</span>
          <p>InsightFace weights <b>&ne; free for commercial products</b>. The code is MIT; the
            packs are research-only unless licensed.</p>
        </div>
        <div class="callout warn">
          <span class="h">Regulation</span>
          <p>Aerial biometric ID is high-risk under the <b>EU AI Act</b>, and in the US is a mix of
            FAA ops rules plus state biometric laws (e.g. BIPA). Government use has additional
            Fourth Amendment / policy constraints.</p>
          <p>Many countries restrict both drone overflight and covert biometrics. Treat this as a
            <b>research / SAR / consented-security architecture</b>, not a general-purpose crowd
            scanner.</p>
        </div>
      </div>
    </details>
  </div>

  <div class="callout" style="margin-top:26px">
    <span class="h">Bottom line</span>
    <p>The open-source solution used by people who care about accuracy is
      <b>YOLO (find) + optical zoom gimbal (pixels) + InsightFace SCRFD/ArcFace (identify) +
      FAISS (search) + ByteTrack (temporal)</b>, running TensorRT on a Jetson Orin NX, with a
      10&ndash;30&times; optical camera (SIYI ZR10/ZR30 or Viewpro 30&times;). Everything else is
      either a toy tracker or a ground-station batch job.</p>
  </div>
</section>

<footer>
  <span>GrayTech Security &mdash; part of the XMRT DAO ecosystem</span>
  <span>Style aligned to graytechsolutions.dev</span>
</footer>

<script>
const cv = document.getElementById('scene'), cx = cv.getContext('2d');
let lastEvents = [], paused = false;

// The 2D canvas cannot read CSS custom properties, so pull the palette out of the
// stylesheet once instead of hard-coding hex here. That keeps :root the single
// source of truth: restyle the console and the scene follows. Translucent strokes
// use the 8-digit hex form, C.gold+'22' == the old rgba(201,162,39,.13).
const CS=getComputedStyle(document.documentElement);
const pv=n=>CS.getPropertyValue(n).trim();
const C={ink:pv('--ink'),panel:pv('--ink-3'),gold:pv('--gold'),paper:pv('--paper'),
         mute:pv('--mute'),grn:pv('--grn'),red:pv('--red')};

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
  cx.fillStyle=C.ink; cx.fillRect(0,0,W,H);
  cx.strokeStyle=C.gold+'22';
  for(let gx=0; gx<W; gx+=60){ cx.beginPath(); cx.moveTo(gx,0); cx.lineTo(gx,H); cx.stroke(); }
  for(let gy=0; gy<H; gy+=60){ cx.beginPath(); cx.moveTo(0,gy); cx.lineTo(W,gy); cx.stroke(); }
  // scan line
  const sx=s.scan_x;
  const grd=cx.createLinearGradient(sx-26,0,sx+26,0);
  grd.addColorStop(0,C.gold+'00');
  grd.addColorStop(.5,C.gold+'33');
  grd.addColorStop(1,C.gold+'00');
  cx.fillStyle=grd; cx.fillRect(sx-26,0,52,H);
  cx.strokeStyle=C.gold+'BF'; cx.beginPath();
  cx.moveTo(sx+.5,0); cx.lineTo(sx+.5,H); cx.stroke();
  cx.fillStyle=C.gold+'D9'; cx.font='11px "Space Mono",monospace';
  cx.fillText('SCAN ZONE', sx-38, 16);

  // people
  s.present.forEach(p=>{
    const x=p.x, y=p.y, R=26;
    const col = !p.scanned ? C.mute : (p.matched ? C.grn : C.red);

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
      } else { cx.fillStyle=C.panel; cx.fillRect(x-R,y-R,d,d); }
    } else if(p.live && p._live){
      // webcam visitor: draw the live frame inside the head
      const img = p._live; const d=R*2;
      if(img.complete && img.naturalWidth){
        const sc = Math.max(d/img.naturalWidth, d/img.naturalHeight);
        cx.drawImage(img, x-d/2, y-d/2, img.naturalWidth*sc, img.naturalHeight*sc);
      } else { cx.fillStyle=C.panel; cx.fillRect(x-R,y-R,d,d); }
    } else {
      cx.fillStyle = p.scanned ? C.panel : C.ink;
      cx.fillRect(x-R,y-R,R*2,R*2);
    }
    cx.restore();

    // ring: colour carries the recognition state, and stays visible over a photo
    cx.strokeStyle=col; cx.lineWidth= p.scanned ? 3 : 1.5;
    cx.beginPath(); cx.arc(x,y,R,0,Math.PI*2); cx.stroke();

    if(p.scanned){
      cx.fillStyle=col; cx.font='bold 15px "Saira Condensed",sans-serif';
      cx.textAlign='center'; cx.fillText(p.name, x, y-34);
      if(p.confidence!=null){
        cx.font='11px "Space Mono",monospace';
        cx.fillStyle=C.paper+'D9';
        cx.fillText(p.confidence.toFixed(3), x, y-20);
      }
    } else {
      cx.fillStyle=C.mute+'E6'; cx.font='11px "Space Mono",monospace';
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
  draw(d.state); drawKpis(d.state); drawLog(d.events); drawCalibration(d.state.calibration);
};

function drawCalibration(c){
  if(!c) return;
  const badge=document.getElementById('calBadge');
  const rec=c.recommend||{};
  if(rec.ok && Math.abs((rec.delta||0))>=0.01){
    badge.style.display='';
    badge.textContent='adjustment suggested';
  } else { badge.style.display='none'; }

  const items=[
    ['Threshold', (c.threshold!=null?c.threshold.toFixed(3):'—'),
      rec.ok?'var(--amb)':''],
    ['Observations', c.n!=null?c.n:'—', ''],
    ['False accepts', c.false_accept!=null?c.false_accept:'—',
      c.false_accept?'var(--red)':'var(--grn)'],
    ['False rejects', c.false_reject!=null?c.false_reject:'—',
      c.false_reject?'var(--amb)':'var(--grn)'],
    ['Median score', c.p50!=null?c.p50.toFixed(3):'—', ''],
    ['Accuracy', c.accuracy!=null?(c.accuracy*100).toFixed(1)+'%':'—',
      c.accuracy!=null&&c.accuracy<1?'var(--amb)':'var(--grn)'],
  ];
  document.getElementById('calKpis').innerHTML = items.map(([l,v,col])=>
    `<div class="kpi"><div class="lab">${l}</div><div class="val" style="color:${col||'var(--txt)'};font-size:20px">${v}</div></div>`
  ).join('');

  const box=document.getElementById('calRec');
  if(rec.ok){
    box.innerHTML = '<b class="mid">Recommendation:</b> move threshold '+
      rec.current.toFixed(3)+' → <b>'+rec.recommended.toFixed(3)+'</b> &nbsp;'+ rec.reason +
      ' <span style="color:var(--mut)">('+rec.caveat+')</span>';
  } else {
    box.innerHTML = '<span style="color:var(--mut)">Not enough evidence yet: '+
      (rec.reason||'')+'</span>';
  }

  const learned=Object.entries(c.profiles||{});
  document.getElementById('calProfiles').innerHTML = learned.length
    ? '<b style="color:var(--grn)">Adaptive profiles:</b> '+ learned.map(([k,v])=>
        k+' <span style="color:var(--mut)">('+(v.samples||0)+' sample'+
        ((v.samples===1)?'':'s')+')</span>').join(', ') +
      ' — these enrolled identities have absorbed corrections and are now matched against a refined vector.'
    : '<span style="color:var(--mut)">No adaptive corrections recorded yet. Correcting a misidentification enriches that person\'s profile immediately, without retraining the network.</span>';
}

async function togglePause(){
  const r=await fetch('/api/pause',{method:'POST'});
  const d=await r.json(); paused=!d.running;
}
</script></body></html>"""

app = application