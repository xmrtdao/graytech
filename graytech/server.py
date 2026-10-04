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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Reuse the service that already exists. Same model, same matcher, same
# threshold - importing rather than reimplementing is the point, so a change to
# app/ changes this dashboard too.
import app as faceapp
from app import IDENTITY_DIR, face as recogniser
from app.personfind import Tracker, personfinder

DATASET_DEFAULT = r"C:\Users\PureTrek\Desktop\Faces\Faces"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCENE_W, SCENE_H = 960, 540
SCAN_X = SCENE_W * 0.5          # the scan line, people walk through here
HEAD_R = 26                      # head radius on the stick figure
BODY_BOTTOM = 96                 # the stick figure's feet, at y + 96
# Spawn just off the leading edge and cull just past the trailing one. This is
# the walk-on effect; the important part is that these in-flight visitors are
# NOT counted, because the dashboard number has to equal the number of people
# actually on screen.
SPAWN_X = -(HEAD_R + 30)
CULL_X = SCENE_W + HEAD_R + 40


def in_frame(v) -> bool:
    """True while any part of this visitor is actually on the canvas.

    The KPI used to be len(self.visitors), which counts people mid-approach at
    x=-52 and people already walking off at x=1070 on a canvas that spans
    0..960. It read 7 while five figures were on screen. Count what is drawn.
    """
    return (v.x + HEAD_R >= 0 and v.x - HEAD_R <= SCENE_W
            and v.y + BODY_BOTTOM >= 0 and v.y - HEAD_R <= SCENE_H)

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
    # Pixel budget, measured on the crop that actually reached the model rather
    # than inferred from the scene. This is the number the rig spec turns on.
    face_px: Optional[float] = None
    iod_px: Optional[float] = None
    sharpness: Optional[float] = None
    band: str = "unmeasured"
    # Set when the quality gate refused to score this crop. A refused crop is
    # NOT an unknown person: counting it as one is how a system ends up
    # reporting "unknown" for a face it never had the pixels to judge.
    gated: bool = False
    gate_reason: str = ""
    # Stage 1's verdict on this person, counted without reference to the face
    # pipeline. Kept in its own fields so a YOLO count can never be mistaken for
    # a face result, or vice versa.
    yolo_persons: int = 0
    yolo_ok: bool = False
    yolo_ms: float = 0.0

    def tick(self, dt: float) -> bool:
        """Advance position. Returns False when they have left the scene."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        # gentle vertical drift so the walk is not a perfectly straight line
        self.y += np.sin((time.time() - self.born) * 1.4) * 0.06
        # Clamp the walk to the canvas. That sine term is a random walk applied
        # every tick, and over a long visit it accumulates: a visitor loitering
        # long enough ended up at y=536 on a 540-tall scene, feet at 632, off
        # the bottom of the picture while still counted as present.
        self.y = max(HEAD_R + 6.0,
                     min(SCENE_H - BODY_BOTTOM - 6.0, self.y))
        if self.x > CULL_X or self.x < SPAWN_X - 40:
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
            "unknown": 0, "gated": 0, "started_at": time.time(),
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
        # Latest YOLO person-find result for the live scene. Empty dict until
        # the camera is switched on, which is what the dashboard keys off.
        self.cam_persons: dict = {}
        # YOLO's own tallies, accumulated across the walk. These are counted by
        # the detector alone: no gallery, no face model, no match threshold.
        self.yolo_seen = 0          # frames where stage 1 ran
        self.yolo_person_hits = 0   # person detections across those frames
        self.yolo_per_visitor: dict = {}

    # -- helpers ----------------------------------------------------------
    def log(self, kind: str, **kw) -> None:
        ev = {"kind": kind, "t": time.time(), **kw}
        self.events.append(ev)

    def in_frame_count(self) -> int:
        """How many visitors are actually on the canvas right now."""
        return sum(1 for v in self.visitors if in_frame(v))

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
        # Stage 1 of the rig loop, run where there is an actual scene to read,
        # with the tracker advanced one frame per capture.
        self.cam_persons = personfinder.detect_bytes_tracked(data)
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
                x=SPAWN_X if from_left else CULL_X, y=y,
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
            x=SPAWN_X if from_left else CULL_X,
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
        # ── Stage 1: YOLO person-find, counted on its own ────────────────────
        # Deliberately independent of the face pipeline below. This number is
        # produced without the gallery, without the face model and without the
        # match threshold, so it stands on its own when the gallery is empty -
        # which is the normal case for search and rescue.
        yres = personfinder.detect_bytes(data)
        self.yolo_seen += 1
        self.yolo_person_hits += yres.get("count", 0)
        v.yolo_persons = yres.get("count", 0)
        v.yolo_ok = bool(yres.get("ok"))
        v.yolo_ms = yres.get("ms", 0.0)

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

        # ── Quality gate ────────────────────────────────────────────────────
        # The rig spec's most load-bearing line is that a quality gate matters
        # more than model choice. This is that gate, and it runs BEFORE the
        # match so a below-spec crop can never produce a confident identity.
        #
        # It has to come first. Scoring a 14px face and then discarding the
        # result still spends the calibration a wrong answer would have
        # poisoned, and "UNKNOWN" for a face we could not resolve is a claim.
        face = faces[0]
        v.face_px = face.get("face_px")
        v.iod_px = face.get("iod_px")
        v.sharpness = face.get("sharpness")
        v.band = face.get("band") or "unmeasured"

        reasons = []
        if v.face_px is not None and v.face_px < faceapp.PX_DETECT:
            reasons.append(f"face {v.face_px:.0f}px < {faceapp.PX_DETECT:.0f}px floor")
        if (v.sharpness is not None
                and v.sharpness < faceapp.SHARPNESS_MIN):
            reasons.append(
                f"sharpness {v.sharpness:.0f} < {faceapp.SHARPNESS_MIN:.0f}")
        if reasons:
            v.scanned = True
            v.matched = False
            v.gated = True
            v.gate_reason = "; ".join(reasons)
            self.stats["gated"] = self.stats.get("gated", 0) + 1
            self.log("gated", name=v.name, reason=v.gate_reason,
                     face_px=v.face_px, band=v.band)
            return
        v.gated = False

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
                 face_px=v.face_px, iod_px=v.iod_px, band=v.band,
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

            # Spawn cadence: fill the *visible* scene to max_present, not the
            # tracked list. Gating on len(self.visitors) let the cap be eaten by
            # people still walking on from off-canvas, so the scene topped out at
            # five visible figures while the counter said seven.
            if now >= spawn_at:
                if self.in_frame_count() < self.max_present:
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
        present = [{
            "name": v.name,
            "x": round(v.x, 1),
            "y": round(v.y, 1),
            "scanned": v.scanned,
            "matched": v.matched,
            "confidence": (round(v.confidence, 3)
                           if v.confidence is not None else None),
            "entering": v.entering,
            "live": v.live,
            # Pixel budget for this person, measured on the crop that reached
            # the model. The canvas draws the face width under the name so the
            # operator can see WHY a match is or is not trustworthy.
            "face_px": v.face_px,
            "iod_px": v.iod_px,
            "sharpness": v.sharpness,
            "band": v.band,
            "gated": v.gated,
            "gate_reason": v.gate_reason,
            # Stage 1, per visitor, on its own terms.
            "yolo_persons": v.yolo_persons,
            "yolo_ok": v.yolo_ok,
            "yolo_ms": v.yolo_ms,
            # Face portrait for the head of the stick figure. Sent only for
            # visitors who exist in a known file; live webcam frames carry
            # their own image instead.
            "face": None if v.live else thumb_data_uri(v.file),
            # A webcam visitor carries its own frame, shown inside the head so
            # the client can see it is live rather than a stored portrait.
            "liveFrame": ("data:image/jpeg;base64," +
                          base64.b64encode(v.frame).decode()) if (v.live and v.frame) else None,
        } for v in self.visitors]

        # Observed pixel budget for everyone in frame, next to the spec floors.
        # Reported side by side so the gap between what this rig delivers and
        # what the target rig needs is visible instead of asserted.
        measured = sorted(v.face_px for v in self.visitors if v.face_px)
        iods = sorted(v.iod_px for v in self.visitors if v.iod_px)
        pixel = {
            "spec": faceapp.pixel_budget(),
            "observed_min": measured[0] if measured else None,
            "observed_median": measured[len(measured) // 2] if measured else None,
            "iod_median": iods[len(iods) // 2] if iods else None,
            "gated_now": sum(1 for v in self.visitors if v.gated),
            "gated_total": self.stats.get("gated", 0),
        }
        return {
            "present": present,
            # The number on the dashboard is the number on the canvas. People
            # mid-walk at x=-52 or x=1070 are tracked but not counted here.
            "in_frame": self.in_frame_count(),
            # Of those in frame, how many are only partway on/off an edge. Worth
            # saying out loud: "7 in frame" with three of them half off the right
            # edge is a different claim from seven people fully on the canvas.
            "edge": sum(1 for v in self.visitors
                        if in_frame(v) and not (0 <= v.x <= SCENE_W)),
            "tracked": len(present),
            "stats": dict(self.stats),
            "pixel": pixel,
            # Stage 1 of the rig loop, and whether it is even loaded. The tallies here
            # are YOLO's alone - no gallery, no face model, no threshold.
            "person": {
                "available": personfinder.available,
                "detail": personfinder.detail,
                "frames": self.yolo_seen,
                "detections": self.yolo_person_hits,
                "hit_rate": (round(self.yolo_person_hits / self.yolo_seen, 2)
                             if self.yolo_seen else None),
                "live": self.cam_persons or None,
                "tracks": (self.cam_persons or {}).get("tracks") or [],
            },
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
    # Stage 1 of the rig loop. Loaded once here, never per request, and
    # non-fatal: if the weights are missing the console still runs and the
    # dashboard says so rather than pretending the stage exists.
    personfinder.load()
    scene.start()
    yield
    scene.stop()


application = FastAPI(title="GrayTech Security", version="0.1.0", lifespan=lifespan)

# Slide imagery, served from the repo rather than inlined as base64. A handful
# of JPEGs inlined would add ~4 MB to every page load; the OS already caches
# these on disk and the browser caches them by URL.
_ASSETS = Path(__file__).resolve().parent.parent / "deck-assets"
if _ASSETS.is_dir():
    application.mount("/deck-assets", StaticFiles(directory=str(_ASSETS)),
                      name="deck-assets")


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


@application.post("/api/persons")
async def persons(file: UploadFile = File(...)) -> dict:
    """
    Stage 1 on demand: YOLO person detection on one uploaded scene frame.

    This is the cheap gate that sits in front of the face models. It is the only
    route that exercises the stage on imagery you choose, which matters because
    YOLO wants a scene and returns near-nothing on a tight headshot crop.
    """
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    return personfinder.detect_bytes(data)


@application.get("/api/persons/status")
def persons_status() -> dict:
    """Whether stage 1 is loaded, and what it last saw."""
    return {"available": personfinder.available, "detail": personfinder.detail,
            "conf_thresh": personfinder.conf}


# ── Live HUD ingest ─────────────────────────────────────────────────────────
# The browser owns the camera. It sends frames here for stage 1 and gets boxes
# and track IDs back, which it draws over its own <video>.
#
# This replaces the old server-side ffmpeg/DirectShow capture as the primary
# path, for two reasons. It needs no ffmpeg on the host, and it lets the person
# watching pick their own camera - including a front-facing one - through the
# browser's own device picker, which the server could never do because it had
# no way to know what they had plugged in.
#
# Trackers are per session id. Two people watching at once must not share track
# IDs, or "TRACK 3" means different humans in different browsers.
_LIVE_TRACKERS: dict[str, tuple[float, Tracker]] = {}
_LIVE_LOCK = threading.Lock()
_LIVE_TTL = 300.0          # drop an idle session's tracker after this


def _live_tracker(session: str) -> Tracker:
    now = time.time()
    with _LIVE_LOCK:
        # Opportunistic sweep. Cheap because the dict only ever holds the
        # sessions currently connected.
        for sid in [s for s, (t, _) in _LIVE_TRACKERS.items() if now - t > _LIVE_TTL]:
            _LIVE_TRACKERS.pop(sid, None)
        hit = _LIVE_TRACKERS.get(session)
        if hit is None:
            tr = Tracker()
            _LIVE_TRACKERS[session] = (now, tr)
            return tr
        _LIVE_TRACKERS[session] = (now, hit[1])
        return hit[1]


@application.post("/api/live/frame")
async def live_frame(file: UploadFile = File(...), session: str = "default") -> dict:
    """
    One frame from the viewer's own camera: detect, track, hand back boxes.

    Detection and tracking only. This endpoint does not enrol anybody, does not
    touch the identity gallery, and returns no names - which is why it is safe
    to point at a camera in a room that happens to contain the person reading
    this page.
    """
    if not personfinder.available:
        raise HTTPException(503, f"stage 1 unavailable: {personfinder.detail}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty frame")
    if len(data) > faceapp.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "frame too large")

    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "undecodable frame")

    res = personfinder.detect(img)
    res["tracks"] = _live_tracker(session).update(res.get("persons") or [])
    res["track_count"] = len(res["tracks"])
    res["frame"] = {"w": int(img.shape[1]), "h": int(img.shape[0])}
    return res


@application.post("/api/live/stop")
def live_stop(session: str = "default") -> dict:
    """Forget a session's tracks so IDs restart clean next time."""
    with _LIVE_LOCK:
        _LIVE_TRACKERS.pop(session, None)
    return {"ok": True, "stopped": session}


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
<title>Gray Tech Security</title>
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

/* Hero: drone plate behind the wordmark, with a live telemetry strip. The image
   is decorative and sits behind the text, so it is an empty alt on a CSS
   background rather than a real <img> the reader has to skip. */
header.hero{position:relative;overflow:hidden;padding:0;
  border-bottom:1px solid var(--line);margin:0 -28px 22px}
.hero-plate{position:absolute;inset:0;background:
    url('/deck-assets/drone-cover.jpg') center 42%/cover no-repeat;
  filter:saturate(.8) contrast(1.06)}
.hero-plate::after{content:"";position:absolute;inset:0;background:
    linear-gradient(180deg, rgba(13,15,12,.62) 0%, rgba(13,15,12,.86) 58%, var(--ink) 100%),
    radial-gradient(120% 80% at 78% 6%, rgba(201,162,39,.16), transparent 55%)}
.heroin{position:relative;z-index:2;padding:44px 28px 26px;max-width:var(--maxw);margin:0 auto}
h1{font-family:'Saira Condensed',sans-serif;font-weight:700;font-size:clamp(30px,4.2vw,46px);
  line-height:1;text-transform:uppercase;letter-spacing:.01em;margin-top:12px;
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1 .g{color:var(--gold)}
h1 .mark{width:26px;height:26px;flex:none}
.tagline{color:var(--mute);font-weight:300;font-size:15.5px;margin-top:10px;max-width:640px}

/* live telemetry strip in the hero - the same numbers the stages below report,
   so the top of the page is alive before you scroll to anything */
.hero-strip{display:flex;flex-wrap:wrap;gap:0;margin-top:22px;
  border:1px solid var(--line-2);background:rgba(23,26,21,.72);backdrop-filter:blur(6px)}
.hero-chip{padding:11px 18px;border-right:1px solid var(--line);min-width:112px}
.hero-chip:last-child{border-right:0}
.hero-chip .lab{font-family:'Space Mono',monospace;font-size:9.5px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--gold)}
.hero-chip .val{font-family:'Saira Condensed',sans-serif;font-weight:700;font-size:24px;
  line-height:1.1;margin-top:5px;color:var(--paper);font-variant-numeric:tabular-nums}
@media(max-width:720px){
  .hero-chip{flex:1 1 33%;min-width:0;padding:10px 12px}
  .hero-chip:nth-child(3n){border-right:0}
}
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

/* ---------- the three stages, side by side ---------- */
.stages{display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin-top:22px;
  border:1px solid var(--line);background:var(--ink-3)}
@media(max-width:900px){.stages{grid-template-columns:1fr}}
.stage{padding:18px 20px;border-right:1px solid var(--line)}
.stage:last-child{border-right:0}
@media(max-width:900px){.stage{border-right:0;border-bottom:1px solid var(--line)}
  .stage:last-child{border-bottom:0}}
.stagename{font-family:'Saira Condensed',sans-serif;font-weight:600;font-size:21px;
  text-transform:uppercase;letter-spacing:.02em;margin-top:4px;color:var(--paper)}
.stagename em{font-style:normal;color:var(--gold);font-family:'Space Mono',monospace;
  font-size:10.5px;letter-spacing:.14em;margin-left:9px;vertical-align:middle}
.stagewhat{color:var(--mute);font-weight:300;font-size:13px;line-height:1.55;margin-top:14px;
  padding-top:12px;border-top:1px solid var(--line)}
.stagewhat b{color:var(--paper);font-weight:500}
.scenebar{display:flex;flex-wrap:wrap;gap:10px 26px;margin-top:14px;padding:12px 2px 0;
  border-top:1px solid var(--line)}
.scenebar .kpi{padding:0;border:0;background:none}
.howline{color:var(--mute);font-weight:300;font-size:13.5px;line-height:1.6;margin-top:14px;
  max-width:900px}
.howline b{color:var(--paper);font-weight:500}
.howline .g{color:var(--gold)}

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

/* ---------- live HUD ---------- */
.ctl{background:var(--ink-3);color:var(--paper);border:1px solid var(--line-2);
  font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.1em;padding:8px 12px}
.ctl:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
.ctlbtn{background:var(--gold);color:var(--ink);border:1px solid var(--gold)}
.ctlbtn:hover{background:var(--gold-bright);border-color:var(--gold-bright);color:var(--ink)}
.hud{position:relative;background:#050706;border:1px solid var(--line-2);
  aspect-ratio:16/9;overflow:hidden}
.hud video,.hud canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.hud video{filter:saturate(.85) contrast(1.05)}
.hud-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  text-align:center;padding:28px;color:var(--mute);font-weight:300;font-size:14px;
  line-height:1.7;background:repeating-linear-gradient(45deg,
    rgba(255,255,255,.012) 0 12px, transparent 12px 24px)}
.hud-empty b{color:var(--paper);font-weight:500}
.hud-empty[hidden]{display:none}
/* corner brackets, the way a tracker HUD frames a subject */
.hud::before,.hud::after{content:"";position:absolute;width:26px;height:26px;
  border:2px solid var(--gold);opacity:.75;pointer-events:none;z-index:3}
.hud::before{top:10px;left:10px;border-right:0;border-bottom:0}
.hud::after{bottom:10px;right:10px;border-left:0;border-top:0}
.hud-tl{position:absolute;top:12px;left:44px;z-index:4;
  font-family:'Space Mono',monospace;font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--gold);background:rgba(5,7,6,.72);
  border:1px solid var(--line);padding:4px 9px}
.hud-telemetry{position:absolute;left:12px;top:44px;z-index:4;display:flex;
  flex-direction:column;gap:1px;font-family:'Space Mono',monospace;font-size:10px;
  letter-spacing:.1em;text-transform:uppercase;pointer-events:none}
.hud-telemetry span{background:rgba(5,7,6,.72);border:1px solid var(--line);
  padding:3px 8px;color:var(--mute)}
.hud-telemetry span b{color:var(--gold);font-weight:400;margin-left:8px}
.hud-scan{position:absolute;left:0;right:0;height:2px;z-index:2;pointer-events:none;
  background:linear-gradient(90deg,transparent,rgba(201,162,39,.5),transparent);
  animation:hudscan 5.5s linear infinite}
@keyframes hudscan{0%{top:2%;opacity:0}8%{opacity:1}92%{opacity:1}100%{top:98%;opacity:0}}
@media(prefers-reduced-motion:reduce){.hud-scan{display:none}}

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

<header class="hero">
  <div class="hero-plate" role="img" aria-label="Airframe on station over a monitored perimeter at dusk"></div>
  <div class="heroin">
    <span class="eyebrow">Gray Tech Solutions &middot; Situational Awareness</span>
    <h1><svg class="mark" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><polygon points="32,7 57,32 32,57 7,32" fill="none" stroke="#C9A227" stroke-width="5"/><polygon points="32,20 44,32 32,44 20,32" fill="#C9A227"/><polygon points="32,27 37,32 32,37 27,32" fill="#0D0F0C"/></svg>Gray Tech <span class="g">Security</span>
        <span class="badge live" id="conn">connecting</span></h1>
    <p class="tagline">Person detection first, identity second &mdash; over a monitored scene,
      with a match threshold derived from observed traffic rather than a generic default.
      Point stage 1 at your own camera and watch it work.</p>
    <div class="hero-strip" id="heroStrip"></div>
  </div>
</header>

<div class="card" style="margin-top:0" id="liveCard">
  <div class="cardhead">
    <div><span class="eyebrow">Stage 1 &middot; Live</span><h2>Your camera, tracked</h2></div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <select id="camSel" class="ctl" disabled><option>no camera yet</option></select>
      <button id="camBtn" class="ctlbtn">Start camera</button>
    </div>
  </div>

  <div class="hud" id="hud">
    <video id="vid" playsinline muted autoplay></video>
    <canvas id="hudc"></canvas>
    <div class="hud-tl" id="hudId">STANDBY</div>
    <div class="hud-telemetry" id="hudTel"></div>
    <div class="hud-scan"></div>
    <div class="hud-empty" id="hudEmpty">
      <b>Camera off.</b> Nothing is being captured or sent.
      Press <b>Start camera</b> to point stage 1 at your own device &mdash; the
      browser will ask which one, including a front-facing camera. Frames go to
      this server for person detection and come straight back as boxes; no
      video is stored, nobody is enrolled, and no name is ever attached.
    </div>
  </div>
  <div class="note" id="liveNote"></div>
</div>

<!-- Three stages, in pipeline order, each labelled with what it actually
     measures. A flat row of ten numbers cannot tell you that "Persons" and
     "Identified" come from different models answering different questions. -->
<div class="stages">
  <div class="stage">
    <span class="eyebrow">Stage 1 &middot; Count</span>
    <div class="stagename">Find <em>YOLO</em></div>
    <div class="kpis" id="kpis1" style="margin-top:14px"></div>
    <p class="stagewhat"><b>Counts people.</b> Runs on the image alone &mdash; no
      gallery, no face model, no match threshold. A person here is a
      <b>body detected</b>, never a name. Works with an empty gallery.</p>
  </div>
  <div class="stage">
    <span class="eyebrow">Stage 2 &middot; Check</span>
    <div class="stagename">Gate <em>PIXEL BUDGET</em></div>
    <div class="kpis" id="kpis2" style="margin-top:14px"></div>
    <p class="stagewhat"><b>Refuses crops it cannot judge.</b> Face width and
      sharpness are measured before matching; too small or too blurred and the
      crop is dropped as <b>gated</b>, not logged as an unknown person.</p>
  </div>
  <div class="stage">
    <span class="eyebrow">Stage 3 &middot; Name</span>
    <div class="stagename">Identify <em>BUFFALO</em></div>
    <div class="kpis" id="kpis3" style="margin-top:14px"></div>
    <p class="stagewhat"><b>Names a face, and only an enrolled one.</b> A 512-d
      embedding matched against references you enrolled. No match means
      <b>unknown</b>, which is an honest miss &mdash; not a person stage 1
      counted.</p>
  </div>
</div>

<div class="scenebar kpis" id="kpis"></div>
<p class="howline" id="howline"></p>

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

<div class="row" style="margin-top:14px">
  <div class="card grow">
    <div class="cardhead">
      <div><span class="eyebrow">Stage 1 &middot; Find</span><h2>Person detection &mdash; YOLO</h2></div>
    </div>
    <div class="kpis" id="yoloKpis" style="margin-top:0"></div>
    <div class="note" id="yoloNote"></div>
  </div>
  <div class="card grow">
    <div class="cardhead">
      <div><span class="eyebrow">Stage 2 &middot; Gate</span><h2>Pixel budget</h2></div>
    </div>
    <div class="kpis" id="pxKpis" style="margin-top:0"></div>
    <div class="note" id="pxNote"></div>
  </div>
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
  <span class="eyebrow">The build &middot; Aerial Biometrics</span>
  <h2 class="wtitle">Bottom line <span class="g">up front</span></h2>
  <p class="wlede">The open-source solution used by people who care about accuracy is
    <b>YOLO (find) + optical zoom gimbal (pixels) + InsightFace SCRFD/ArcFace (identify) +
    FAISS (search) + ByteTrack (temporal)</b>, running TensorRT on a Jetson Orin NX, with a
    10&ndash;30&times; optical camera (SIYI ZR10/ZR30 or Viewpro 30&times;). Everything else is
    either a toy tracker or a ground-station batch job.</p>
  <p class="wlede"><b>Most missions do not need identity.</b> For search and rescue the useful
    system is <b>YOLO-person + thermal + optical zoom</b> &mdash; that is person detection, not
    face recognition (TEXSAR&rsquo;s ADIAT is the open-source SAR image tool). ArcFace earns its
    place only against a consented, small gallery: a missing person, authorised crew. So this
    build leads with <b>find</b>, gates on the <b>pixel budget</b>, and treats identity as the
    optional third stage.</p>

  <div class="subhead">Where this console stands against that spec</div>
  <table class="wt">
    <tr><th>Stage</th><th>Spec</th><th>This console, today</th><th>Status</th></tr>
    <tr><td>1 &middot; Find</td><td>YOLO person on a wide frame, gating everything after it</td><td>yolov8n on onnxruntime, reads the live scene; <code>POST /api/persons</code></td><td><span class="ok">live</span></td></tr>
    <tr><td>2 &middot; Gate</td><td>Quality gate before matching &mdash; blur, pose, face size</td><td>Face width, IOD and Laplacian sharpness measured per crop; refused before match</td><td><span class="ok">live</span></td></tr>
    <tr><td>3 &middot; Identify</td><td>SCRFD &rarr; align &rarr; ArcFace (buffalo_l / antelopev2)</td><td>SCRFD + <b>MobileFaceNet</b> (buffalo_s), 512-d, CPU</td><td><span class="mid">swap the head</span></td></tr>
    <tr><td>4 &middot; Search</td><td>FAISS, sub-ms at gallery scale</td><td>numpy cosine matmul over the 512-d index</td><td><span class="ok">fine to ~50k</span></td></tr>
    <tr><td>5 &middot; Temporal</td><td>ByteTrack; embed on new tracks, not every frame</td><td>one embed per crossing of the scan line</td><td><span class="mid">build</span></td></tr>
    <tr><td>6 &middot; Pixels</td><td>10&ndash;30&times; optical; &ge;80 px before ArcFace</td><td>fixed ground camera; measured px shown live above</td><td><span class="mid">needs optics</span></td></tr>
    <tr><td>7 &middot; Runtime</td><td>TensorRT on Jetson Orin NX</td><td>onnxruntime, CPUExecutionProvider</td><td><span class="mid">needs hardware</span></td></tr>
  </table>
  <p class="note">Two of the seven stages are already running on this page and you can watch
    them work. The rest is the build.</p>

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
          <span class="c-title">Build constraints</span>
          <span class="c-sub">Procurement and regulatory gates &mdash; settle these before hardware.</span>
        </span>
        <span class="c-toggle"></span>
      </summary>
      <div class="c-body">
        <p>These two decide what the build is allowed to be. Neither is a formality, and both
          are cheaper to resolve on paper than after the airframe is flying.</p>
        <table class="wt">
          <tr><th>Gate</th><th>Constraint</th><th>What it forces</th></tr>
          <tr><td>Model licence</td>
            <td>InsightFace code is MIT, but the pretrained packs (buffalo_l, antelopev2) are
              <b>non-commercial research only</b>.</td>
            <td>Licence the weights from InsightFace, retrain ArcFace on data you hold rights
              to, or buy a commercially licensed SDK. Settle which one before ordering the
              Jetson &mdash; it changes the bill either way.</td></tr>
          <tr><td>Regulation</td>
            <td>Aerial biometric ID is high-risk under the <b>EU AI Act</b>. In the US it is FAA
              ops rules plus state biometric laws (e.g. BIPA), with additional Fourth Amendment
              and policy constraints for government use.</td>
            <td>Ship as research / SAR / consented-security. Many countries restrict both
              drone overflight and covert biometrics, so consent is a design input here rather
              than paperwork.</td></tr>
        </table>
        <p>Because of the second row, this build leads with <b>person detection</b> and keeps
          identity behind a consented gallery. That is the cheaper path as well as the
          defensible one &mdash; stage 1 carries most missions on its own.</p>
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
  <span>Gray Tech Security &mdash; part of the XMRT DAO ecosystem</span>
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
         mute:pv('--mute'),grn:pv('--grn'),red:pv('--red'),amb:pv('--amb'),cyan:pv('--cyn')};

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
      cx.textAlign='center';
      // The face width in px sits next to the name because it is the number
      // that decides whether the name is worth reading.
      cx.fillText(p.gated ? (p.name+' · gated')
                          : (p.name+(p.face_px? ' · '+Math.round(p.face_px)+'px':'')),
                 x, y-34);
      if(p.gated){
        cx.font='11px "Space Mono",monospace';
        cx.fillStyle=C.amb+'D9';
        cx.fillText(p.gate_reason||'', x, y-20);
      } else if(p.confidence!=null){
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

function kpiHtml(items,small){
  return items.map(([l,v,c])=>
    '<div class="kpi"><div class="lab">'+l+'</div>'+
    '<div class="val" style="color:'+(c||'var(--txt)')+';font-size:'+(small?20:26)+'px">'+
    v+'</div></div>').join('');
}

function drawKpis(s){
  const st=s.stats, px=s.pixel||{}, sp=s.person||{}, spec=px.spec||{};
  // "In frame" is s.in_frame, not s.present.length. present.length counted
  // people still walking on from off-canvas and people already walking off, so
  // it read 7 while five stick figures were on the picture.
  const inFrame=s.in_frame ?? s.present.length;

  // Stage 1 - YOLO. Its own count, independent of everything below it.
  document.getElementById('kpis1').innerHTML = kpiHtml([
    ['People seen', sp.detections ?? 0, (sp.detections?'var(--cyn)':'')],
    ['Images checked', sp.frames ?? 0, ''],
    ['Named', 'none', 'var(--mute-2)'],
  ],true);

  // Stage 2 - the quality gate.
  document.getElementById('kpis2').innerHTML = kpiHtml([
    ['Face size', px.observed_median? Math.round(px.observed_median)+' px':'—', 'var(--gold)'],
    ['Too small to judge', st.gated||0, (st.gated?'var(--amb)':'')],
    ['Floor', (spec.detect_px??'—')+' px', ''],
  ],true);

  // Stage 3 - buffalo, against the enrolled gallery only.
  const judged=(st.identified||0)+(st.unknown||0);
  document.getElementById('kpis3').innerHTML = kpiHtml([
    ['Named', st.identified ?? 0, 'var(--grn)'],
    ['Not in gallery', st.unknown ?? 0, (st.unknown?'var(--amb)':'')],
    ['Match rate', judged? ((st.identified/judged)*100).toFixed(1)+'%':'—', ''],
  ],true);

  // Scene state - not a pipeline stage, just where people are.
  document.getElementById('kpis').innerHTML = kpiHtml([
    ['In frame', inFrame, 'var(--gold)'],
    ['Walked in', st.entered ?? 0, ''],
    ['Walked out', st.exited ?? 0, ''],
    ['Scan time', (s.scan_ms||0).toFixed(0)+' ms', ''],
    ['Uptime', Math.floor((s.uptime||0)/60)+'m '+Math.floor((s.uptime||0)%60)+'s', ''],
  ]);

  // The relationship between the three numbers, in words. Deliberately NOT
  // phrased as "YOLO found N, we named M of them": the two counters have
  // different denominators. YOLO sums bodies per image, buffalo names one face
  // per crossing of the scan line, so neither number bounds the other and a
  // subset reading would be plainly wrong (stage 3 can exceed stage 1).
  document.getElementById('howline').innerHTML =
    '<b>These three numbers do not add up, and are not meant to.</b> '+
    'Stage 1 counts <span class="g">bodies per image</span> &mdash; '+(sp.detections??0)+
    ' detections over '+(sp.frames??0)+' images &mdash; and it is unreliable on tight '+
    'headshots, which is why it is the stage you point at a real scene. '+
    'Stage 3 counts <span class="g">faces per crossing</span>: '+(st.identified??0)+
    ' named against the enrolled gallery, '+(st.unknown??0)+' not in it. '+
    'Neither number bounds the other, so a gap between them is not an error. '+
    'Nothing here identifies anyone who has not been enrolled.';

  const off=(s.tracked ?? s.present.length) - inFrame;
  document.getElementById('scaninfo').innerHTML =
    '<b>'+inFrame+'</b> in frame'+
    (off>0? ', '+off+' walking on or off':'')+
    (s.edge? ', '+s.edge+' at the edge':'')+' &middot; '+
    'model '+(paused?'<span class="mid">paused</span>':'<span class="ok">running</span>');

  // Hero strip - the same figures, so the top of the page is alive on arrival.
  const chips=[
    ['In frame', inFrame, 'var(--gold)'],
    ['People seen', sp.detections ?? 0, (sp.detections?'var(--cyn)':'')],
    ['Named', st.identified ?? 0, 'var(--grn)'],
    ['Face size', px.observed_median? Math.round(px.observed_median)+'px':'—', ''],
    ['Scan', (s.scan_ms||0).toFixed(0)+'ms', ''],
    ['Uptime', Math.floor((s.uptime||0)/60)+'m '+Math.floor((s.uptime||0)%60)+'s', ''],
  ];
  document.getElementById('heroStrip').innerHTML = chips.map(([l,v,c])=>
    '<div class="hero-chip"><div class="lab">'+l+'</div>'+
    '<div class="val" style="color:'+(c||'var(--paper)')+'">'+v+'</div></div>').join('');
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
function drawStages(s){
  // Stage 1 - YOLO person-find. Reads a real SCENE (the live camera), which is
  // the only place it is the right instrument; on the tight headshot crops the
  // demo walks past it returns almost nothing, and that is a property of the
  // input, not of the model.
  const sp=s.person||{}, lv=sp.live||{};
  const tracks=sp.tracks||[];
  const ys=[
    ['Model', sp.available? 'yolov8n':'off', sp.available?'':'var(--amb)'],
    ['Persons found', sp.detections ?? 0, (sp.detections?'var(--cyn)':'')],
    ['Frames run', sp.frames ?? 0, ''],
    ['Live tracks', (lv.ok? (sp.tracks||[]).length : '—'), (tracks.length?'var(--cyn)':'')],
    ['Latency', lv.ok? Math.round(lv.ms)+' ms' : (sp.frames? '~250 ms':'—'), ''],
  ];
  document.getElementById('yoloKpis').innerHTML = ys.map(([l,v,c])=>
    `<div class="kpi"><div class="lab">${l}</div><div class="val" style="color:${c||'var(--txt)'};font-size:20px">${v}</div></div>`
  ).join('');

  const trackList = tracks.length
    ? tracks.map(t=>`<span class="feat spec">#${t.id} &middot; ${t.hits} hit`+
        `${t.hits===1?'':'s'} &middot; ${Math.round(t.conf*100)}% &middot; `+
        `${t.box[2]-t.box[0]|0}&times;${t.box[3]-t.box[1]|0}</span>`).join('')
    : '';
  document.getElementById('yoloNote').innerHTML =
    (sp.available? '' : '<b class="mid">Stage 1 not loaded.</b> '+sp.detail+' &mdash; ')+
    `<b>Counted on its own.</b> ${sp.detections??0} person detection`+
    `${(sp.detections??0)===1?'':'s'} across ${sp.frames??0} frames, with no gallery, `+
    `no face model and no match threshold involved. `+
    (sp.hit_rate!=null? `Hit rate ${Math.round(sp.hit_rate*100)}% on this imagery. `:'')+
    `Track IDs come from IoU association over consecutive frames &mdash; a track `+
    `ID is "this box over time", never a face.`+
    (trackList? `<div class="feats" style="margin-top:10px">${trackList}</div>`
               : ` <span class="mid">Switch live camera on for continuous tracks</span>`+
               ` &mdash; the simulated walk is drawn, so there is no scene for the `+
               `detector to read; it sees the still portrait each visitor carries.`);

  // Stage 2 - the quality gate, and the pixel budget it enforces.
  const px=s.pixel||{}, spec=px.spec||{};
  const ps=[
    ['Detect floor', (spec.detect_px??'—')+' px', ''],
    ['ID floor', (spec.recognise_px??'—')+' px', ''],
    ['Robust', (spec.robust_px??'—')+' px', ''],
    ['Median face', px.observed_median? Math.round(px.observed_median)+' px':'—', 'var(--gold)'],
    ['Median IOD', px.iod_median? Math.round(px.iod_median)+' px':'—', ''],
    ['Gated', px.gated_total||0, (px.gated_total?'var(--amb)':'')],
  ];
  document.getElementById('pxKpis').innerHTML = ps.map(([l,v,c])=>
    `<div class="kpi"><div class="lab">${l}</div><div class="val" style="color:${c||'var(--txt)'};font-size:20px">${v}</div></div>`
  ).join('');
  document.getElementById('pxNote').innerHTML =
    `Face width and inter-ocular distance are measured on the crop that actually `+
    `reached the model, from the detector's own bbox and landmarks. A crop under `+
    `the <b>${spec.detect_px}px</b> detect floor or under the sharpness floor is `+
    `<b>refused before matching</b> and logged as <span class="mid">gated</span> `+
    `&mdash; not as an unknown person, because refusing to judge a face is not the `+
    `same as failing to recognise one.`;
}

es.onmessage=(m)=>{
  const d=JSON.parse(m.data);
  draw(d.state); drawKpis(d.state); drawLog(d.events);
  drawCalibration(d.state.calibration); drawStages(d.state);
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

// ── Live camera + stage 1 overlay ──────────────────────────────────────────
// The browser owns the camera, so the viewer picks their own device and the
// server never has to guess what is plugged in. Nothing here starts on its own:
// it takes a click, and it stops when you stop it.
const vid=document.getElementById('vid'), hcv=document.getElementById('hudc'),
      hx=hcv.getContext('2d'), hud=document.getElementById('hud'),
      camBtn=document.getElementById('camBtn'), camSel=document.getElementById('camSel'),
      hudEmpty=document.getElementById('hudEmpty'), hudId=document.getElementById('hudId'),
      hudTel=document.getElementById('hudTel'), liveNote=document.getElementById('liveNote');
const SESSION=(crypto.randomUUID?crypto.randomUUID():String(Date.now())+Math.random());
let camStream=null, camLoop=null, camBusy=false, camFrames=0;

function hudSize(){
  hcv.width=hud.clientWidth; hcv.height=hud.clientHeight;
}

function paintTelemetry(rows){
  hudTel.innerHTML=rows.map(([k,v])=>`<span>${k}<b>${v}</b></span>`).join('');
}

async function listCams(){
  try{
    const devs=await navigator.mediaDevices.enumerateDevices();
    const cams=devs.filter(d=>d.kind==='videoinput');
    if(!cams.length){ camSel.innerHTML='<option>no camera found</option>'; return; }
    camSel.innerHTML=cams.map((d,i)=>
      `<option value="${d.deviceId}">${d.label||('Camera '+(i+1))}</option>`).join('');
    camSel.disabled=false;
  }catch(e){ camSel.innerHTML='<option>camera list blocked</option>'; }
}

async function startCam(){
  camBtn.disabled=true; camBtn.textContent='Starting…';
  try{
    // Ask for the camera only after the click, and prefer the device the viewer
    // actually selected. A front-facing camera is just another entry here.
    const sel=camSel.value;
    const constraints={audio:false, video: sel
      ? {deviceId:{exact:sel}, width:{ideal:1280}, height:{ideal:720}}
      : {facingMode:'user', width:{ideal:1280}, height:{ideal:720}}};
    camStream=await navigator.mediaDevices.getUserMedia(constraints);
    vid.srcObject=camStream;
    await vid.play();
    hudEmpty.hidden=true;
    camBtn.textContent='Stop camera';
    await listCams();                    // labels are only readable once permitted
    hudSize();
    fetch('/api/live/stop?session='+encodeURIComponent(SESSION),{method:'POST'});
    camLoop=setInterval(pumpFrame, 220);  // ~4.5 Hz: one stage-1 pass per tick
    liveNote.innerHTML='<b class="ok">Live.</b> Sending frames to stage 1 for '+
      'person detection and tracking. Detection and boxes only &mdash; nothing '+
      'is enrolled, no frame is stored, and no name is attached to any track.';
  }catch(err){
    camBtn.disabled=false; camBtn.textContent='Start camera';
    liveNote.innerHTML='<b class="no">Camera unavailable.</b> '+
      (err&&err.name==='NotAllowedError'
        ? 'Permission denied. Allow camera access for this site, then try again.'
        : (err&&err.message? err.message : 'Could not open a camera.'))+
      ' The rest of the console keeps working without it.';
  }
}

function stopCam(){
  if(camLoop){clearInterval(camLoop);camLoop=null;}
  if(camStream){camStream.getTracks().forEach(t=>t.stop());camStream=null;}
  vid.srcObject=null;
  hx.clearRect(0,0,hcv.width,hcv.height);
  hudEmpty.hidden=false;
  hudId.textContent='STANDBY';
  camFrames=0; camBusy=false;
  camBtn.disabled=false; camBtn.textContent='Start camera';
  fetch('/api/live/stop?session='+encodeURIComponent(SESSION),{method:'POST'});
  liveNote.innerHTML='Camera stopped. Streams are released and the session '+
    'track IDs are discarded.';
}

async function pumpFrame(){
  if(camBusy||!camStream||vid.readyState<2) return;
  camBusy=true;
  try{
    const off=document.createElement('canvas');
    const W=vid.videoWidth||640, H=vid.videoHeight||480;
    off.width=Math.min(W,960); off.height=Math.round(off.width*H/W);
    off.getContext('2d').drawImage(vid,0,0,off.width,off.height);
    const blob=await new Promise(r=>off.toBlob(r,'image/jpeg',0.72));
    if(!blob) return;
    const fd=new FormData(); fd.append('file',blob,'frame.jpg');
    const r=await fetch('/api/live/frame?session='+encodeURIComponent(SESSION),
                         {method:'POST',body:fd});
    if(!r.ok) throw new Error('stage 1 returned '+r.status);
    const d=await r.json();
    camFrames++;
    drawOverlay(d);
  }catch(e){
    hudId.textContent='STAGE 1 ERROR';
    liveNote.innerHTML='<b class="no">Stage 1 error.</b> '+
      (e&&e.message?e.message:'frame not accepted')+
      ' &mdash; the overlay is paused, the camera is still yours.';
  }finally{ camBusy=false; }
}

function drawOverlay(d){
  const W=hcv.width, H=hcv.height;
  hx.clearRect(0,0,W,H);
  const fr=d.frame||{};
  // Draw boxes in normalised source coordinates so the overlay lines up with
  // the video whatever size the frame came in at.
  const sx=fr.w? W/fr.w : 1, sy=fr.h? H/fr.h : 1;
  const tracksByBox={};
  (d.tracks||[]).forEach(t=>{ tracksByBox[t.box.map(Math.round).join(',')]=t; });

  (d.persons||[]).forEach(p=>{
    const [x1,y1,x2,y2]=p.box;
    const X=x1*sx, Y=y1*sy, BW=(x2-x1)*sx, BH=(y2-y1)*sy;
    const key=[x1,y1,x2,y2].map(Math.round).join(',');
    const tr=tracksByBox[key];
    const col=tr? C.cyan : C.gold;
    hx.strokeStyle=col; hx.lineWidth=Math.max(1.5,W/620);
    hx.strokeRect(X,Y,BW,BH);
    // corner ticks, so a wide box still reads as a tracked subject
    hx.lineWidth=Math.max(2,W/460);
    const t=Math.min(BW,BH)*0.22;
    [[X,Y,1,1],[X+BW,Y,-1,1],[X,Y+BH,1,-1],[X+BW,Y+BH,-1,-1]].forEach(([px,py,dx,dy])=>{
      hx.beginPath(); hx.moveTo(px+dx*t,py); hx.lineTo(px,py); hx.lineTo(px,py+dy*t); hx.stroke();
    });
    const px=Math.round(p.conf*100)+'%';
    const lbl=(tr? ('PERSON — TRACK '+tr.id) : 'PERSON — UNTRACKED')+'  '+px+
              (tr? '  ·  '+tr.hits+' HIT'+(tr.hits===1?'':'S') : '');
    hx.font='600 '+(W/58|0)+'px "Space Mono",monospace';
    const tw=hx.measureText(lbl).width, pad=5, fsz=W/58|0;
    hx.fillStyle='rgba(5,7,6,.78)';
    hx.fillRect(X,Y-Math.max(fsz,13)-pad*2,X+tw+pad*2,Math.max(fsz,13)+pad*2);
    hx.fillStyle=col;
    hx.fillText(lbl,X+pad,Y-pad);
  });

  const n=(d.persons||[]).length;
  hudId.textContent = n
    ? 'PERSON — '+n+' · TRACKS '+(d.track_count||0)
    : 'SCANNING · NO PERSON';
  paintTelemetry([
    ['STAGE 1', d.ok?'YOLOV8N':'—'],
    ['PERSONS', n],
    ['TRACKS', d.track_count||0],
    ['CONF', '≥ '+(d.conf||0.25)],
    ['LATENCY', Math.round(d.ms||0)+' ms'],
    ['SOURCE', (fr.w||0)+'×'+(fr.h||0)],
    ['FRAMES', camFrames],
  ]);
}

camBtn.addEventListener('click',()=>{ camStream? stopCam() : startCam(); });
camSel.addEventListener('change',()=>{ if(camStream){ stopCam(); } });
window.addEventListener('resize',()=>{ if(camStream) hudSize(); });
if(navigator.mediaDevices&&navigator.mediaDevices.enumerateDevices) listCams();

async function togglePause(){
  const r=await fetch('/api/pause',{method:'POST'});
  const d=await r.json(); paused=!d.running;
}
</script></body></html>"""

app = application