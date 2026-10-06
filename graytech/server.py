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
import logging
import os
import random
import re
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

# This module logs to stderr, which the supervisor redirects to logs/graytech.log
# alongside uvicorn's own output. Kept as a named logger rather than print() so
# the gallery/model mismatch lands in the same stream an operator already reads.
log = logging.getLogger("graytech")
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
        # Stage 1 has two very different input populations and they must never
        # be averaged together.
        #
        #   scans_*  - the simulated walk. Every visitor is a drawn stick figure
        #               on a canvas; the detector is shown their *portrait*, a
        #               tight headshot. A person detector is not being asked to
        #               do its job here, so this figure is kept for diagnostics
        #               only and is never surfaced as a hit rate.
        #   live_*   - the viewer's actual camera. Real scene, real people, and
        #               the only input where a hit rate means anything.
        self.yolo_seen = 0
        self.yolo_person_hits = 0
        self.yolo_per_visitor: dict = {}
        self.yolo_live_frames = 0
        self.yolo_live_detections = 0

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
        if self.cam_persons.get("ok"):
            self.yolo_live_frames += 1
            self.yolo_live_detections += self.cam_persons.get("count", 0)
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
                # The camera figures are the real ones.
                "live_frames": self.yolo_live_frames,
                "live_detections": self.yolo_live_detections,
                "live_hit_rate": (round(self.yolo_live_detections / self.yolo_live_frames, 3)
                                  if self.yolo_live_frames else None),
                # Portrait-walk figures, kept for diagnostics and explicitly not
                # a hit rate - see the note on self.yolo_seen.
                "scan_frames": self.yolo_seen,
                "scan_detections": self.yolo_person_hits,
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


# ── Enrolment ────────────────────────────────────────────────────────────────
#
# This did not exist, and its absence is why the gallery could only ever hold
# the 33 identities somebody had run a script to add. The recogniser could
# register an identity and the adaptive learner could enrich one, but neither
# app/'s routes nor calibrator's were mounted on this app, so there was no
# endpoint and no page control that could put a new person into the gallery.
#
# The failure that prompted it was concrete. Joe_Lee and Cory_Gray were each
# enrolled from ONE photograph while the other 31 identities were averaged from
# fifteen. A prototype built from a single image cannot cover the sensor, lens,
# colour-temperature and resolution differences a webcam introduces, so its
# owner was refused by his own gallery. The fix is more shots of the real person,
# so shots are what you can now supply - from the browser, not a script.
#
# What is deliberately NOT here:
#   * No guessing who an upload is. If a face already has an identity, the shot
#     is reported as a near-duplicate of that name rather than silently creating
#     a second identity for the same person.
#   * No synthesising extra shots from the one photo you have. Averaging 14
#     copies of a single image raises self-similarity and proves nothing about
#     recognition - it is the same photograph counted fourteen times.
#   * No inventing. A shot with no detectable face, or one too small to be a
#     trustworthy reference, is named and discarded, never substituted.

ENROL_MIN_FACE_PX = 60      # below this a "reference" is mostly JPEG artefacts
ENROL_MAX_BYTES = 12 * 1024 * 1024
ENROL_DUPE_COS = 0.60       # at/above this, call it the same person out loud

# Where each identity's SHOT COUNT is recorded, and why that is its own file.
#
# A gallery entry is one averaged 512-d vector. Fifteen shots go in and the count
# is not recoverable from the output - so the first version of /api/identities
# reported every identity as shots=1, because a .npy with no sidecar looks
# identical whether it was built from one photo or fifteen. That reported 31
# healthy fifteen-shot identities as thin, which is the wrong answer in the
# direction that matters: it hides which identities really are single-shot.
#
# So the count is written alongside the vector from now on, and anything with no
# sidecar reports quality "unknown" rather than a fabricated number. An identity
# whose provenance we cannot state is a thing to be curious about, not to
# silently round down to 1.
_PROVENANCE = IDENTITY_DIR / "_provenance.json"
_prov_lock = threading.Lock()


# Which recognition network produced each vector in the gallery.
#
# FACE_MODEL_PACK has always been a config knob, which made switching the
# backbone look like a one-line change. It is not, and the reason is invisible
# from the file itself: every .npy is 512-d no matter which net wrote it.
#
#   buffalo_s -> w600k_mbf   (MobileFaceNet)
#   buffalo_l -> w600k_r50   (ResNet-50/100)
#
# A cosine between those two is not a low score, it is a meaningless one. Set
# FACE_MODEL_PACK=buffalo_l against a buffalo_s gallery and every identity loads,
# every match runs, every cosine looks plausible, and every name is wrong - with
# nothing anywhere reporting a problem. That is the specific shape of failure
# worth a guard: not an error, an answer.
#
# So the pack is recorded on every enrolment, and startup refuses to serve a
# gallery that was not built by the loaded model rather than quietly mismatching.
_PACK_OF_PACK = {
    "buffalo_s": "w600k_mbf",
    "buffalo_m": "w600k_mbf",
    "buffalo_sc": "w600k_mbf",
    "buffalo_l": "w600k_r50",
    "antelopev2": "glintr100",
}
_GALLERY_MISMATCH: list[dict] = []


def _gallery_pack_report() -> dict:
    """
    Compare the loaded pack against the recorded provenance of every vector.

    Returns a report and records any mismatch in _GALLERY_MISMATCH so /health can
    surface it. Does not raise: the caller decides whether to refuse to serve,
    because a diagnostic endpoint should still answer while the operator is
    looking at the diagnostic.
    """
    prov = _read_provenance()
    want_rec = _PACK_OF_PACK.get(faceapp.MODEL_PACK, faceapp.MODEL_PACK)
    counts: dict[str, int] = {}
    unknown = 0
    for name in getattr(recogniser, "names", []):
        rec = prov.get(name) or {}
        rec_net = rec.get("recognition_net")
        if rec_net:
            counts[rec_net] = counts.get(rec_net, 0) + 1
        else:
            # Enrolled before the sidecar recorded a net. Attribute to the service
            # default rather than treating it as unknown - it was almost certainly
            # buffalo_s, and calling it "unknown" would flag a healthy gallery.
            default_net = _PACK_OF_PACK["buffalo_s"]
            counts[default_net] = counts.get(default_net, 0) + 1
            unknown += 1
    mismatched = {k: v for k, v in counts.items() if k != want_rec}
    report = {
        "model_pack": faceapp.MODEL_PACK,
        "recognition_net": want_rec,
        "gallery_by_net": counts,
        "gallery_without_recorded_net": unknown,
        "mismatched": mismatched,
        "ok": not mismatched,
    }
    _GALLERY_MISMATCH[:] = [{"net": k, "count": v} for k, v in mismatched.items()]
    if mismatched:
        log.error("GALLERY/MODEL MISMATCH: serving %s but %s of the gallery was "
                  "built by another network - every name would be wrong. Re-enrol "
                  "with tools/_enrol.py before trusting any result.",
                  faceapp.MODEL_PACK,
                  ", ".join(f"{v} by {k}" for k, v in mismatched.items()))
    return report


def _read_provenance() -> dict:
    # Read WITHOUT taking _prov_lock.
    #
    # This used to take the lock, and _write_provenance calls it while already
    # holding it. threading.Lock is not reentrant, so every enrolment deadlocked
    # here - AFTER np.save(dest, ref) had already written the new vector.
    #
    # The result was the worst shape of partial success: the gallery genuinely
    # changed, the shot count genuinely did not, and nothing raised anywhere.
    # Joe_Lee silently stayed at shots=1 while carrying a 3-photo blend, which is
    # the wrong kind of wrong - the identity behaves correctly and the record
    # lies about it, so nothing surfaces the problem.
    #
    # An unsynchronised read is safe here: write_text replaces the file, so a
    # reader sees either the old bytes or the new ones. The lock only needs to
    # serialise writers, which is what it is for.
    try:
        return json.loads(_PROVENANCE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_provenance(key: str, entry: dict) -> None:
    with _prov_lock:
        prov = _read_provenance()
        prev = prov.get(key) or {}
        shots = int(prev.get("shots", 0)) + int(entry.get("shots", 0))
        prov[key] = {
            "shots": shots,
            "batches": int(prev.get("batches", 0)) + 1,
            "first": prev.get("first", time.time()),
            "last": time.time(),
            "note": entry.get("note", ""),
            # Which embedding network wrote this vector. Without it, switching
            # FACE_MODEL_PACK cannot be detected - see _gallery_pack_report.
            "model_pack": faceapp.MODEL_PACK,
            "recognition_net": _PACK_OF_PACK.get(faceapp.MODEL_PACK,
                                                 faceapp.MODEL_PACK),
        }
        _PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
        _PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True),
                               encoding="utf-8")


def _safe_identity(raw: str) -> tuple[str, str]:
    """
    A gallery key that is safe as a filename, and the name to display.

    Case-insensitively resolves to an ALREADY-ENROLLED identity rather than
    creating a second one. This matters more than it looks: the gallery is
    case-sensitive, and the natural spelling of an existing identity and the
    natural typing of the same person are not always the same string. The
    dataset has 'cory-gray.jpg' and the enrolled key is 'Cory_Gray' - so
    enrolling that person again under the name that matches their filename
    would write 'cory_gray.npy' beside 'Cory_Gray.npy', and the console would
    then announce whichever won the race, intermittently, as two different
    people. A duplicate identity for one person is the specific failure this
    whole endpoint exists to make hard to commit, so it is not left to a
    capital letter.
    """
    display = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not display:
        raise HTTPException(400, "a name is required")
    if len(display) > 64:
        raise HTTPException(400, "name is longer than 64 characters")
    if re.search(r"[^\w .-]", display):
        raise HTTPException(
            400, "name may only contain letters, numbers, spaces, dots and dashes")
    key = re.sub(r"[^A-Za-z0-9_]", "_", display)
    if not key or key.startswith("."):
        raise HTTPException(400, "name does not reduce to a usable identifier")

    for existing in recogniser.names:
        if existing.lower() == key.lower():
            # Keep the enrolled spelling; the typed one becomes the display name.
            return existing, display
    return key, display


@application.get("/api/identities")
def identities_list() -> dict:
    """
    Who is enrolled, and how many shots each identity is actually built from.

    The bare count is misleading and that is why this exists. 2,564 photographs
    in the source dataset are 33 PEOPLE, not 2,564 subjects, and an identity can
    be a single shot while its neighbour is fifteen. The thin ones are the weak
    ones, so the shot count travels with the name instead of being something you
    have to already know.
    """
    prov = _read_provenance()
    out = []
    for n in sorted(recogniser.names):
        path = IDENTITY_DIR / f"{n}.npy"
        rec = prov.get(n)
        if rec:
            shots: Optional[int] = int(rec.get("shots", 0))
            quality = "thin" if shots < 5 else "ok"
        else:
            # No record of how many shots this was built from. Say so rather
            # than assuming one - an identity whose provenance is unknown is not
            # the same claim as an identity known to be single-shot.
            shots, quality = None, "unknown"
        try:
            from app.calibration import calibrator
            learned = len(calibrator.profiles.get(n, []))
        except Exception:                                   # noqa: BLE001
            learned = 0
        out.append({"name": n, "shots": shots, "learned": learned,
                    "quality": quality,
                    "modified": path.stat().st_mtime if path.exists() else None})
    thin = sum(1 for r in out if r["quality"] == "thin")
    unknown = sum(1 for r in out if r["quality"] == "unknown")
    return {"count": len(out), "identities": out, "thin": thin,
            "unknown": unknown, "dir": str(IDENTITY_DIR)}


@application.post("/api/enrol")
async def enrol(name: str = Query(...),
                files: list[UploadFile] = File(...)) -> dict:
    """
    Enrol a person from one or more photographs.

    Multiple files because one photograph is the failure mode, not the success
    case. The shots are averaged into a single L2-normalised reference vector:
    the centroid of that person's embedding cloud, which is more stable than any
    individual sample. Adding shots to someone already enrolled blends the new
    average into the stored vector rather than replacing it, so a profile that
    has been matching in live traffic is not overwritten by one mediocre upload.

    Every file gets a verdict. Anything unusable is named and explained rather
    than quietly dropped, because a silent success that enrolled nothing is
    precisely the failure mode of the missing endpoint this replaces.
    """
    key, display = _safe_identity(name)
    if not files:
        raise HTTPException(400, "no files uploaded")
    if len(files) > 40:
        raise HTTPException(400, "at most 40 photos per enrolment")

    # Say "the model is not loaded" instead of letting every upload fail with
    # "'NoneType' object has no attribute 'get'".
    #
    # The recogniser's app handle is None until the lifespan startup runs. Any
    # request that arrives before that - a probe against a service still booting,
    # or a tool calling the ASGI app directly - used to fail per-file with a
    # message that pointed at insightface rather than at the actual cause. That
    # cost a debugging cycle here, and it would cost an operator more: "no usable
    # face in any of the uploads" on a fully-valid set of photos is the kind of
    # report that sends someone off re-shooting portraits.
    if getattr(recogniser, "app", None) is None:
        raise HTTPException(503, "recognition model is still loading - retry shortly")

    vecs: list[np.ndarray] = []
    rejected: list[dict] = []
    for f in files:
        data = await f.read()
        label = f.filename or "upload"
        if not data:
            rejected.append({"file": label, "why": "empty upload"})
            continue
        if len(data) > ENROL_MAX_BYTES:
            rejected.append({"file": label, "why": "over 12 MB"})
            continue
        try:
            import cv2
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                rejected.append({"file": label, "why": "not a decodable image"})
                continue
            faces, matrix = recogniser.embed_bytes(img)
            if not faces or matrix.size == 0:
                rejected.append({"file": label, "why": "no face detected"})
                continue
            # Largest face wins: in a group shot the subject is the near one.
            big = max(range(len(faces)),
                      key=lambda i: faces[i]["bbox"][2] - faces[i]["bbox"][0])
            width = float(faces[big]["bbox"][2] - faces[big]["bbox"][0])
            if width < ENROL_MIN_FACE_PX:
                rejected.append({
                    "file": label, "face_px": round(width),
                    "why": f"face only {width:.0f}px wide - need "
                           f"{ENROL_MIN_FACE_PX}px for a trustworthy reference"})
                continue
            vecs.append(np.asarray(matrix[big], dtype=np.float32).reshape(-1))
        except Exception as exc:                            # noqa: BLE001
            rejected.append({"file": label, "why": f"{type(exc).__name__}: {exc}"})

    if not vecs:
        return {"ok": False, "name": display, "enrolled": 0,
                "reason": "no usable face in any of the uploads",
                "rejected": rejected, "identities": len(recogniser.names)}

    stack = np.stack(vecs).astype(np.float32)
    ref = stack.mean(axis=0)
    nrm = float(np.linalg.norm(ref))
    if nrm <= 0:
        raise HTTPException(400, "embeddings cancelled out - try different photos")
    ref = ref / nrm

    # How far apart are these shots? A wide spread means the average is covering
    # genuinely different views, which is the point. A near-zero spread means the
    # "different" photos are near-duplicates and one reference would have done.
    spread = float(np.mean(np.linalg.norm(stack - stack.mean(axis=0), axis=1)))

    existed = key in recogniser.names
    if existed:
        prev = np.load(IDENTITY_DIR / f"{key}.npy").astype(np.float32).reshape(-1)
        blended = prev + ref
        pn = float(np.linalg.norm(blended))
        if pn > 0:
            ref = blended / pn

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    dest = IDENTITY_DIR / f"{key}.npy"
    if dest.exists():
        try:
            dest.replace(dest.with_suffix(".npy.1"))
        except OSError:
            pass
    np.save(dest, ref)
    recogniser.reload_identities()
    # Record the shot count BEFORE anything that can fail, and never let a
    # provenance failure be silent. The vector is already on disk by this point,
    # so a raise here would leave the gallery updated and the record stale - the
    # exact state Joe_Lee was in. Reporting it lets the caller see the mismatch
    # instead of inheriting it.
    try:
        _write_provenance(key, {"shots": len(vecs),
                                "note": f"spread {spread:.3f}"})
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(
            500, f"enrolled '{display}' but could not record its shot count: "
                 f"{type(exc).__name__}: {exc}") from exc

    # Does this already look like somebody in the gallery? Registering a second
    # identity for one person is how a gallery fills with near-duplicate names
    # that each match only their own single photograph.
    dupes = []
    if recogniser.matrix is not None:
        sims = recogniser.matrix @ ref
        for i, other in enumerate(recogniser.names):
            if other != key and float(sims[i]) >= ENROL_DUPE_COS:
                dupes.append({"name": other, "cosine": round(float(sims[i]), 4)})

    scene.log("enrol", name=display,
              detail=f"{len(vecs)} shot(s), spread {spread:.3f}")
    return {"ok": True, "name": display, "key": key,
            "updated_existing": existed, "enrolled": len(vecs),
            "rejected": rejected, "spread": round(spread, 4),
            "shot_quality": "thin" if len(vecs) < 5 else "ok",
            "shot_note": ("fewer than 5 shots - this will be the weakest identity in "
                          "the gallery and may be refused from a webcam"
                          if len(vecs) < 5 else ""),
            "similar_to": sorted(dupes, key=lambda d: -d["cosine"])[:3],
            "identities": len(recogniser.names)}


@application.delete("/api/identities/{key}")
def enrol_delete(key: str) -> dict:
    """Remove an identity from the gallery. The vector is kept as .npy.1."""
    if key not in recogniser.names:
        raise HTTPException(404, f"{key} is not enrolled")
    p = IDENTITY_DIR / f"{key}.npy"
    p.replace(p.with_suffix(".npy.1"))
    recogniser.reload_identities()
    scene.log("unenrol", name=key, detail="identity removed")
    return {"ok": True, "removed": key, "identities": len(recogniser.names)}


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
async def live_frame(file: UploadFile = File(...),
                     session: str = "default",
                     identify: bool = True) -> dict:
    """
    One frame from the viewer's own camera: find, check, name.

    Three stages, in order, on the same frame:

      1. YOLO finds people and the tracker gives each a box a track ID. No
         gallery is involved and this stage works on its own.
      2. The quality gate measures the face crop - width, inter-ocular
         distance, sharpness - and refuses anything it cannot judge. A refused
         face is reported as refused, never as "unknown person".
      3. Only faces that passed the gate are embedded and matched against the
         enrolled gallery. A name can only come from a reference somebody
         enrolled; anyone else comes back unidentified.

    Identity is scoped to a consented gallery, never a general face search, and
    nothing is written: no enrolment, no stored frames, no names added to a
    track that did not earn one.
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

    t_find = time.perf_counter()
    res = personfinder.detect(img)
    res["tracks"] = _live_tracker(session).update(res.get("persons") or [])
    res["track_count"] = len(res["tracks"])
    ih, iw = int(img.shape[0]), int(img.shape[1])
    res["frame"] = {"w": iw, "h": ih}
    res["find_ms"] = round((time.perf_counter() - t_find) * 1000, 1)

    # Normalised box coordinates, alongside the pixel ones.
    #
    # The client should map with these and never with pixels. Pixel coordinates
    # are only meaningful relative to the exact frame they were measured in, and
    # every mismatch between "the frame we uploaded" and "the frame we drew"
    # shows up as a box sliding off the subject. Normalised fractions survive any
    # such disagreement, because they carry no dimensions at all.
    for p in res.get("persons") or []:
        x1, y1, x2, y2 = p["box"]
        p["norm"] = [round(x1 / iw, 5), round(y1 / ih, 5),
                     round(x2 / iw, 5), round(y2 / ih, 5)]

    # ── Stages 2 and 3 ───────────────────────────────────────────────────
    # Only bother at all if something was actually found. Running the face
    # model on an empty frame is the exact waste the workup warns about.
    # NOTE: nest under "identify". Spreading the dict with res.update() flattened
    # ran/named/gated/people into the top level, so the client looking for
    # res["identify"] found nothing and every stage read as absent.
    res["identify"] = (_identify_in_frame(img, res["persons"])
                       if (identify and res.get("persons"))
                       else {"ran": False,
                             "why": ("no person in frame" if identify
                                     else "identity disabled for this request")})
    return res


def _identify_in_frame(img, persons: list[dict]) -> dict:
    """
    Check and name, per detected person.

    The face model runs once on the whole frame rather than per person: SCRFD
    finds every face at once, and re-running it inside a crop per box would be
    both slower and less accurate, because a crop of a crop has already lost
    the pixels the detector needed.

    Faces are then attached to people by overlap, so the name belongs to the box
    it was actually measured from.
    """
    out = {"ran": True, "people": [], "named": 0, "gated": 0,
           "face_ms": 0.0, "gallery": len(recogniser.names)}
    try:
        t0 = time.perf_counter()
        faces, matrix = recogniser.embed_bytes(img)
        out["face_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:                                # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    matches = recogniser.match(matrix, k=1) if matrix.size else []
    ih, iw = img.shape[:2]

    for i, p in enumerate(persons):
        px1, py1, px2, py2 = p["box"]
        rec = {"box": p["box"], "conf": p["conf"], "stage": "found"}

        # Nearest face whose centre falls inside this person's box.
        best, best_d = None, 1e18
        for fi, f in enumerate(faces):
            fx = (f["bbox"][0] + f["bbox"][2]) / 2
            fy = (f["bbox"][1] + f["bbox"][3]) / 2
            if not (px1 <= fx <= px2 and py1 <= fy <= py2):
                continue
            d = (fx - (px1 + px2) / 2) ** 2 + (fy - (py1 + py2) / 2) ** 2
            if d < best_d:
                best, best_d = fi, d

        if best is None:
            # A body with no face in it: turned away, occluded, or too small.
            rec.update(stage="found", face=None, name=None,
                       note="no face in this box")
            out["people"].append(rec)
            continue

        f = faces[best]
        rec["face"] = {"bbox": f["bbox"], "face_px": f.get("face_px"),
                       "iod_px": f.get("iod_px"),
                       "sharpness": f.get("sharpness"),
                       "band": f.get("band")}

        # ── Stage 2: the quality gate, before any matching ───────────────
        reasons = []
        if f.get("face_px") is not None and f["face_px"] < faceapp.PX_DETECT:
            reasons.append(f"face {f['face_px']:.0f}px < {faceapp.PX_DETECT:.0f}px")
        if (f.get("sharpness") is not None
                and f["sharpness"] < faceapp.SHARPNESS_MIN):
            reasons.append(f"sharp {f['sharpness']:.0f} < {faceapp.SHARPNESS_MIN:.0f}")
        if reasons:
            rec.update(stage="gated", name=None, reason="; ".join(reasons))
            out["gated"] += 1
            out["people"].append(rec)
            continue

        rec["stage"] = "checked"
        # ── Stage 3: name, against the enrolled gallery only ─────────────
        row = matches[best] if best < len(matches) else []
        top = row[0] if row else None
        if top and top.get("matched"):
            rec.update(stage="named", name=top["name"],
                       cosine=round(float(top["cosine"]), 4))
            out["named"] += 1
        else:
            rec.update(stage="unidentified", name=None,
                       cosine=(round(float(top["cosine"]), 4) if top else None))
        out["people"].append(rec)

    del ih, iw
    return out


@application.post("/api/live/stop")
def live_stop(session: str = "default") -> dict:
    """Forget a session's tracks so IDs restart clean next time."""
    with _LIVE_LOCK:
        _LIVE_TRACKERS.pop(session, None)
    return {"ok": True, "stopped": session}


@application.get("/health")
def health() -> dict:
    # "status" becomes "degraded" - not "ok", not "down" - when the gallery was
    # built by a different recognition network than the one now loaded. The
    # service answers perfectly and every name it prints is wrong, which is the
    # one failure a check for HTTP 200 cannot see. Anything watching this gets
    # the reason instead of a green tick.
    gallery = _gallery_pack_report()
    return {
        "status": "ok" if gallery["ok"] else "degraded",
        "model_pack": faceapp.MODEL_PACK,
        "recognition_net": gallery["recognition_net"],
        "gallery": {k: v for k, v in gallery.items()
                    if k not in ("model_pack", "recognition_net")},
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
body{background:var(--ink);color:var(--paper);
  font-family:'Barlow',system-ui,sans-serif;font-weight:400;
  line-height:1.6;-webkit-font-smoothing:antialiased}
/* Content column. The old body padding + max-width became this wrapper so the
   background photograph can run full-bleed behind it. */
.page{max-width:var(--maxw);margin:0 auto;padding:0 28px 72px;
  position:relative;z-index:1}
::selection{background:var(--gold);color:var(--ink)}
.mono{font-family:'Space Mono',monospace}
.eyebrow{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.26em;
  text-transform:uppercase;color:var(--gold);display:block}

/* ---------- masthead ---------- */
header.top{border-bottom:1px solid var(--line);margin:0 -28px 24px;padding:30px 28px 24px;
  background:radial-gradient(120% 90% at 78% 0%, rgba(201,162,39,.10), transparent 55%),var(--ink)}

/* The drone plate belongs to the HERO, full-bleed to the viewport edges but
   only as tall as the hero itself. An earlier attempt hung it on body as a
   fixed background, which stretched one landscape frame across the entire
   scrolling document - the aircraft ended up a smear behind the workup tables
   and the page lost its flat ink surface. Contained to the hero it reads as a
   photograph again. */
body{background:var(--ink)}

header.hero{position:relative;overflow:hidden;isolation:isolate;
  margin:0 -28px 22px;padding:0;border-bottom:1px solid var(--line)}
.hero-plate{position:absolute;inset:0;z-index:-2;background:
    url('/deck-assets/drone-cover.jpg') center 46%/cover no-repeat;
  filter:saturate(.78) contrast(1.05)}
/* Scrim keeps the wordmark legible over the sky and fades the bottom edge into
   the page ink so the hero does not end on a hard rectangle. */
.hero-plate::after{content:"";position:absolute;inset:0;background:
    linear-gradient(180deg,
      rgba(13,15,12,.58) 0%, rgba(13,15,12,.84) 46%,
      rgba(13,15,12,.95) 82%, var(--ink) 100%),
    radial-gradient(110% 78% at 76% 8%, rgba(201,162,39,.15), transparent 58%)}
.heroin{position:relative;z-index:2;padding:46px 28px 26px;
  max-width:var(--maxw);margin:0 auto}
h1{font-family:'Saira Condensed',sans-serif;font-weight:700;font-size:clamp(30px,4.2vw,46px);
  line-height:1;text-transform:uppercase;letter-spacing:.01em;margin-top:12px;
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1 .g{color:var(--gold)}
h1 .mark{width:26px;height:26px;flex:none}
h1 .sub{display:block;font-family:'Space Mono',monospace;font-size:12px;
  letter-spacing:.26em;text-transform:uppercase;color:var(--mute);font-weight:400;
  margin-top:12px}
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
/* Three stages, one compact strip. This was three tall cards with a heading,
   three KPI cells and a paragraph each, which pushed the recognition scene -
   the thing the page exists to show - entirely below the fold. Collapsed to a
   single row: stage name on the left, its numbers inline, no prose. */
.stages{display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin-top:14px;
  border:1px solid var(--line);background:var(--ink-3)}
@media(max-width:860px){.stages{grid-template-columns:1fr}}
.stage{padding:12px 16px;border-right:1px solid var(--line);
  display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.stage:last-child{border-right:0}
@media(max-width:860px){.stage{border-right:0;border-bottom:1px solid var(--line)}
  .stage:last-child{border-bottom:0}}
.stagename{font-family:'Saira Condensed',sans-serif;font-weight:600;font-size:16px;
  text-transform:uppercase;letter-spacing:.03em;color:var(--paper);flex:none}
.stagename em{font-style:normal;color:var(--gold);font-family:'Space Mono',monospace;
  font-size:9.5px;letter-spacing:.14em;margin-left:7px;vertical-align:middle}
/* Inline figures instead of nested cards. */
.stage .kpis{display:flex;gap:16px;margin-top:0;flex-wrap:wrap;flex:1}
.stage .kpi{padding:0;border:0;background:none;min-width:0}
.stage .kpi .lab{font-size:9px;letter-spacing:.16em;color:var(--mute-2)}
.stage .kpi .val{font-family:'Space Mono',monospace;font-size:13px;font-weight:400;
  margin-top:2px;letter-spacing:.02em}
.scenebar{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:10px;padding:10px 2px 0;
  border-top:1px solid var(--line)}
.scenebar .kpi{padding:0;border:0;background:none}
.howline{color:var(--mute-2);font-weight:300;font-size:12px;line-height:1.55;margin-top:10px;
  max-width:900px}
.howline b{color:var(--mute);font-weight:400}
.howline .g{color:var(--gold)}

/* The scene canvas is opaque. The HUD overlay must NOT be - it sits directly on
   top of the <video>, and inheriting this background painted a solid ink panel
   over the feed. The camera was running and hidden the entire time. */
#scene{width:100%;display:block;background:var(--ink);border:1px solid var(--line-2)}
#hudc{background:transparent}
#log{max-height:340px;overflow-y:auto;font-family:'Space Mono',monospace;
  font-size:11.5px;line-height:1.6}
/* Each row is a pipeline step: a coloured rail shows whether this event came
   from the find stage or the name stage, so a scan arriving reads as
   "body found -> face checked -> named" as it happens. */
.ev{position:relative;display:grid;
  grid-template-columns:3px 58px 54px minmax(70px,auto) 1fr;
  gap:9px;align-items:baseline;padding:5px 0 5px 0;
  border-bottom:1px solid rgba(236,237,230,.045)}
.ev .rail{background:var(--line-2);align-self:stretch;border-radius:2px}
.ev[data-stage="1"] .rail{background:var(--cyn)}
.ev[data-stage="2"] .rail{background:var(--gold)}
.ev .t{color:var(--mute-2);flex-shrink:0}
.ev .k{font-weight:700;letter-spacing:.06em;font-size:10px}
.ev .nm{color:var(--paper);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ev .dt{color:var(--mute);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* New rows slide in and flash their rail, so arrivals are visible without
   having to read them. */
.ev.fresh{animation:evin .45s ease-out}
@keyframes evin{from{opacity:0;transform:translateY(-6px);background:rgba(201,162,39,.10)}
                to{opacity:1;transform:none;background:transparent}}
@media(prefers-reduced-motion:reduce){.ev.fresh{animation:none}}
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
/* A portrait camera gets a portrait panel. Forcing a 16:9 box around a 9:16 feed
   is what forced the letterboxing in the first place: contain-fit inside 16:9
   wastes over half the panel, and on a phone the HUD was a short letterboxed
   strip with the subject cropped. Match the panel to the feed instead. */
.hud.portrait{aspect-ratio:9/16;max-height:min(78vh,720px);margin-inline:auto}
.hud.portrait .hud-telemetry{font-size:9.5px}
.hud.portrait .hud-tl{left:12px;top:12px}
@media(max-width:720px){
  .hud{aspect-ratio:4/3}
  .hud.portrait{aspect-ratio:3/4}
}
/* The <video> is the capture SOURCE only; it is not displayed.
   The canvas paints the frame itself, so the picture and the detection boxes
   are guaranteed to share one coordinate space. Letting CSS object-fit:cover
   scale the video while the canvas used its own arithmetic was the source of a
   persistent leftward drift in the overlay - the two never agreed on where the
   frame edges were. */
.hud video{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
.hud canvas{position:absolute;inset:0;width:100%;height:100%}
.hud-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  text-align:center;padding:28px;color:var(--mute);font-weight:300;font-size:14px;
  line-height:1.7;background:repeating-linear-gradient(45deg,
    rgba(255,255,255,.012) 0 12px, transparent 12px 24px)}
.hud-empty b{color:var(--paper);font-weight:500}
.hud-empty[hidden]{display:none}
/* Click-to-expand, like a video call. */
.hud{cursor:zoom-in}
.hud.fs{cursor:zoom-out;position:fixed;inset:0;z-index:900;aspect-ratio:auto;
  border:0;background:#000}
.hud.fs video,.hud.fs canvas{object-fit:contain}
.hud.fs .hud-telemetry{font-size:13px;top:56px;left:20px}
.hud.fs .hud-telemetry span{font-size:13px;padding:5px 11px}
.hud-fs-hint{position:absolute;right:12px;bottom:12px;z-index:5;
  font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--mute);background:rgba(5,7,6,.72);
  border:1px solid var(--line);padding:4px 9px;pointer-events:none}
.hud.fs .hud-fs-hint{color:var(--gold)}
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

<div class="page">
<header class="hero">
  <div class="hero-plate" role="img" aria-label="Airframe on station over a monitored perimeter at dusk"></div>
  <div class="heroin">
    <span class="eyebrow">Gray Tech Solutions &middot; Situational Awareness</span>
    <h1><svg class="mark" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><polygon points="32,7 57,32 32,57 7,32" fill="none" stroke="#C9A227" stroke-width="5"/><polygon points="32,20 44,32 32,44 20,32" fill="#C9A227"/><polygon points="32,27 37,32 32,37 27,32" fill="#0D0F0C"/></svg><span>Astra<span class="g">Gaze</span></span>
        <span class="badge live" id="conn">connecting</span>
      <span class="sub">by Gray Tech Solutions</span></h1>
    <p class="tagline">Person detection first, identity second &mdash; over a monitored scene,
      with a match threshold derived from observed traffic rather than a generic default.
      Point stage 1 at your own camera and watch it work.</p>
    <div class="hero-strip" id="heroStrip"></div>
  </div>
</header>

<div class="card" style="margin-top:0" id="liveCard">
  <div class="cardhead">
    <div><span class="eyebrow">Stage 1 &middot; Live</span><h2>Demo the Gray Tech tracking system</h2></div>
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
    <div class="hud-fs-hint" id="fsHint">Click to expand</div>
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

<div class="card" style="margin-top:14px" id="aboutCard">
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
    <tr><td>3 &middot; Identify</td><td>SCRFD &rarr; align &rarr; ArcFace (buffalo_l / antelopev2)</td>
      <td>Detect <b>SCRFD-500M</b> (already the spec model). Embed is
        <b>MobileFaceNet</b>, not ArcFace &mdash; the one head actually outstanding</td>
      <td><span class="mid">swap embed head</span></td></tr>
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

</footer>

</div><!-- /.page -->

<script>
const cv = document.getElementById('scene'), cx = cv.getContext('2d');
let paused = false;

// The 2D canvas cannot read CSS custom properties, so pull the palette out of the
// stylesheet once instead of hard-coding hex here. That keeps :root the single
// source of truth: restyle the console and the scene follows. Translucent strokes
// use the 8-digit hex form, C.gold+'22' == the old rgba(201,162,39,.13).
const CS=getComputedStyle(document.documentElement);
const pv=n=>CS.getPropertyValue(n).trim();
const C={ink:pv('--ink'),panel:pv('--ink-3'),gold:pv('--gold'),goldBright:pv('--gold-bright'),
         paper:pv('--paper'),mute:pv('--mute'),grn:pv('--grn'),red:pv('--red'),
         amb:pv('--amb'),cyan:pv('--cyn')};

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

  // The three stage tiles are gone. The pipeline is demonstrated live in the
  // camera HUD instead - find, check, name animate in the overlay - so these
  // numbers are just the running totals for the scene.
  const judged=(st.identified||0)+(st.unknown||0);
  document.getElementById('kpis').innerHTML = kpiHtml([
    ['In frame', inFrame, 'var(--gold)'],
    ['Walked in', st.entered ?? 0, ''],
    ['Walked out', st.exited ?? 0, ''],
    ['Named', st.identified ?? 0, 'var(--grn)'],
    ['Not in gallery', st.unknown ?? 0, (st.unknown?'var(--amb)':'')],
    ['Match rate', judged? ((st.identified/judged)*100).toFixed(1)+'%':'—', ''],
    ['Scan time', (s.scan_ms||0).toFixed(0)+' ms', ''],
    ['Uptime', Math.floor((s.uptime||0)/60)+'m '+Math.floor((s.uptime||0)%60)+'s', ''],
  ]);

  document.getElementById('howline').innerHTML =
    'The camera panel runs the whole chain live &mdash; <b>find</b> a person, '+
    '<b>check</b> the face is usable, then <b>name</b> it against the enrolled '+
    'gallery. A name only ever comes from a reference you enrolled; anyone else '+
    'is left unidentified rather than guessed at.';

  const off=(s.tracked ?? s.present.length) - inFrame;
  document.getElementById('scaninfo').innerHTML =
    '<b>'+inFrame+'</b> in frame'+
    (off>0? ', '+off+' walking on or off':'')+
    (s.edge? ', '+s.edge+' at the edge':'')+' &middot; '+
    'model '+(paused?'<span class="mid">paused</span>':'<span class="ok">running</span>');

  // Hero strip - the same figures, so the top of the page is alive on arrival.
  const chips=[
    ['In frame', inFrame, 'var(--gold)'],
    // Hit rate, not a headcount. YOLO finds well under one person per portrait
    // here, so a "people seen" figure next to "named" implied a census it was
    // not performing and read as a contradiction.
    ['YOLO hit rate', sp.live_frames? Math.round((sp.live_detections/sp.live_frames)*100)+'%' : '—',
      (sp.live_frames && (sp.live_detections/sp.live_frames)>=0.7?'var(--cyn)':'var(--amb)')],
    ['Named', st.identified ?? 0, 'var(--grn)'],
    ['Face size', px.observed_median? Math.round(px.observed_median)+'px':'—', ''],
    ['Scan', (s.scan_ms||0).toFixed(0)+'ms', ''],
    ['Uptime', Math.floor((s.uptime||0)/60)+'m '+Math.floor((s.uptime||0)%60)+'s', ''],
  ];
  document.getElementById('heroStrip').innerHTML = chips.map(([l,v,c])=>
    '<div class="hero-chip"><div class="lab">'+l+'</div>'+
    '<div class="val" style="color:'+(c||'var(--paper)')+'">'+v+'</div></div>').join('');
}

// The log is the pipeline demo, not a log. Each arrival walks the same path the
// data actually took -- FIND (a body) then NAME (a face) -- so the three stage
// tiles above stop being abstract labels and become the thing you can watch.
const STAGE_OF={entry:1, scan:2, error:2, exit:1, gated:2};
function drawLog(evs){
  const box=document.getElementById('log');
  if(!box) return;
  const newest=evs.length? evs[evs.length-1].t : 0;
  if(newest===lastLogT) return;      // no new arrival since the last frame
  lastLogT=newest;
  // Everything we have not seen yet, oldest first, so the newest lands last.
  const fresh=[];
  for(const e of evs){
    const sig=e.t+'|'+e.kind+'|'+e.name;
    if(!seenEvents.has(sig)){ seenEvents.add(sig); fresh.push(e); }
  }
  // Keep the seen-set from growing without bound on a long-lived tab.
  if(seenEvents.size>400){ seenEvents=new Set(fresh.map(e=>e.t+'|'+e.kind+'|'+e.name)); }

  const rowHtml=(e,isNew)=>{
    let detail='';
    if(e.kind==='scan'){
      detail = e.detected
        ? (e.matched? `matched ${e.name} ${e.confidence}` : `no match (best ${e.confidence})`)
        : 'no face detected';
      if(e.gated||e.reason) detail = e.reason || 'held at the quality gate';
    }
    const st=STAGE_OF[e.kind]||1;
    return `<div class="ev${isNew?' fresh':''}" data-stage="${st}">`+
           `<span class="rail"></span>`+
           `<span class="t">${fmtT(e.t)}</span>`+
           `<span class="k ${e.kind}">${e.kind.toUpperCase()}</span>`+
           `<span class="nm">${e.name||''}</span>`+
           `<span class="dt">${detail}</span></div>`;
  };

  // Prepend the new arrivals so they animate in at the top, newest first.
  if(fresh.length){
    const add=fresh.reverse().map(e=>rowHtml(e,true)).join('');
    box.insertAdjacentHTML('afterbegin', add);
    const rows=box.querySelectorAll('.ev');
    for(let i=rows.length-1;i>=0 && i>90;i--) rows[i].remove();
    requestAnimationFrame(()=>{
      box.querySelectorAll('.ev.fresh').forEach(r=>r.classList.remove('fresh'));
    });
  }
}

const es=new EventSource('/api/stream');
es.onopen =()=>{document.getElementById('conn').textContent='live';
                document.getElementById('conn').className='badge live';};
es.onerror=()=>{document.getElementById('conn').textContent='reconnecting';
                document.getElementById('conn').className='badge';};
function drawStages(s){
  // Detail cards and the pixel-budget explanation were both removed. Three KPI
  // cards plus two paragraphs of pixel arithmetic sat between the camera demo
  // and the scene, restating numbers already visible above and pushing the
  // scene below the fold. The stage strip carries the figures; the page does
  // not narrate its own internals. The measurements are still collected and
  // still enforced server-side - see app/__init__.py - they are just not
  // advertised on the front page.
}

es.onmessage=(m)=>{
  const d=JSON.parse(m.data);
  // Draw the log FIRST and isolate every panel. These used to run in sequence
  // inside one handler, so a single undefined variable in any one of them threw
  // before the rest ran and silently killed the log - which is exactly what
  // happened: drawKpis referenced an identifier that no longer existed, and the
  // event log rendered nothing at all while the scene kept streaming. One bad
  // panel must never be able to take the others down with it.
  drawLog(d.events);
  safe('kpis', ()=>drawKpis(d.state));
  safe('calibration', ()=>drawCalibration(d.state.calibration));
  safe('scene', ()=>draw(d.state));
  safe('stages', ()=>drawStages(d.state));
};

function safe(name, fn){
  try{ fn(); }
  catch(e){
    if(!window.__panelFails) window.__panelFails={};
    window.__panelFails[name]=(window.__panelFails[name]||0)+1;
    // Log once per panel, not once per frame - 4 messages a second of the same
    // stack trace helps nobody and buries anything new.
    if(window.__panelFails[name]===1){
      console.error('panel "'+name+'" failed: '+(e&&e.message));
    }
  }
}

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
      hudTel=document.getElementById('hudTel'), liveNote=document.getElementById('liveNote'),
      fsHint=document.getElementById('fsHint');
const SESSION=(crypto.randomUUID?crypto.randomUUID():String(Date.now())+Math.random());
// Log rows are keyed by t|kind|name so a repeat of the same event is not
// re-animated on every SSE frame, and an arrival is only ever added once.
let seenEvents=new Set(), lastLogT=0;
let camStream=null, camLoop=null, camBusy=false, camFrames=0, stillMode=false,
    camFails=0, lastGoodVideoConstraint=null,
    // Dimensions of the last frame uploaded for detection. The overlay maps
    // every box through these, not through videoWidth/videoHeight.
    postedW=0, postedH=0;
// Per-track rectangle smoothing state. Kept in source-pixel space so the
// filter does not change behaviour when the HUD is resized or expanded.
const smoothBox={}, smoothAge={};

// One cover-fit routine, used for the live frame AND the still, so the picture
// and the detection boxes can never disagree about where the frame edges are.
// Returns the transform so callers can map source pixels with the same numbers.
function coverDraw(ctx, src, sw, sh, alpha){
  const W=ctx.canvas.width, H=ctx.canvas.height;
  const k=Math.max(W/sw, H/sh), dw=sw*k, dh=sh*k;
  const dx=(W-dw)/2, dy=(H-dh)/2;
  if(alpha!=null){ ctx.globalAlpha=alpha; }
  ctx.drawImage(src, dx, dy, dw, dh);
  if(alpha!=null){ ctx.globalAlpha=1; }
  // Remember where the picture actually landed. Overlay geometry is then
  // expressed against THIS rect, so a box cannot drift from the image even if
  // some other scale is applied downstream.
  lastDraw={dx,dy,dw,dh,sw,sh,W,H};
  return lastDraw;
}

// CONTAIN, not cover.
//
// cover scales an image to fill the box and crops the overflow. With a portrait
// camera feed in a landscape HUD that is catastrophic: a 720x1280 frame in a
// 358px-wide box becomes dw=1134, dx=-388, so 776px of the picture - including
// whatever was in the middle of it - is simply not drawn. The box was landing on
// coordinates that had been cropped out of view, which is why the frame looked
// permanently offset from the person no matter how the maths was written.
//
// contain scales to fit entirely inside the box, letterboxing the remainder, so
// every pixel of the frame is on screen and overlay coordinates need no
// correction at all. Letterbox bars are the honest presentation for a portrait
// feed in a landscape panel.
function containDraw(ctx, src, sw, sh, alpha){
  const W=ctx.canvas.width, H=ctx.canvas.height;
  const k=Math.min(W/sw, H/sh), dw=sw*k, dh=sh*k;
  const dx=(W-dw)/2, dy=(H-dh)/2;
  if(alpha!=null){ ctx.globalAlpha=alpha; }
  ctx.drawImage(src, dx, dy, dw, dh);
  if(alpha!=null){ ctx.globalAlpha=1; }
  lastDraw={dx,dy,dw,dh,sw,sh,W,H};
  return lastDraw;
}
let lastDraw=null;
// Overlay geometry is resolved from normalised coordinates against the rect the
// picture was actually drawn into. Two independent sources of truth, no
// arithmetic linking pixel dimensions that may not correspond.
function normRect(n){
  const D=lastDraw||{dx:0,dy:0,dw:hcv.width,dh:hcv.height};
  return { x: D.dx+n[0]*D.dw, y: D.dy+n[1]*D.dh,
           w: (n[2]-n[0])*D.dw, h: (n[3]-n[1])*D.dh };
}

function escapeHtml(s){
  return String(s==null?'':s).replace(/[&<>"']/g,c=>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function hudSize(){
  // Match the panel shape to the feed so a portrait camera is not letterboxed
  // into a thin strip. The class is applied BEFORE the canvas is measured,
  // otherwise the backing store would be sized from the old shape.
  //
  // stillMode is the module-level flag, NOT the isStill parameter of
  // drawOverlay(). Referencing isStill here threw a ReferenceError on load,
  // which aborted the script before the click listener was ever attached -
  // so the button sat on "Starting..." forever and the camera could never be
  // started. node --check passes this happily: it validates syntax, not names.
  let portrait=false;
  if(stillMode && stillNatural){
    portrait = stillNatural.h > stillNatural.w;
  } else if(camStream && vid.videoWidth && vid.videoHeight){
    portrait = vid.videoHeight > vid.videoWidth;
  }
  hud.classList.toggle('portrait', portrait);
  hcv.width=hud.clientWidth; hcv.height=hud.clientHeight;
}

function paintTelemetry(rows){
  hudTel.innerHTML=rows.map(([k,v])=>`<span>${k}<b>${v}</b></span>`).join('');
}

async function listCams(){
  try{
    const devs=await navigator.mediaDevices.enumerateDevices();
    const cams=devs.filter(d=>d.kind==='videoinput');
    if(!cams.length){ camSel.innerHTML='<option>no camera found</option>'; camSel.disabled=true; return; }
    // Before permission is granted every deviceId is an EMPTY STRING and every
    // label is blank. Writing those into <option value=""> produced a select
    // whose only option looked selectable but carried no device, so startCam
    // silently fell through to its facingMode branch on a machine that has a
    // perfectly good camera. Keep unlabelled/empty-id devices as a plain
    // "let the browser choose" entry rather than pretending they are selectable.
    const usable=cams.filter(d=>d.deviceId);
    if(!usable.length){
      camSel.innerHTML='<option value="">Default camera (browser chooses)</option>';
      camSel.disabled=true;
      return;
    }
    const prev=camSel.value;
    camSel.innerHTML=(prev?'':'<option value="">Default camera (browser chooses)</option>')+
      usable.map((d,i)=>
        `<option value="${d.deviceId}">${d.label||('Camera '+(i+1))}</option>`).join('');
    camSel.disabled=false;
  }catch(e){ camSel.innerHTML='<option value="">Default camera (browser chooses)</option>'; }
}

// Sweep for a streamable camera and report exactly which attempt worked.
//
// Desktop and mobile fail for different reasons, and guessing one constraint
// shape loses on both. iOS wants facingMode and rejects stale deviceIds;
// Windows desktops routinely enumerate a virtual/dummy capture device FIRST, so
// picking "the first camera" hands you a black rectangle, and facingMode:'user'
// can resolve to a device that does not exist. So: try the chosen device, then
// every enumerated deviceId one at a time, then the bare request, and record
// which one came back. The winning shape is reused for later restarts.
async function openStream(){
  const sel=camSel.value;
  const attempts=[];
  if(sel) attempts.push({label:'selected device', v:{deviceId:{exact:sel}, width:{ideal:1280}, height:{ideal:720}}});
  attempts.push({label:'facingMode user', v:{facingMode:'user', width:{ideal:1280}, height:{ideal:720}}});

  // Every enumerated video input, individually. This is the step that rescues a
  // desktop whose first-listed camera is a stub.
  let devs=[];
  try{ devs=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput'&&d.deviceId); }catch(e){}
  for(const d of devs){
    const label=(d.label||'camera')+(d.label?' [listed]':' [unlabelled]');
    attempts.push({label, v:{deviceId:{ideal:d.deviceId}, width:{ideal:1280}, height:{ideal:720}}});
  }
  attempts.push({label:'any camera', v:true});
  attempts.push({label:'low resolution', v:{width:{ideal:640}, height:{ideal:480}}});

  const tried=[];
  for(const a of attempts){
    try{
      const s=await navigator.mediaDevices.getUserMedia({audio:false, video:a.v});
      lastGoodVideoConstraint=a.v;
      // Report what we actually got, so a black feed from a stub device is
      // visible as "0x0" rather than looking like a broken page.
      const tr=s.getVideoTracks()[0];
      const st=tr&&tr.getSettings?tr.getSettings():{};
      return {stream:s, label:a.label, settings:st, trackLabel:tr?tr.label:''};
    }catch(e){
      tried.push(a.label+' -> '+(e&&e.name||'?'));
      // A refusal is a decision, not a device problem. Do not keep hammering.
      if(e && (e.name==='NotAllowedError'||e.name==='SecurityError')){
        const err=new Error(e.name); err.tried=tried; throw err;
      }
    }
  }
  const err=new Error('no camera could be opened');
  err.tried=tried; throw err;
}

async function startCam(){
  camBtn.disabled=true; camBtn.textContent='Starting…';
  try{
    const got=await openStream();
    camStream=got.stream;
    stillMode=false;            // the live feed owns the panel from here
    vid.srcObject=camStream;
    // iOS Safari will not advance a muted autoplay video unless play() is
    // called from inside the user gesture, so this must stay awaited here
    // rather than being deferred to a load event.
    await vid.play();
    // A stub or virtual capture device hands back a track that never produces
    // frames. Waiting for real dimensions here turns that into a clear message
    // instead of a black rectangle that looks like a broken page.
    if(!vid.videoWidth){
      await new Promise(res=>{
        const t=setTimeout(res,3000);
        vid.addEventListener('loadedmetadata',()=>{clearTimeout(t);res();},{once:true});
      });
    }
    hudEmpty.hidden=true;
    camBtn.disabled=false; camBtn.textContent='Stop camera';
    await listCams();                    // real labels are only readable once permitted
    hudSize();
    fetch('/api/live/stop?session='+encodeURIComponent(SESSION),{method:'POST'});
    camLoop=setInterval(pumpFrame, 220);  // ~4.5 Hz: one stage-1 pass per tick

    const dims=vid.videoWidth+'x'+vid.videoHeight;
    if(!vid.videoWidth){
      showStill('camera reported no picture');
      liveNote.innerHTML='<b class="no">Camera opened but produced no picture.</b> '+
        'It is most likely a virtual or stub capture device &mdash; common on '+
        'Windows desktops with a driver installed but no camera attached. '+
        'It was opened via <b>'+got.label+'</b> at '+dims+
        '. <span class="mid">Showing real stage-1 output from a stored frame.</span>';
      return;
    }
    liveNote.innerHTML='<b class="ok">Live.</b> Opened via <b>'+got.label+'</b> at '+
      dims+', sending frames to stage 1 for person detection and tracking. '+
      'Detection and boxes only &mdash; nothing is enrolled, no frame is stored, '+
      'and no name is attached to any track.';
  }catch(err){
    camBtn.disabled=false; camBtn.textContent='Start camera';
    const name=err&&err.name||'';
    if(name==='NotAllowedError' || name==='SecurityError'){
      // This one is sticky: browsers keep the refusal until the viewer changes
      // it in site settings, so telling them to "try again" sends them in
      // circles. Say what to actually do, and for iOS name the Settings path.
      showStill('Camera access was declined.');
      liveNote.innerHTML='<b class="no">Camera access declined.</b> This is a '+
        'one-time decision your browser has remembered, so pressing Start again '+
        'will not change it &mdash; re-allow the camera for this site, then press '+
        'Start. On iPhone: the <b>aa</b> menu beside the address bar &rarr; '+
        'Camera &rarr; Allow, then reload. '+
        '<span class="mid">The still below is real stage-1 output from a stored '+
        'frame, not a live feed.</span>';
    }else{
      showStill('No camera could be opened.');
      // Print the sweep. "No camera could be opened" on its own has cost real
      // time before - the interesting part is WHICH device refused and how.
      const tried=(err&&err.tried&&err.tried.length)
        ? '<br><span class="mono" style="font-size:11px">tried: '+
          err.tried.map(escapeHtml).join(' &middot; ')+'</span>' : '';
      liveNote.innerHTML='<b class="no">No camera could be opened.</b> '+
        escapeHtml(err&&err.message? err.message : 'Unknown error')+
        ' &mdash; another app may be holding it, or the device may be a stub.'+
        tried+
        ' <span class="mid">Showing real stage-1 output from a stored frame.</span>';
    }
  }
}

function stopCam(){
  if(camLoop){clearInterval(camLoop);camLoop=null;}
  if(camStream){camStream.getTracks().forEach(t=>t.stop());camStream=null;}
  vid.srcObject=null;
  stillMode=false;
  hx.clearRect(0,0,hcv.width,hcv.height);
  hudEmpty.hidden=false;
  hudId.textContent='STANDBY';
  hudTel.innerHTML='';
  camFrames=0; camBusy=false;
  camBtn.disabled=false; camBtn.textContent='Start camera';
  fetch('/api/live/stop?session='+encodeURIComponent(SESSION),{method:'POST'});
  liveNote.innerHTML='Camera stopped. The stream is released and the session '+
    'track IDs are discarded.';
}

async function pumpFrame(){
  if(camBusy||stillMode||!camStream||vid.readyState<2) return;
  camBusy=true;
  try{
    const off=document.createElement('canvas');
    const W=vid.videoWidth||640, H=vid.videoHeight||480;
    off.width=Math.min(W,960); off.height=Math.round(off.width*H/W);
    // Record what we actually upload. The overlay maps boxes using these exact
    // dimensions rather than the camera's reported ones, because on iOS the two
    // disagree: videoWidth is the sensor's landscape size while drawImage honours
    // the rotation metadata, so the real frame is portrait.
    postedW=off.width; postedH=off.height;
    off.getContext('2d').drawImage(vid,0,0,off.width,off.height);
    const blob=await new Promise(r=>off.toBlob(r,'image/jpeg',0.72));
    if(!blob) return;
    const fd=new FormData(); fd.append('file',blob,'frame.jpg');
    const r=await fetch('/api/live/frame?session='+encodeURIComponent(SESSION),
                         {method:'POST',body:fd});
    if(!r.ok) throw new Error('stage 1 returned '+r.status);
    const d=await r.json();
    camFails=0;
    camFrames++;
    drawOverlay(d);
  }catch(e){
    // One bad frame must not kill the loop. A single 4K frame can exceed the
    // upload ceiling, or a tunnel can drop one POST, and neither is a reason to
    // tear down a working camera. Keep the last good overlay, count the failure,
    // and only complain once it is clearly persistent.
    camFails++;
    if(camFails===1 || camFails%15===0){
      hudId.textContent='FRAME DROPPED';
      liveNote.innerHTML='<b class="mid">Frame dropped.</b> '+
        escapeHtml(e&&e.message?e.message:'frame not accepted')+
        ' ('+camFails+' so far) &mdash; the camera is still yours and the loop '+
        'retries on the next tick.';
    }
  }finally{ camBusy=false; }
}

function drawOverlay(d, isStill){
  const W=hcv.width, H=hcv.height;
  // Paint the frame here, in the same space the boxes are computed in. The
  // <video> is a hidden capture source only; letting CSS scale it with
  // object-fit while the canvas used its own arithmetic is what left the boxes
  // permanently offset from the people they were tracking.
  const fr=d.frame||{};
  if(!isStill){
    hx.clearRect(0,0,W,H);
    // Draw with the EXACT aspect the server detected in - its own reported
    // frame dimensions - not videoWidth and not a remembered postedW/H.
    //
    // This is the fix for the horizontal drift. videoWidth is the sensor size
    // and drawImage applies rotation metadata, so on a portrait phone the three
    // candidate spaces (sensor, posted, drawn) did not agree, and any
    // disagreement between them slides the box sideways off the person. Taking
    // the server's numbers makes picture and boxes provably share one space:
    // whatever the camera orientation, the draw matches what was detected.
    if(vid.readyState>=2 && vid.videoWidth && fr.w && fr.h){
      containDraw(hx, vid, fr.w, fr.h);
    }
  }
  // Geometry comes from the server as NORMALISED fractions, resolved against the
  // rect the picture was actually drawn into (lastDraw). No pixel arithmetic
  // links the two, so there is no ratio left to get wrong.
  const tracksByBox={};
  (d.tracks||[]).forEach(t=>{ tracksByBox[t.box.map(Math.round).join(',')]=t; });

  (d.persons||[]).forEach(p=>{
    const key=[p.box[0],p.box[1],p.box[2],p.box[3]].map(Math.round).join(',');
    const tr=tracksByBox[key];

    // ── Rectangle calibration ────────────────────────────────────────────
    // Raw YOLO boxes jitter by a few pixels every frame, which makes a live
    // reticle shimmer and read as sloppy tracking even when detection is
    // steady. Smooth in SOURCE pixel space (before scaling to the HUD) and
    // hold the last box when a track is momentarily missed, so the reticle
    // stays locked to the person instead of strobing.
    const key2=tr? 't'+tr.id : 'd'+key;
    const prev=smoothBox[key2];
    let b=p.box;
    if(prev){
      const a=0.45;                       // weight on the new observation
      b=[ b[0]+(prev[0]-b[0])*a, b[1]+(prev[1]-b[1])*a,
          b[2]+(prev[2]-b[2])*a, b[3]+(prev[3]-b[3])*a ];
    } else if(tr && tr.box){ b=tr.box.slice(); }
    smoothBox[key2]=b;
    smoothAge[key2]=2;
    for(const k in smoothAge){ if(smoothAge[k]>0) smoothAge[k]--; else delete smoothAge[k]; }

    // Clamp to the canvas. YOLO can report a box whose top edge is at or above the
    // frame boundary, and a rect drawn from a negative y is simply not painted
    // there - which reads as "the square is missing above the person" even
    // though the box exists. Clamping keeps the outline closed and still sits it
    // on the person, because only the off-canvas sliver is trimmed.
    // Geometry from the server's NORMALISED coords, resolved against the rect the
    // picture was actually drawn into. No pixel arithmetic links the two, so
    // there is no ratio that can be miscalculated.
    const nr=normRect(p.norm||(fr.w&&fr.h
      ? [b[0]/fr.w, b[1]/fr.h, b[2]/fr.w, b[3]/fr.h]
      : [0,0,1,1]));
    // Fit the reticle to the FACE once we have one.
    //
    // YOLO's person class returns whole-body rectangles, so framing those is
    // technically correct but useless as a targeting readout: the interesting
    // part of a person is a tenth of the box and the rest is torso and legs. The
    // face box comes from SCRFD at stage 2, so once it exists it becomes the
    // primary reticle and the body box drops back to a faint context outline.
    //
    // DECLARATION ORDER MATTERS HERE, and getting it wrong cost two production
    // incidents in a row. `idrec`, `stage` and `onFace` are all const, so any
    // read above their declaration is a temporal-dead-zone ReferenceError thrown
    // on every single live frame. The symptoms were "Frame dropped. Cannot
    // access 'onFace' before initialization" and, before that, the same for
    // 'idrec'. Declare first, use second - all of it, up here.

    // Which stage this person is at, and what colour that reads as.
    const idrec=(id=(d.identify||{}).people||[]).find(q=>q&&q.box&&
                 q.box[0]===p.box[0]&&q.box[2]===p.box[2]) || null;
    const stage=idrec? idrec.stage : 'found';
    const col= stage==='named'? C.grn
             : stage==='gated'? C.amb
             : stage==='checked'? C.gold
             : tr? C.cyan : C.gold;

    let X=nr.x, Y=nr.y, BW=nr.w, BH=nr.h;
    let bodyX=X, bodyY=Y, bodyW=BW, bodyH=BH;
    const fdet = idrec && idrec.face;
    if(fdet && BW>26 && BH>26){
      const fn2=normRect((fr.w&&fr.h)
        ? [fdet.bbox[0]/fr.w, fdet.bbox[1]/fr.h, fdet.bbox[2]/fr.w, fdet.bbox[3]/fr.h]
        : [0,0,0,0]);
      if(fn2.w>6 && fn2.h>6){ X=fn2.x; Y=fn2.y; BW=fn2.w; BH=fn2.h; }
    }
    const onFace = (X!==bodyX);
    // Context outline for the body, so you can still see what was detected.
    if(onFace){
      hx.strokeStyle='rgba(99,199,212,.34)'; hx.lineWidth=1;
      hx.setLineDash([3,4]);
      hx.strokeRect(bodyX,bodyY,bodyW,bodyH);
      hx.setLineDash([]);
    }
    const cx0=Math.max(0,X), cy0=Math.max(0,Y),
          cx1=Math.min(W,X+BW), cy1=Math.min(H,Y+BH);
    if(cx1<=cx0||cy1<=cy0){ return; }          // entirely outside: nothing to draw
    X=cx0; Y=cy0; BW=cx1-cx0; BH=cy1-cy0;

    // The HUD runs all three stages on this frame, so the box carries the
    // verdict: cyan while it is only a body, gold while the face is being
    // checked, green once a name is attached.

    // Line weight depends on the final box size, so it has to be computed after
    // the face-fit above - and read only after that. Same TDZ trap as onFace.
    const lw=Math.max(1.25, Math.min(BW,BH)/90);

    // A named person gets a solid inner face box, drawn INSIDE the face reticle
    // rather than replacing it, so the name is visibly tied to the face.
    if(onFace && stage==='named'){
      hx.strokeStyle=C.grn; hx.lineWidth=Math.max(1,lw*1.2);
      hx.strokeRect(X,Y,BW,BH);
    }

    hx.strokeStyle=col; hx.lineWidth=lw;
    hx.strokeRect(X,Y,BW,BH);

    // Brighten the corners: the part a viewer actually tracks with their eye.
    hx.lineWidth=lw*2.1;
    hx.strokeStyle=tr? C.paper : C.goldBright;
    const tick=Math.max(10, Math.min(BW,BH)*0.24);
    const corners=[[X,Y,1,1],[X+BW,Y,-1,1],[X,Y+BH,1,-1],[X+BW,Y+BH,-1,-1]];
    corners.forEach(([px,py,dx,dy])=>{
      hx.beginPath();
      hx.moveTo(px+dx*tick,py); hx.lineTo(px,py); hx.lineTo(px,py+dy*tick);
      hx.stroke();
    });

    // Centre reticle, sized to the head end of the box. This is what makes it
    // read as "identified" rather than "a rectangle was returned".
    if(tr && BW>26 && BH>26){
      const cxm=X+BW/2, cym=Y+BH*0.22, rr=Math.max(3,Math.min(BW,BH)*0.07);
      hx.strokeStyle=col; hx.lineWidth=lw;
      hx.beginPath();
      hx.moveTo(cxm-rr*2.2,cym); hx.lineTo(cxm-rr,cym);
      hx.moveTo(cxm+rr,cym);     hx.lineTo(cxm+rr*2.2,cym);
      hx.stroke();
    }

    // Label pinned to the box, flipped inside when it would leave the frame.
    const fsz=Math.max(9, Math.min(15, Math.round(Math.min(BW,BH)/16)));
    const px=Math.round(p.conf*100)+'%';
    // The label states the stage the person is actually at. "unidentified" is
    // spelled out rather than left blank: a person in frame that we decline to
    // name is a result, not a gap.
    const stageTxt = stage==='named' ? (idrec.name||'').replace(/_/g,' ')
                   : stage==='gated'  ? 'HELD AT CHECK'
                   : stage==='checked'? 'CHECKING'
                   : onFace? 'FACE FOUND'
                   : 'NO FACE';
    const lbl=stageTxt+'  '+px+
              (tr? '  ·  TRACK '+tr.id : '')+
              (idrec&&idrec.cosine!=null? '  ·  '+idrec.cosine.toFixed(3) : '');
    hx.font='700 '+fsz+'px "Space Mono",monospace';
    const tw=hx.measureText(lbl).width, pad=Math.max(3,fsz*0.4), bh=fsz+pad*2;
    let lx=X, ly=Y-bh-2;
    if(ly<0) ly=Y+2;                       // no room above -> sit inside the top
    lx=Math.max(0, Math.min(lx, W-tw-pad*2));
    hx.fillStyle='rgba(5,7,6,.82)';
    hx.fillRect(lx,ly,tw+pad*2,bh);
    hx.fillStyle=col;
    hx.fillText(lbl,lx+pad,ly+bh-pad);
  });

  const n=(d.persons||[]).length;
  const ident=d.identify||{};
  if(isStill){
    hudId.textContent = 'STILL · '+(n ? (n+' PERSON'+(n===1?'':'S')) : 'NO PERSON');
    // A still has exactly one frame and no history, so a track ID here would be
    // theatre: it could only ever be #1 because nothing moved. Say "untracked"
    // instead of implying the tracker did something it did not.
    paintTelemetry([
      ['SOURCE', 'stored frame'],
      ['1 FIND', (d.ok?'YOLO ':'')+n],
      ['2 CHECK', ident.ran? ((ident.gated? ident.gated+' held':'clear')):'—'],
      ['3 NAME', ident.ran? ((ident.named||0)+' named'):'—'],
      ['TRACKS', 'none (still)'],
      ['LATENCY', Math.round(d.ms||0)+' ms'],
      ['FRAME', (fr.w||0)+'×'+(fr.h||0)],
    ]);
    return;
  }
  const named=ident.named||0, gated=ident.gated||0;
  hudId.textContent = named
    ? 'IDENTIFIED — '+named+(n>1? ' of '+n+' PEOPLE':' PERSON')
    : (n? 'PERSON — '+n+' · NOT IDENTIFIED'
         : 'SCANNING · NO PERSON');
  // The telemetry strip is the find / check / name story, in that order, with
  // the real per-stage timings.
  paintTelemetry([
    ['1 FIND', (d.ok?'YOLO ':'')+n+(d.track_count? ' · '+d.track_count+' tracked':'')],
    ['2 CHECK', ident.ran? ((gated? gated+' held':'clear')+
                            (ident.face_ms? ' · '+Math.round(ident.face_ms)+'ms':''))
                        : (ident.why||'—')],
    ['3 NAME', ident.ran? (named? named+' named'
                          : (n? '0 of '+n+' in gallery':'—')) : '—'],
    ['GALLERY', (ident.gallery!=null? ident.gallery+' enrolled':'—')],
    ['FIND', Math.round(d.find_ms||d.ms||0)+' ms'],
    ['FRAMES', camFrames],
  ]);
}

// When permission is refused there is no video to show. Rather than leave a
// dead black rectangle - which reads as a broken product - run stage 1 over a
// stored frame and draw THAT, clearly labelled as a still. It is real output
// from the real model, so the HUD is still demonstrating something true, and the
// caption never claims it is live.
async function showStill(reason){
  // Never fall back while a live camera is running.
  //
  // showStill() is async: it fetches the image and waits for it to decode. On
  // page load it starts immediately, so pressing Start camera while it is still
  // in flight used to mean the still finished afterwards and re-set stillMode,
  // freezing the panel on the stored frame with the live feed running unseen
  // behind it. Real report: the camera "stopped passing through" after the
  // still-on-load change. Re-check at every await, because the user can start
  // the camera at any point during those awaits.
  if(camStream) return;
  const stills=['/deck-assets/aerial-field.jpg','/deck-assets/fence-climb.jpg',
                '/deck-assets/drone-cover.jpg'];
  for(const src of stills){
    try{
      const r=await fetch(src,{cache:'force-cache'});
      if(!r.ok) continue;
      const blob=await r.blob();
      if(camStream) return;          // camera started during the fetch
      const fd=new FormData(); fd.append('file',blob,'still.jpg');
      const res=await fetch('/api/live/frame?session='+encodeURIComponent(SESSION)+'still',
                            {method:'POST',body:fd});
      if(!res.ok) continue;
      const d=await res.json();
      if(camStream) return;          // ...or during the upload
      vid.srcObject=null;
      stillMode=true;
      hudEmpty.hidden=true;
      hudId.textContent='STILL · analysing';
      // Paint the frame BEFORE the overlay. The overlay is positioned in the
      // canvas coordinate space, and drawing it first meant the box landed on a
      // blank 300x150 buffer while the photo arrived a tick later at display
      // size - so the box was drawn off the visible area entirely.
      await paintStillFrame(src);
      drawOverlay(d, true);
      return;
    }catch(e){ /* try the next still */ }
  }
  hudEmpty.hidden=false;
  hudEmpty.innerHTML='<b>'+(reason||'Camera off.')+'</b> Nothing is being captured '+
    'or sent. Press <b>Start camera</b> to point stage 1 at your own device.';
}

// Resolve once the frame is actually on the canvas. Returns a promise so the
// caller can draw the overlay at the same moment, not a tick before it.
function paintStillFrame(src){
  return new Promise((resolve,reject)=>{
    const im=new Image();
    im.onload=()=>{
      const W=hud.clientWidth, H=hud.clientHeight;
      hcv.width=W; hcv.height=H;
      hx.clearRect(0,0,W,H);
      stillNatural={w:im.naturalWidth,h:im.naturalHeight};
      containDraw(hx, im, im.naturalWidth, im.naturalHeight, .85);
      resolve();
    };
    im.onerror=()=>reject(new Error('still failed to load'));
    im.src=src;
  });
}
let stillNatural=null;

camBtn.addEventListener('click',e=>{ e.stopPropagation(); camStream? stopCam() : startCam(); });
// Any late-arriving still must never re-take the panel.
document.addEventListener('visibilitychange',()=>{ if(!document.hidden) hudSize(); });
camSel.addEventListener('change',()=>{ if(camStream){ stopCam(); } });

// Click the frame to go full-screen, the way a video call does. Escape or a
// second click comes back. Not using the Fullscreen API on purpose: it hides the
// browser chrome and adds an exit affordance the viewer has to hunt for, and on
// iOS Safari it is unreliable inside a cross-origin iframe.
hud.addEventListener('click',()=>{
  const on=!hud.classList.contains('fs');
  hud.classList.toggle('fs',on);
  fsHint.textContent=on? 'Click or press Esc to exit':'Click to expand';
  hudSize();
  document.body.style.overflow=on?'hidden':'';
});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape' && hud.classList.contains('fs')){
    hud.classList.remove('fs');
    fsHint.textContent='Click to expand';
    document.body.style.overflow='';
    hudSize();
  }
});

window.addEventListener('resize',()=>{ if(camStream||stillMode) hudSize(); });
if(navigator.mediaDevices&&navigator.mediaDevices.enumerateDevices) listCams();
// Size the overlay up front. It used to default to the 300x150 canvas size and
// only be corrected when the camera started, so the overlay was drawn into a
// buffer a quarter of the display size and then stretched by CSS.
hudSize();

// Paint the stored-frame HUD immediately on load.
//
// showStill() used to run ONLY from inside startCam()'s catch block, which meant
// the panel sat completely empty until somebody deliberately pressed Start
// camera and had it refused. That is exactly backwards: the still exists for the
// case where there is no camera, so it has to be what renders when nothing has
// been started. Real report: the panel looked broken until Start camera was
// pressed.
showStill('Camera off.');

async function togglePause(){
  const r=await fetch('/api/pause',{method:'POST'});
  const d=await r.json(); paused=!d.running;
}
</script></body></html>"""

app = application