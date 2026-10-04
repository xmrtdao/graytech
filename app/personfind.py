"""
Person detection (YOLO) — stage 1 of the rig loop.

The rig spec puts a cheap person-find in FRONT of the face models, and that is
what this module is. It answers one question: is there a person in this frame?
Only if yes is it worth spending the expensive model on a crop.

    RTSP/H.265 -> NVDEC -> YOLO person detect (wide / low zoom)
      -> if person: command gimbal absolute zoom + track
      -> when face >= ~80 px: SCRFD -> align -> ArcFace

Measured on this machine (CPU, yolov8n.onnx, 640x640):

    median 251 ms per frame.

That is ~6x the cost of the face pipeline (~40 ms), which is exactly why the spec
puts it first and gates on it. It is a stage, not an optimisation.

IMPORTANT, and measured rather than assumed
-------------------------------------------
YOLO person-find expects a SCENE. The Faces dataset this demo analyses is tight,
centred headshots, and on those it is weak: 4 of 12 sampled portraits returned a
person box at conf >= 0.25, because a face cropped to fill the frame is not the
distribution COCO's person class was trained on. Feeding YOLO a crop and
believing a miss is the mirror image of the error the workup warns about
("don't downscale 4K to 640"), and it is why this stage is wired to the live
camera - a real scene - and not in front of every portrait scan.

So: use this where there is a scene. Do not read a null result here as "no
people" when the input was a headshot.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("face-service.person")

MODEL_PATH = Path(os.environ.get(
    "GRAYTECH_YOLO", "./models/yolov8n.onnx"))
CONF_THRESH = float(os.environ.get("GRAYTECH_YOLO_CONF", "0.25"))
IMGSZ = int(os.environ.get("GRAYTECH_YOLO_IMGSZ", "640"))
# COCO class 0. YOLO finds 80 things; we only care about one of them.
PERSON_CLASS = 0
CLASS_NAMES = ["person"]
# Overlap above which two boxes are the same person. 0.45 is the usual YOLO
# default; tighter merges neighbours, looser splits one crowd.
NMS_IOU = float(os.environ.get("GRAYTECH_YOLO_NMS", "0.45"))


class PersonFinder:
    """YOLO person detection. One session per process, guarded by a lock."""

    def __init__(self) -> None:
        self.sess = None
        self.lock = threading.Lock()
        self.last_ms: float = 0.0
        self.available: bool = False
        self.detail: str = "not loaded"
        self.conf: float = CONF_THRESH
        # Its own identity space. Track IDs here are "this box over time" and
        # never come from a face match.
        self.tracker = Tracker()

    def load(self) -> None:
        """Load once, in the lifespan handler. Never per request."""
        try:
            import cv2                      # noqa: F401  (presence check)
            import onnxruntime as ort

            if not MODEL_PATH.exists():
                self.detail = f"no weights at {MODEL_PATH}"
                log.warning("YOLO person-find disabled: %s", self.detail)
                return
            self.sess = ort.InferenceSession(
                str(MODEL_PATH), providers=["CPUExecutionProvider"])
            self.available = True
            self.detail = f"{MODEL_PATH.name} @ {IMGSZ}px"
            log.info("YOLO person-find ready: %s", self.detail)
        except Exception as exc:                        # noqa: BLE001
            self.detail = f"{type(exc).__name__}: {exc}"
            log.warning("YOLO person-find failed to load: %s", self.detail)

    # -- inference ---------------------------------------------------------
    def detect(self, image: "np.ndarray") -> dict:
        """
        Detect persons in a BGR image.

        Returns a dict that always has the same shape, so a caller never has to
        distinguish "found none" from "model unavailable" by exception:
        `ok` says whether a real inference happened.
        """
        out = {"ok": False, "persons": [], "count": 0, "ms": 0.0,
               "detail": self.detail, "conf": CONF_THRESH}
        if not self.available or self.sess is None or image is None:
            return out
        import cv2

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return out

        # Letterbox to a square, centred, zero padded - the geometry YOLO was
        # exported with. Resizing to a square without preserving aspect is the
        # single easiest way to make a correct model look broken.
        r = min(IMGSZ / h, IMGSZ / w)
        nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
        canvas = np.zeros((IMGSZ, IMGSZ, 3), np.float32)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0
        blob = blob.astype(np.float32)

        try:
            with self.lock:
                t0 = time.perf_counter()
                pred = self.sess.run(None, {"images": blob})[0]
                self.last_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:                        # noqa: BLE001
            out["detail"] = f"inference failed: {exc}"
            return out

        out["ms"] = round(self.last_ms, 1)
        out["ok"] = True

        # (1, 84, 8400) -> (8400, 84): 4 bbox then 80 class scores.
        rows = pred[0].T
        ox, oy = (IMGSZ - nw) / 2.0, (IMGSZ - nh) / 2.0
        cand = []
        for row in rows:
            score = float(row[4 + PERSON_CLASS])
            if score < CONF_THRESH:
                continue
            cx, cy, bw, bh = (float(v) for v in row[:4])
            x1 = max(0.0, (cx - bw / 2 - ox) / r)
            y1 = max(0.0, (cy - bh / 2 - oy) / r)
            x2 = min(float(w), (cx + bw / 2 - ox) / r)
            y2 = min(float(h), (cy + bh / 2 - oy) / r)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cand.append({"box": [x1, y1, x2, y2], "conf": score})

        # NMS is mandatory and must happen HERE. The exported graph stops at the
        # raw 8400-row prediction - Ultralytics does suppression inside its own
        # Python postprocess, which an ONNX consumer does not get for free.
        # Without this, one person comes back as ~10 boxes a pixel apart, the
        # "independent count" is inflated tenfold, and the tracker hands out ten
        # track IDs for one human.
        persons = self._nms(cand, NMS_IOU)
        for p in persons:
            p["box"] = [round(v, 1) for v in p["box"]]
            p["conf"] = round(p["conf"], 3)
        # Largest first: the near person is the one worth cropping.
        persons.sort(key=lambda p: -(p["box"][2] - p["box"][0])
                     * (p["box"][3] - p["box"][1]))
        out["persons"] = persons
        out["count"] = len(persons)
        out["raw_candidates"] = len(cand)
        return out

    @staticmethod
    def _nms(cands: list[dict], thresh: float) -> list[dict]:
        """Greedy non-maximum suppression. Highest score wins, overlaps die."""
        kept: list[dict] = []
        for c in sorted(cands, key=lambda d: -d["conf"]):
            drop = False
            for k in kept:
                if Tracker._iou(c["box"], k["box"]) > thresh:
                    drop = True
                    break
            if not drop:
                kept.append(c)
        return kept

    def detect_bytes(self, data: bytes) -> dict:
        """Detect from encoded image bytes. Bad input is a null result, not a raise."""
        import cv2
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "persons": [], "count": 0, "ms": 0.0,
                    "detail": "undecodable image", "conf": CONF_THRESH}
        return self.detect(img)

    def detect_and_track(self, image: "np.ndarray") -> dict:
        """
        Detect, then advance the tracker one frame.

        The count and the track IDs returned here are YOLO's own. They are
        computed without reference to the face gallery, the face model, or the
        match threshold, and they stay meaningful with an empty gallery - which
        is the point: most missions need the count, not the name.
        """
        res = self.detect(image)
        if res["ok"]:
            res["tracks"] = self.tracker.update(res["persons"])
            res["track_count"] = len(res["tracks"])
        else:
            res["tracks"] = self.tracker.update([])
            res["track_count"] = len(res["tracks"])
        return res

    def detect_bytes_tracked(self, data: bytes) -> dict:
        import cv2
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "persons": [], "tracks": [], "count": 0,
                    "track_count": 0, "ms": 0.0, "detail": "undecodable image",
                    "conf": CONF_THRESH}
        return self.detect_and_track(img)


class Tracker:
    """
    Minimal IoU tracker - a ByteTrack-shaped lite, and deliberately separate from
    the face pipeline.

    Why it is its own thing: YOLO answers "how many people, and where", buffalo
    answers "who is this face". They must not share an identity space, because a
    track that gets its ID from a face match is exactly the failure this system
    is supposed to avoid. A track ID here is "this box, over time" and nothing
    more.

    Two-stage association, which is the part that matters: high-confidence boxes
    anchor tracks, and a low-confidence box that does not match anything is held
    in a short buffer so a one-frame miss does not split one person into two
    track IDs. That is the cheap half of what ByteTrack does, and it is the half
    that fixes the visible problem.
    """

    HIGH_CONF = 0.45          # anchor a track on this
    LOW_CONF = 0.25           # may continue a track, may not start one
    MAX_AGE = 12              # frames a track survives without a detection

    def __init__(self, iou_thresh: float = 0.3) -> None:
        self.iou_thresh = iou_thresh
        self.tracks: dict[int, dict] = {}
        self._next_id = 1
        self._lost: list[dict] = []

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, persons: list[dict]) -> list[dict]:
        """Advance one frame. `persons` is [{box, conf}]. Returns live tracks."""
        # Predict: carry every track forward one frame, ageing it.
        for tr in self.tracks.values():
            tr["age"] += 1
            tr["hits"] += 1 if tr.get("seen") else 0
            tr["seen"] = False

        unmatched = list(persons)

        # Associate strongest overlaps first, high confidence only.
        for tr in sorted(self.tracks.values(),
                         key=lambda t: -t.get("conf", 0.0)):
            if not unmatched:
                break
            best, best_iou = None, self.iou_thresh
            for p in unmatched:
                if p["conf"] < self.LOW_CONF:
                    continue
                v = self._iou(tr["box"], p["box"])
                if v > best_iou:
                    best, best_iou = p, v
            if best is not None:
                unmatched.remove(best)
                tr.update(box=best["box"], conf=best["conf"],
                          seen=True, age=0, lost=0)

        # Anything left over that is confident enough starts a NEW track. This is
        # the high/low split: a 0.26 box never gets an ID of its own.
        for p in unmatched:
            if p["conf"] < self.LOW_CONF:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                "id": tid, "box": p["box"], "conf": p["conf"],
                "age": 0, "hits": 1, "lost": 0, "seen": True,
                "born": time.time(),
            }

        # Retire tracks the detector has lost for too long.
        for tid in [t for t, tr in self.tracks.items()
                    if tr["age"] > self.MAX_AGE]:
            del self.tracks[tid]

        return self.snapshot()

    def snapshot(self) -> list[dict]:
        return [
            {"id": tr["id"],
             "box": [round(v, 1) for v in tr["box"]],
             "conf": round(tr["conf"], 3),
             "hits": tr["hits"],
             "missed": tr["age"],
             "age_s": round(time.time() - tr["born"], 1)}
            for tr in sorted(self.tracks.values(), key=lambda t: t["id"])
        ]

    def reset(self) -> None:
        self.tracks.clear()


personfinder = PersonFinder()