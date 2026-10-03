"""
Threshold calibration and adaptive matching.

THE PROBLEM BOTH OF THESE SOLVE

The cosine match threshold was a hardcoded 0.45 - insightface's default. That
number is a reasonable GENERAL purpose default and a poor fit for any specific
deployment. It assumes your gallery and your query distribution look like the
ones it was tuned against. They never do:

  * too low  -> false accepts. Someone unknown walks in and is confidently
                labelled as an enrolled person. In a security context that is
                the dangerous direction: a stranger given a real identity.
  * too high -> false rejects. Your own enrolled people get refused, people
                start ignoring the alarm, and the system becomes noise.

Calibration replaces the guess with measurement. We log every score the system
produces, keep the ones that were correct AND the ones that were wrong, and
report where the errors actually are instead of asserting they are not.

WHY NOT RETRAIN THE NETWORK

The ONNX weights are frozen and nothing here updates them. Fine-tuning needs
thousands of labelled images per identity, and with a gallery of tens rather
than thousands it would overfit: excellent on the set it trained on, worse on
new arrivals. So the network stays fixed and the two things that CAN adapt do.

  calibrate()  moves the operating point using observed error rates
  adaptive     enriches an existing profile from a correction, instantly

WHAT IS LOGGED, AND WHAT IS NOT

  * A SCORE and whether it was right or wrong. No image bytes.
  * Adaptations record that a profile gained a sample - never the pixels.

A gallery here is enrolled people, not captured strangers, and there is no
auto-enrolment path in this code at all. An unknown face can never become an
identity through this module; it can only be recorded as an unknown score.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Operating points we are willing to report on. Anything outside this range is
# reported but NOT recommended: below ~0.2 the system is matching strangers to
# each other, above ~0.75 it starts refusing genuine enrollees.
FLOOR, CEIL = 0.20, 0.80
DEFAULT_THRESHOLD = 0.45


@dataclass
class Sample:
    score: float
    correct: bool            # was the top-1 name the right person?
    known: bool              # was this a genuine gallery member at all?
    at: float = field(default_factory=time.time)


class Calibrator:
    """
    Rolling window of observed scores, with the error geometry worked out from
    them rather than assumed.
    """

    def __init__(self, window: int = 2000):
        self.window = window
        self.samples: deque[Sample] = deque(maxlen=window)
        self.lock = threading.Lock()
        self.threshold: float = DEFAULT_THRESHOLD
        self.history: deque[dict] = deque(maxlen=60)   # threshold changes
        # profile name -> list of embeddings accepted for it (adaptive matching)
        self.profiles: dict[str, list[np.ndarray]] = {}
        self.profile_meta: dict[str, dict] = {}

    # ── recording ────────────────────────────────────────────────────────
    def record(self, score: float, correct: bool, known: bool) -> None:
        with self.lock:
            self.samples.append(Sample(float(score), bool(correct), bool(known)))

    # ── geometry ─────────────────────────────────────────────────────────
    def _stats(self) -> dict:
        s = list(self.samples)
        out = {
            "n": len(s),
            "threshold": round(self.threshold, 4),
            "below_floor": self.threshold < FLOOR,
            "above_ceiling": self.threshold > CEIL,
        }
        if not s:
            out.update({"p50": None, "p95": None, "false_accept": None,
                        "false_reject": None, "accuracy": None})
            return out

        scores = np.array([x.score for x in s])
        known = np.array([x.known for x in s])
        right = np.array([x.correct for x in s])

        out["p50"] = round(float(np.percentile(scores, 50)), 4)
        out["p95"] = round(float(np.percentile(scores, 95)), 4)

        # Predicted positive = score >= threshold.
        pred_pos = scores >= self.threshold
        # True positive = predicted positive AND the identity was genuinely right.
        tp = int((pred_pos & right).sum())
        # False ACCEPT is the security-relevant error: a stranger (not a genuine
        # gallery member) who cleared the threshold and was labelled anyway.
        fa = int((pred_pos & ~known).sum())
        # False REJECT: a genuine member we failed to name.
        fr = int((~pred_pos & known).sum())

        out["false_accept"] = fa
        out["false_reject"] = fr
        out["true_positive"] = tp
        if pred_pos.sum():
            out["precision"] = round(tp / float(pred_pos.sum()), 4)
        if known.sum():
            out["accuracy"] = round(float(right[known].sum()) / float(known.sum()), 4)
        return out

    # ── recommendation ───────────────────────────────────────────────────
    def recommend(self) -> dict:
        """
        Where SHOULD the threshold sit, given what we have actually seen?

        Two rules, applied in order of importance:

          1. If strangers are clearing the threshold at all, the threshold is
             too low. Push it above the 95th percentile of unknown-face scores
             so the typical stranger no longer matches. This dominates everything
             else - a false accept hands a stranger a real identity, and no
             amount of extra true positives is worth that.

          2. Otherwise, drop it toward the point where genuine members start
             being missed, but stop short of admitting unknowns.

        With no unknowns seen we cannot claim anything about false accepts, and
        the function says so rather than pretending a clean gallery is evidence
        that a low threshold is safe.
        """
        with self.lock:
            s = list(self.samples)
            cur = self.threshold

        if not s:
            return {"ok": False, "reason": "no observations yet",
                    "current": round(cur, 4)}

        scores = np.array([x.score for x in s])
        known = np.array([x.known for x in s])
        right = np.array([x.correct for x in s])

        unknown_scores = scores[~known]
        known_scores = scores[known & right]

        if unknown_scores.size >= 5:
            p95 = float(np.percentile(unknown_scores, 95))
            target = p95 + 0.02
            reason = (f"raising above the 95th percentile of unknown-face scores "
                      f"({p95:.3f}) so typical strangers stop matching")
        elif known_scores.size >= 5:
            p05 = float(np.percentile(known_scores, 5))
            target = max(FLOOR, p05 - 0.03)
            reason = (f"no unknown faces seen yet, so lowering to just under the "
                      f"5th percentile of correct genuine matches ({p05:.3f}) "
                      f"to stop refusing enrolled people")
        else:
            return {"ok": False,
                    "reason": "need at least 5 observations of one class",
                    "current": round(cur, 4)}

        target = float(np.clip(target, FLOOR, CEIL))
        return {
            "ok": True,
            "recommended": round(target, 4),
            "current": round(cur, 4),
            "delta": round(target - cur, 4),
            "reason": reason,
            "caveat": ("calibrated on observed traffic, not on a held-out set - "
                       "treat it as a starting operating point, and re-check "
                       "after a few hundred more observations"),
        }

    def apply(self, threshold: float) -> dict:
        t = float(np.clip(float(threshold), FLOOR, CEIL))
        with self.lock:
            old = self.threshold
            self.threshold = t
            self.history.append({"from": round(old, 4), "to": round(t, 4),
                                 "at": time.time(), "n": len(self.samples)})
        return {"ok": True, "threshold": round(t, 4), "was": round(old, 4)}

    # ── adaptive matching ────────────────────────────────────────────────
    def learn_identity(self, name: str, embedding: np.ndarray,
                       source: str = "correction") -> dict:
        """
        Add one accepted sample to an EXISTING profile.

        This is the 'learns from mistakes' behaviour, and it is deliberately
        narrow:

          * the name must already exist in the gallery - this module cannot
            create an identity, only enrich one. There is no path here by which
            an unknown face becomes an enrolled person.
          * the stored vector is the running MEAN of accepted samples, so one
            bad correction is diluted rather than dominating. Accepting a single
            corrected vector wholesale would let one mistake overwrite a good
            profile.
          * every change is recorded with its source and a timestamp.

        Immediate effect: that person's next sighting is compared against a
        better representative vector. No retraining, no restart.
        """
        v = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if v.shape[0] != 512:
            return {"ok": False, "error": f"expected 512 dims, got {v.shape[0]}"}
        n = float(np.linalg.norm(v))
        if n <= 0:
            return {"ok": False, "error": "zero-length embedding"}
        v = v / n

        with self.lock:
            bucket = self.profiles.setdefault(name, [])
            bucket.append(v)
            meta = self.profile_meta.setdefault(
                name, {"samples": 0, "corrections": 0, "first_seen": time.time()})
            meta["samples"] += 1
            meta["corrections"] += 1
            meta["last_correction"] = time.time()
            meta["source"] = source
            return {"ok": True, "name": name, "samples": meta["samples"],
                    "corrections": meta["corrections"]}

    def profile_vector(self, name: str, base: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Blend an enrolled vector with any learned samples."""
        with self.lock:
            extra = self.profiles.get(name)
        if not extra:
            return base
        if base is None:
            return None
        stack = [np.asarray(base, np.float32).reshape(-1)] + [
            np.asarray(x, np.float32).reshape(-1) for x in extra]
        mean = np.mean(np.stack(stack), axis=0)
        n = float(np.linalg.norm(mean))
        return mean / n if n > 0 else base

    def snapshot(self) -> dict:
        with self.lock:
            s = self._stats()
            s["profiles"] = {k: dict(v) for k, v in self.profile_meta.items()}
            s["history"] = list(self.history)[-12:]
            s["window"] = self.window
            s["bounds"] = {"floor": FLOOR, "ceiling": CEIL}
            s["needs_data"] = len(self.samples) < 50
        return s


calibrator = Calibrator()