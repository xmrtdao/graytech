"""
Enrol a face dataset and report REAL accuracy and throughput.

The dataset filenames carry the identity: "Akshay Kumar_0.jpg". That gives us
ground truth for free, so this can measure whether recognition actually works
rather than just how fast the pipeline runs.

Two phases, deliberately separate:

  1. enrol   - one embedding per identity, AVERAGED over N photos of that
               person. Averaging multiple shots is the standard trick: it
               averages out pose and lighting, so the stored vector is
               representative rather than one arbitrary frame. We do not enrol
               all 2562 photos; a few per person is both faster and better.

  2. verify  - re-detect held-out images, match against the index, and compare
               the predicted name against the filename. That is a real top-1
               accuracy number, and the photos used for enrolment are EXCLUDED
               so we are not grading our own homework.

Measurement note, learned the hard way:

The first version of this file hoisted `t0 = time.perf_counter()` above the
verify loop, so every sample recorded the CUMULATIVE elapsed time since the run
started rather than that one image. The reported "median per-image latency" was
therefore just half the total wall clock - 61,285 ms - which contradicted the
10.2 fps measured over the very same pass. Accuracy was never affected, because
it is computed from the detections rather than from the timer. The timer is now
per-image and inside the loop, where it belongs.

If the two numbers ever disagree again, trust the wall clock: it is the one
counted from an unambiguous pair of clock reads.
"""

import argparse
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def identity_from_name(stem: str) -> str:
    """'Akshay Kumar_0' -> 'Akshay Kumar'. The trailing _N is the sample index."""
    return re.sub(r"_\d+$", "", stem).strip()


def safe_name(ident: str) -> str:
    """Filesystem-safe identity key, so 'Akshay Kumar' and 'a/b' cannot collide."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", ident)


def load_images(folder: Path, limit: int = 0):
    by_identity: dict[str, list[Path]] = defaultdict(list)
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if limit:
        files = files[:limit]
    for p in files:
        by_identity[identity_from_name(p.stem)].append(p)
    return by_identity


def main() -> int:
    ap = argparse.ArgumentParser(description="enrol + verify on a labelled face dataset")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--per-identity", type=int, default=4,
                    help="photos per person used for enrolment (also excluded from verify)")
    ap.add_argument("--limit", type=int, default=0, help="cap total images scanned")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the cosine match threshold for this run")
    ap.add_argument("--reset", action="store_true", help="clear identities/ first")
    ap.add_argument("--verify-limit", type=int, default=1200)
    args = ap.parse_args()

    import numpy as np
    import app as appmod

    if args.threshold is not None:
        appmod.MATCH_THRESHOLD = args.threshold

    from app import face, IDENTITY_DIR

    if args.reset and IDENTITY_DIR.exists():
        for f in IDENTITY_DIR.glob("*.npy"):
            f.unlink()
        print(f"  cleared identities from {IDENTITY_DIR}")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"  folder not found: {folder}", file=sys.stderr)
        return 1

    by_identity = load_images(folder, args.limit)
    total_images = sum(len(v) for v in by_identity.values())
    if not by_identity:
        print(f"  no images in {folder}", file=sys.stderr)
        return 1
    counts = sorted(len(v) for v in by_identity.values())

    print(f"\n  folder    : {folder}")
    print(f"  images    : {total_images}")
    print(f"  identities: {len(by_identity)}")
    print(f"  per person: min {counts[0]}  median {statistics.median(counts)}  max {counts[-1]}")

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    face.load()
    face.reload_identities()
    print(f"\n  model loaded in {face.load_ms:.0f} ms   threshold {appmod.MATCH_THRESHOLD}")

    # ── Phase 1: enrol, averaging embeddings per identity ──
    print(f"\n  --- enrol ({args.per_identity} photo(s)/person, embeddings averaged) ---")
    t_enrol = time.perf_counter()
    enrolled = no_face = bad = 0
    for ident, paths in sorted(by_identity.items()):
        vecs = []
        for p in paths[: args.per_identity]:
            try:
                _, matrix = face.embed(p.read_bytes())
            except Exception:                                  # noqa: BLE001
                bad += 1
                continue
            if matrix.size == 0:
                no_face += 1
                continue
            # largest face in frame: for a portrait that is the subject
            vecs.append(matrix[0])
        if not vecs:
            continue
        mean = np.mean(np.stack(vecs), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        np.save(IDENTITY_DIR / f"{safe_name(ident)}.npy", mean.astype(np.float32))
        enrolled += 1
        if enrolled % 25 == 0:
            print(f"    {enrolled}/{len(by_identity)}")

    enrol_s = time.perf_counter() - t_enrol
    print(f"  enrolled {enrolled}/{len(by_identity)} identities in {enrol_s:.1f}s "
          f"({enrolled*args.per_identity/max(enrol_s,1e-9)*60:.0f} images/min)")
    if no_face:
        print(f"  WARNING: {no_face} enrolment photo(s) had no detectable face")
    if bad:
        print(f"  WARNING: {bad} enrolment photo(s) failed to decode")
    face.reload_identities()
    print(f"  index now holds {len(face.names)} identities")

    # ── Phase 2: verify on held-out images ──
    print(f"\n  --- verify (top-1 vs filename ground truth) ---")
    per_id = max(1, args.verify_limit // max(1, len(by_identity)))
    sample: list[Path] = []
    for ident, paths in sorted(by_identity.items()):
        sample.extend(paths[args.per_identity:][:per_id])   # exclude enrolment photos
    sample = sample[: args.verify_limit]
    print(f"  {len(sample)} holdout image(s), enrolment photos excluded")

    top1 = top1_detected = detected = 0
    noface = 0
    failures: list[tuple] = []
    infer_ms: list[float] = []
    t_wall = time.perf_counter()
    for p in sample:
        truth = identity_from_name(p.stem)
        # Per-image timer, inside the loop. See the module docstring: this was
        # once hoisted above the loop and reported cumulative time as if it were
        # per-image latency.
        t_img = time.perf_counter()
        try:
            _, matrix = face.embed(p.read_bytes())
        except Exception:                                      # noqa: BLE001
            continue
        infer_ms.append((time.perf_counter() - t_img) * 1000)
        if matrix.size == 0:
            noface += 1
            continue
        detected += 1
        m = face.match(matrix, k=1)
        if m and m[0] and m[0][0]["matched"]:
            top1_detected += 1
            if m[0][0]["name"] == safe_name(truth):
                top1 += 1
            elif len(failures) < 8:
                failures.append((p.name, truth, m[0][0]["name"], m[0][0]["cosine"]))
    wall = time.perf_counter() - t_wall

    print(f"\n  images evaluated     : {len(infer_ms)}")
    print(f"  faces detected       : {detected}  ({detected/max(1,len(infer_ms))*100:.1f}%)")
    print(f"  no face detected     : {noface}")
    print(f"  matched above thresh : {top1_detected}")
    print(f"  top-1 CORRECT        : {top1}")
    print(f"  top-1 / detected     : {top1/max(1,detected)*100:.1f}%")
    print(f"  top-1 / all images   : {top1/max(1,len(infer_ms))*100:.1f}%")
    print(f"\n  wall clock           : {wall:.2f}s  ->  {len(infer_ms)/wall:.1f} fps end-to-end")
    if infer_ms:
        print(f"  per-image median     : {statistics.median(infer_ms):.1f} ms")
        print(f"  per-image mean       : {statistics.mean(infer_ms):.1f} ms")
        print(f"  brief's CPU target   : ~17 fps  ->  "
              f"{'MEETS' if len(infer_ms)/wall >= 17 else 'BELOW'} it")

    if failures:
        print("\n  sample misidentifications (file, truth, predicted, cosine):")
        for f in failures:
            print(f"    {f[0]:<30} truth={f[1]:<20} got={f[2]:<20} {f[3]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())