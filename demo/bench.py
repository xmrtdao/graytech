"""
Measure throughput properly.

WHY THIS FILE EXISTS: the first full run reported "10.2 fps" and "61285 ms
median" in the same breath. Those contradict each other - 1156 images in 113
seconds is 10.2 fps, but 61 seconds per image would be 2,562 images in 19
hours. The wall clock was right and the per-image figure was wrong.

The bug: in demo/dataset_eval.py the per-image timer was

    t0 = time.perf_counter()          # set ONCE, before the loop
    for p in sample:
        ...
        ms = (time.perf_counter() - t0) * 1000

so every sample recorded the CUMULATIVE elapsed time since the loop started
rather than that one image. The median of that is just "half the total run",
which is why it came out at 61 seconds. The accuracy numbers are unaffected -
they were computed from the detections themselves - but the latency line was
meaningless and should never have been printed as if it were real.

This script measures each image independently, and separates decode from
inference, because "how fast can it detect+embed" and "how fast can it pull
bytes off disk" are different questions and the first full run conflated them.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description="honest throughput measurement")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"  no images in {folder}", file=sys.stderr)
        return 1
    sample = files[: args.count + args.warmup]
    print(f"\n  {len(sample)} images sampled from {folder}")
    print(f"  model: loading...")

    from app import face
    t0 = time.perf_counter()
    face.load()
    face.reload_identities()
    print(f"  model loaded in {time.perf_counter()-t0:.1f}s   "
          f"{len(face.names)} identities already enrolled")

    # Warm up: first inference pays lazy allocation inside onnxruntime.
    for p in sample[: args.warmup]:
        try:
            face.embed(p.read_bytes())
        except Exception:                                       # noqa: BLE001
            pass

    decode_ms, infer_ms, total_ms = [], [], []
    faces_seen = 0

    for p in sample[args.warmup:]:
        t0 = time.perf_counter()
        data = p.read_bytes()
        t1 = time.perf_counter()
        try:
            _, matrix = face.embed(data)
        except Exception:                                       # noqa: BLE001
            continue
        t2 = time.perf_counter()

        decode_ms.append((t1 - t0) * 1000)
        infer_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t0) * 1000)
        faces_seen += int(matrix.shape[0] > 0)

    n = len(total_ms)
    if n == 0:
        print("  no successful inferences", file=sys.stderr)
        return 1

    wall_start = time.perf_counter()
    for p in sample[args.warmup:]:
        try:
            face.embed(p.read_bytes())
        except Exception:                                       # noqa: BLE001
            pass
    wall = time.perf_counter() - wall_start

    print(f"\n  n = {n} images ({faces_seen} had at least one face)")
    print(f"  sustained over a full pass: {n/wall:.1f} fps")
    print()
    print(f"  {'stage':<26}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}")
    for label, s in (("read from disk", decode_ms),
                     ("detect + embed", infer_ms),
                     ("total per image", total_ms)):
        print(f"  {label:<26}{statistics.median(s):>9.1f}ms{statistics.mean(s):>9.1f}ms"
              f"{min(s):>9.1f}ms{max(s):>9.1f}ms")

    print()
    print(f"  brief's CPU target: ~17 fps")
    print(f"  measured           : {n/wall:.1f} fps  "
          f"({'MEETS' if n/wall >= 17 else 'BELOW'} the target)")
    if faces_seen:
        print(f"  faces found        : {faces_seen}/{n} images")
    print()
    print("  Single process, single thread. This is the CPU path on an Intel")
    print("  UHD 620 machine with no CUDA - see README, 'Hardware this was built")
    print("  and measured on'. It is not the GPU configuration the brief headslined.")
    return 0


if __name__ == "__main__":
    sys.exit(main())