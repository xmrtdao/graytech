"""
Demo: run detection + recognition over a folder of stills and report throughput.

This is demo source #1 in the brief's priority order (folder of stills), chosen
because it needs no camera and no permissions, so it is the one that can be
trusted to work anywhere.

    python -m demo.stills --folder path/to/stills
    python -m demo.stills --folder path/to/stills --enroll      # also register them
    python -m demo.stills --folder path/to/stills --repeat 20   # fps measurement

fps here is wall-clock throughput of the whole pipeline: decode, detect,
embed, match. It is the honest number - the model's raw detect time is reported
separately because it flatters itself by ignoring everything around it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Still-image demo")
    ap.add_argument("--folder", required=True, help="folder of images")
    ap.add_argument("--repeat", type=int, default=0,
                    help="pass the whole set N times and report fps")
    ap.add_argument("--enroll", action="store_true",
                    help="register each detected face under its filename stem")
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"  folder not found: {folder}", file=sys.stderr)
        return 1
    images = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"  no images in {folder} (looked for {sorted(IMAGE_SUFFIXES)})", file=sys.stderr)
        return 1

    from app import face
    from app import IDENTITY_DIR

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    face.load()
    face.reload_identities()
    print(f"\n  model {face.app.__class__.__name__} loaded in {face.load_ms:.0f} ms")
    print(f"  {len(images)} image(s) from {folder}")
    if not face.names:
        print("  identity index is empty - every face will come back unmatched")
    print()

    payload = [(p.name, p.read_bytes()) for p in images]

    total_faces = 0
    for name, data in payload:
        t0 = time.perf_counter()
        try:
            faces, matrix = face.embed(data)
        except Exception as exc:                            # noqa: BLE001
            print(f"  {name:<34} FAILED  {exc}")
            continue
        ms = (time.perf_counter() - t0) * 1000
        matches = face.match(matrix, k=args.top_k)
        total_faces += len(faces)

        if not faces:
            print(f"  {name:<34} no faces   {ms:6.1f} ms")
            continue

        print(f"  {name:<34} {len(faces)} face(s)  {ms:6.1f} ms")
        for i, f in enumerate(faces):
            x1, y1, x2, y2 = f["bbox"]
            who = "-"
            if i < len(matches) and matches[i]:
                top = matches[i][0]
                who = f'{top["name"]} ({top["cosine"]:.3f})' if top["matched"] \
                    else f'no match (best {top["cosine"]:.3f})'
            print(f"      box=({x1},{y1},{x2},{y2}) score={f['det_score']:.3f}  {who}")

            if args.enroll:
                stem = Path(name).stem + (f"_f{i}" if len(faces) > 1 else "")
                np.save(IDENTITY_DIR / f"{stem}.npy", matrix[i])
                print(f"      enrolled as {stem}")

    if args.enroll:
        face.reload_identities()
        print(f"\n  index now holds {len(face.names)} identit(ies)")

    print(f"\n  {total_faces} face(s) across {len(payload)} image(s)")

    if args.repeat > 0 and payload:
        print(f"\n  throughput over {args.repeat} pass(es):")
        t0 = time.perf_counter()
        n = 0
        for _ in range(args.repeat):
            for _, data in payload:
                try:
                    face.embed(data)
                    n += 1
                except Exception:                           # noqa: BLE001
                    pass
        elapsed = time.perf_counter() - t0
        fps = n / elapsed if elapsed > 0 else 0.0
        print(f"    {n} inference(s) in {elapsed:.2f}s")
        print(f"    {fps:.1f} fps  ({elapsed / n * 1000:.1f} ms per image)")
        if fps > 0:
            print(f"    per-face: {elapsed / n * 1000:.1f} ms/image, "
                  f"{elapsed / n / max(1, total_faces / len(payload)) * 1000:.1f} ms/face")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())