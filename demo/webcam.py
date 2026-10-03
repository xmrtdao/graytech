"""
Demo: live webcam, using the SAME camera path as the relay's `vex-vision` tool.

Camera access deliberately goes through vex-vision's mechanism rather than
cv2.VideoCapture. Two reasons:

  * One convention. relay/server.js `toolHandlers['vex-vision']` grabs a frame
    with:
        ffmpeg -f dshow -i video="HP TrueVision HD Camera" \
               -frames:v 1 -q:v 2 -update 1 <out> -y
    pointing at C:\\tools\\ffmpeg and writing relay-data/vex-capture.jpg. If
    this service used a different device string or backend it would be opening
    a second, competing handle on the same webcam - which on Windows frequently
    fails outright rather than sharing.

  * cv2.VideoCapture on Windows opens its own capture graph and ignores
    DirectShow device naming, so asking for "HP TrueVision HD Camera" there
    means nothing. DirectShow is the reliable path on this box.

Priority order in the brief is folder-of-stills then webcam then recorded clip
then drone. The folder demo (demo/stills.py) is source #1 and needs no camera;
this is source #2 and proves the live path.

    python -m demo.webcam --count 30 --fps
    python -m demo.webcam --count 1 --out frame.jpg --enroll
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Mirrors relay/server.js toolHandlers['vex-vision'] exactly. If you change one,
# change this - a second device string means a second handle on the webcam.
FFMPEG = r"C:\tools\ffmpeg\ffmpeg.exe"
CAMERA_NAME = "HP TrueVision HD Camera"


def grab_frame(ffmpeg: str, camera: str, dest: Path) -> bytes:
    """One frame from the webcam, via DirectShow, the way vex-vision does it."""
    if not Path(ffmpeg).exists():
        raise FileNotFoundError(
            f"ffmpeg not found at {ffmpeg} - this is the same path relay's "
            f"vex-vision uses; override with --ffmpeg"
        )
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "dshow", "-i", f'video={camera}',
        "-frames:v", "1", "-q:v", "2", "-update", "1",
        str(dest), "-y",
    ]
    # Each call opens and closes the device. That costs ~200-400ms on Windows,
    # which is slower than a held-open stream - so this measures PIPELINE cost,
    # not model cost. See --hold for a persistent stream.
    subprocess.run(cmd, check=True, capture_output=True, timeout=15)
    return dest.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description="Webcam demo")
    ap.add_argument("--count", type=int, default=30, help="frames to process")
    ap.add_argument("--camera", default=CAMERA_NAME)
    ap.add_argument("--ffmpeg", default=FFMPEG)
    ap.add_argument("--out", help="save one frame to this path")
    ap.add_argument("--enroll", action="store_true", help="register each face")
    ap.add_argument("--interval", type=float, default=0.0, help="seconds between grabs")
    ap.add_argument("--fps", action="store_true", help="report throughput")
    args = ap.parse_args()

    from app import face, IDENTITY_DIR

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    face.load()
    face.reload_identities()
    print(f"\n  model loaded in {face.load_ms:.0f} ms")
    print(f"  camera: {args.camera}")
    print(f"  ffmpeg: {args.ffmpeg}")
    if not face.names:
        print("  identity index is empty - faces will come back unmatched")
    print()

    import numpy as np

    total_faces = 0
    infer_ms: list[float] = []
    wall_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        frame_path = Path(tmp) / "frame.jpg"
        for i in range(args.count):
            try:
                data = grab_frame(args.ffmpeg, args.camera, frame_path)
            except FileNotFoundError as exc:
                print(f"  {exc}", file=sys.stderr)
                return 1
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.decode(errors="replace").strip()[:200]
                print(f"  frame {i+1}/{args.count}: capture failed: {detail}", file=sys.stderr)
                return 1

            t0 = time.perf_counter()
            try:
                faces, matrix = face.embed(data)
            except Exception as exc:                        # noqa: BLE001
                print(f"  frame {i+1}/{args.count}: inference failed: {exc}", file=sys.stderr)
                continue
            ms = (time.perf_counter() - t0) * 1000
            infer_ms.append(ms)
            total_faces += len(faces)

            matches = face.match(matrix, k=1)
            if not faces:
                print(f"  frame {i+1}/{args.count}: no faces      {ms:7.1f} ms")
            else:
                bits = []
                for fi in range(len(faces)):
                    top = matches[fi][0] if matches[fi] else None
                    bits.append(f'{top["name"]}({top["cosine"]:.2f})' if top and top["matched"]
                                else (f'unknown({top["cosine"]:.2f})' if top else 'unknown'))
                print(f"  frame {i+1}/{args.count}: {len(faces)} face(s)  "
                      f"{', '.join(bits)}   {ms:7.1f} ms")

            if args.enroll:
                for fi in range(len(faces)):
                    name = f"webcam_{int(time.time())}_{fi}"
                    np.save(IDENTITY_DIR / f"{name}.npy", matrix[fi])
                    print(f"      enrolled as {name}")
                if faces:
                    face.reload_identities()

            if args.out and i == 0:
                Path(args.out).write_bytes(data)
                print(f"      wrote {args.out}")

            if args.interval:
                time.sleep(args.interval)

    wall = time.perf_counter() - wall_start
    print(f"\n  {total_faces} face(s) over {len(infer_ms)} frame(s)")
    if infer_ms:
        print(f"  inference: mean {sum(infer_ms)/len(infer_ms):.1f} ms  "
              f"min {min(infer_ms):.1f}  max {max(infer_ms):.1f}")
    print(f"  end-to-end: {wall:.2f}s for {args.count} frames "
          f"({args.count/wall:.1f} fps, includes reopening the device each time)")

    if args.fps and infer_ms:
        mean = sum(infer_ms) / len(infer_ms)
        print(f"  model-only ceiling: {1000/mean:.1f} fps "
              f"(the number a held-open camera stream would approach)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())