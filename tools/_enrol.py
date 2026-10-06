"""
Multi-shot enrolment for the Faces gallery.

The rig spec is explicit about this: "Enroll 5-15 shots: frontal, +/-30 yaw,
slight down-pitch, sunglasses on/off. One LinkedIn headshot will fail from the
air." Every identity currently has exactly ONE embedding built from one photo.
This rebuilds each one by embedding a spread of shots and averaging them into a
single L2-normalised reference vector.

Averaging is the right operation for a gallery reference: it is the centroid of
that person's embedding cloud, which is more stable than any single sample and
does not need labels or quality filtering to be useful. Nothing about an
identity is invented here - every vector comes from a photo already on disk
under that person's own name.

Writes .npy into identities/. Existing single-shot vectors are kept as .npy.1
so nothing is destroyed.
"""
from __future__ import annotations

import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import face  # noqa: E402
from app import _l2  # noqa: E402

FACES = Path(r"C:\Users\PureTrek\Desktop\Faces\Faces")
IDENT = Path(__file__).resolve().parent.parent / "identities"
SHOTS_PER_PERSON = 15          # the spec's upper bound
SEED = 7                        # deterministic spread, so re-runs are stable


def person_of(stem: str) -> str:
    """'Hugh Jackman_12' -> 'Hugh Jackman'. Handles the hyphen/underscore mix."""
    return re.sub(r"[_-]?\d+$", "", stem).strip()


def pick_shots(paths: list[Path], n: int) -> list[Path]:
    """
    Choose n shots spread across the set rather than the first n.

    Filenames usually run sequentially over a session, so taking a stride keeps
    the variety (framing, angle, expression) instead of 15 near-identical
    frames from the same second.
    """
    if len(paths) <= n:
        return paths
    rng = np.random.default_rng(SEED)
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    # Nudge off exact duplicates, which happen when a set is short.
    seen, out = set(), []
    for i in idx:
        j = int(i)
        while j in seen and j < len(paths) - 1:
            j += 1
        if j not in seen:
            seen.add(j)
            out.append(paths[j])
    return out


def main() -> int:
    if not FACES.is_dir():
        print("dataset missing:", FACES)
        return 1
    IDENT.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(FACES.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            groups[person_of(p.stem)].append(p)

    print(f"{len(groups)} people, {sum(len(v) for v in groups.values())} photos")
    face.load()
    print(f"model={face.app} gallery before={len(face.names)}")

    summary = []
    for name in sorted(groups):
        shots = pick_shots(groups[name], SHOTS_PER_PERSON)
        vecs, used, failed = [], 0, 0
        for p in shots:
            try:
                img = cv2.imread(str(p))
                if img is None:
                    failed += 1
                    continue
                faces, m = face.embed_bytes(img)
                if not faces or m.size == 0:
                    failed += 1
                    continue
                # Largest face wins: in a group shot the subject is the near one.
                big = max(range(len(faces)),
                          key=lambda i: (faces[i]["bbox"][2] - faces[i]["bbox"][0]))
                vecs.append(m[big])
                used += 1
            except Exception as exc:                     # noqa: BLE001
                failed += 1
                print(f"    {p.name}: {type(exc).__name__}")
        if not vecs:
            summary.append((name, 0, 0, failed, None))
            continue

        stack = np.stack(vecs).astype(np.float32)
        # Centroid of the embedding cloud, renormalised. A plain mean then L2 is
        # the reference; median resists one bad frame better, so use it where the
        # spread is wide.
        spread = float(np.mean(np.linalg.norm(stack - stack.mean(axis=0), axis=1)))
        ref = _l2(np.median(stack, axis=0)) if used >= 5 else _l2(stack.mean(axis=0))

        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        dest = IDENT / f"{safe}.npy"
        if dest.exists():
            shutil.copy2(dest, dest.with_suffix(".npy.1"))
        np.save(dest, ref)

        # Report how tight the shots were: a wide spread means the gallery entry
        # is averaging genuinely different views, which is the point.
        summary.append((name, used, len(shots), failed, spread))

    print(f"\n{'identity':26} {'used':>5} {'tried':>6} {'failed':>7} {'spread':>8}")
    for name, used, tried, failed, spread in summary:
        s = f"{spread:.3f}" if spread is not None else "-"
        print(f"{name:26} {used:>5} {tried:>6} {failed:>7} {s:>8}")

    face.reload_identities()
    print(f"\ngallery after = {len(face.names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())