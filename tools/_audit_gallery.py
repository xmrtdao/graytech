"""
What is actually in the gallery, and how well-supported is each identity?

The HUD shows a single number - "33 enrolled" - which reads like a count of
photos. It is a count of PEOPLE. 2,564 photos collapse into 33 identities, and
for 31 of those there are 50-120 photos each to build a reference vector from.
For two of them there is exactly one.

That matters because a prototype built from a single image cannot cover the
spread a webcam introduces - different sensor, different lens, far less face
resolution, different colour temperature, browser JPEG artefacts. The match
threshold is 0.45, so a single-shot identity needs to stay within about 0.5
cosine of its own reference to be named at all. This script measures how much
margin each identity actually has, so a thin one is visible as a thin one rather
than showing up later as "it doesn't recognise me".

The numbers here are deliberately unflattering about our own gallery: it reports
the weakest identities first, because those are the ones that will be refused.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app                                    # noqa: E402
from app import face, _l2                    # noqa: E402

FACES = Path(r"C:\Users\PureTrek\Desktop\Faces\Faces")
IDENT = Path(__file__).resolve().parent.parent / "identities"
SHOTS_PER_PERSON = 15                        # the rig spec's upper bound
TRUTH_DEGRADE = 0.30                         # downscale: ~300px frame, webcam-ish
TRUTH_QUALITY = 72                          # the browser encodes at 0.72


def person_of(stem: str) -> str:
    """'Hugh Jackman_12' -> 'Hugh Jackman'. Strips a trailing numeric index."""
    return re.sub(r"[_-]?\d+$", "", stem).strip()


def pick_shots(paths: list[Path], n: int) -> list[Path]:
    if len(paths) <= n:
        return paths
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    seen, out = set(), []
    for i in idx:
        j = int(i)
        while j in seen and j < len(paths) - 1:
            j += 1
        if j not in seen:
            seen.add(j)
            out.append(paths[j])
    return out


def degrade(img: np.ndarray, scale: float, quality: int) -> np.ndarray:
    """Approximate what the browser camera path hands the server."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(64, int(w * scale)), max(64, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else small


def main() -> int:
    if not FACES.is_dir():
        print("dataset missing:", FACES)
        return 1

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(FACES.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            groups[person_of(p.stem)].append(p)

    total = sum(len(v) for v in groups.values())
    print(f"\n{total} photos -> {len(groups)} identities")
    print(f"gallery on disk: {len(list(IDENT.glob('*.npy')))} .npy files")
    print(f"live threshold:  {app.current_threshold()}\n")

    face.load()
    face.reload_identities()
    names = set(face.names)

    rows = []
    # Gallery filenames are matched case-insensitively. The dataset uses
    # 'joe-lee.jpg' and the enrolled file is 'Joe_Lee.npy', so a case-sensitive
    # check reports an enrolled identity as missing - which is exactly the kind
    # of false negative that sends someone off re-enrolling a face that is
    # already in the gallery.
    lower = {n.lower(): n for n in names}

    for name in sorted(groups):
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        enrolled_as = lower.get(safe.lower())
        if enrolled_as is None:
            rows.append((safe, len(groups[name]), 0, None, "NOT ENROLLED"))
            continue
        shots = pick_shots(groups[name], SHOTS_PER_PERSON)
        if not shots:
            rows.append((safe, len(groups[name]), 0, None, "NO PHOTOS"))
            continue

        # Held-out test image: one this person has plenty of, but which is NOT in
        # the shot set the prototype was built from. This matters more than it
        # looks. pick_shots() takes an linspace across the range, and that range
        # INCLUDES the last photo - so naively testing on the last photo tests
        # against an image the prototype already contains, and the resulting
        # cosine is a self-match wearing the clothes of a generalisation check.
        # Every identity scored 0.82+ that way, which is why an earlier version
        # of this script looked reassuring and was not.
        held = next((p for p in groups[name] if p not in shots), None)
        circular = held is None
        if circular:
            held = groups[name][0]
        img = cv2.imread(str(held))
        if img is None:
            rows.append((enrolled_as, len(groups[name]), len(shots), None, "UNREADABLE"))
            continue
        small = degrade(img, TRUTH_DEGRADE, TRUTH_QUALITY)
        faces_, m = face.embed_bytes(small)
        if not faces_ or m.size == 0:
            rows.append((enrolled_as, len(groups[name]), len(shots), None,
                         "NO FACE AT WEBCAM SIZE"))
            continue
        rows.append((enrolled_as, len(groups[name]), len(shots),
                     float(max(face.match(m)[0], key=lambda r: r["cosine"])["cosine"]),
                     "SELF-MATCH" if circular else "held out"))

    thr = app.current_threshold()
    rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0.0))
    print(f"{'identity':<22} {'photos':>7} {'shots':>6} {'cosine':>8}  {'x thr':>6}  note")
    print("-" * 78)
    for safe, nphotos, shots, cos, note in rows:
        if cos is None:
            print(f"{safe:<22} {nphotos:>7} {shots:>6} {'-':>8}  {'-':>6}  {note}")
            continue
        mult = cos / thr if thr else 0.0
        warn = ""
        if mult < 1.5:
            warn = "  <-- thin margin"
        print(f"{safe:<22} {nphotos:>7} {shots:>6} {cos:>8.4f}  {mult:>6.2f}  {note}{warn}")

    thin = [r for r in rows if r[2] < SHOTS_PER_PERSON]
    print()
    print(f"identities built from fewer than {SHOTS_PER_PERSON} shots: {len(thin)}")
    for safe, nphotos, shots, cos, note in thin:
        print(f"  {safe}: {shots} shot(s) from {nphotos} photo(s)"
              + (f", cosine {cos:.4f}" if cos is not None else f", {note}"))
    print()
    print("A single-shot prototype is the weakest thing in a gallery. The rig spec")
    print("asks for 5-15 shots per person for exactly this reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
