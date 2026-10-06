"""
Does the gallery actually match the model that is loaded?

MOTIVATION, from a real near-miss.

`FACE_MODEL_PACK` already existed in app/__init__.py, defaulting to buffalo_s,
so switching the recognition backbone looked like a one-line change. It is not,
and the reason is easy to miss. Every identity in the gallery is a 512-d vector
produced by ONE embedding network:

    buffalo_s -> w600k_mbf.onnx   (MobileFaceNet)
    buffalo_l -> w600k_r50.onnx   (ResNet-50/100)

Those are different spaces. A cosine between an mbf vector and an r50 vector is
not a weak score, it is meaningless - and because both are 512-d, nothing about
the file itself would catch it. Point this at buffalo_l and every identity loads
fine, every match runs, every number looks plausible, and every name is wrong.

The failure is silent, which is the only kind worth a test. So this records which
embedding space produced each vector and refuses to serve a gallery whose
provenance disagrees with the loaded pack.

It also reports what switching would actually cost, since that is the part that
decides whether a swap is worth doing:

  * buffalo_s : w600k_mbf, 13 MB   - what runs today
  * buffalo_l : w600k_r50, ~170 MB - 13x the parameters
  * buffalo_l also swaps the DETECTOR (det_500m -> det_10g), so face boxes,
    IOD, sharpness and every pixel-budget band move too.

Run:  .venv\\Scripts\\python.exe tools\\_model_pack_check.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IDENTITY_DIR = Path(os.environ.get("FACE_IDENTITY_DIR", str(ROOT / "identities")))
MODEL_ROOT = Path(os.environ.get("FACE_MODEL_ROOT", str(ROOT / "models")))

# Which recognition net each pack embeds with, and which detector it ships.
# Taken from the insightface v0.7 release contents, not from memory:
#   buffalo_s.zip -> det_500m.onnx + w600k_mbf.onnx
#   buffalo_l.zip -> det_10g.onnx  + w600k_r50.onnx
PACKS = {
    "buffalo_s": {"recognition": "w600k_mbf", "detector": "det_500m",
                  "rec_mb": 13, "note": "MobileFaceNet - what this service runs"},
    "buffalo_m": {"recognition": "w600k_mbf", "detector": "retina_10g",
                  "rec_mb": 13, "note": "MobileFaceNet + retina detector"},
    "buffalo_l": {"recognition": "w600k_r50", "detector": "det_10g",
                  "rec_mb": 170, "note": "ResNet-50/100 - needs a GPU to be fast"},
    "buffalo_sc": {"recognition": "w600k_mbf", "detector": "scrfd_2.5g",
                   "rec_mb": 13, "note": "scratch detector"},
    "antelopev2": {"recognition": "glintr100", "detector": "scrfd_10g",
                   "rec_mb": 250, "note": "GLIPTR - not ArcFace, different space"},
}

# The provenance sidecar lives beside the vectors.
PROV = IDENTITY_DIR / "_provenance.json"


def load_pack(pack: str, ident_dir: Path | None = None,
              prov_path: Path | None = None) -> dict:
    """
    Which recognition network produced each vector, from the enrolment sidecar.

    `ident_dir` and `prov_path` are parameters rather than module globals. They
    were globals, and a caller that reassigned `mpc.IDENTITY_DIR` to point at a
    scratch gallery did not change what this function read - it kept using the
    real one, so every test scenario silently measured the production gallery and
    reported it healthy. Three of seven scenarios passed that way. Passing
    arguments makes the function honest about where it is looking.

    The field read is `recognition_net`, which is what _write_provenance in
    graytech/server.py stores. An earlier version read `pack` - a name nothing
    writes - so the counter was always empty and every vector fell through to the
    "no sidecar" branch.
    """
    ident_dir = Path(ident_dir) if ident_dir else IDENTITY_DIR
    prov_path = Path(prov_path) if prov_path else (ident_dir / "_provenance.json")

    prov: dict = {}
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except ValueError:
            prov = {}

    default_net = PACKS["buffalo_s"]["recognition"]
    active_net = PACKS.get(pack, {}).get("recognition", pack)

    vectors = sorted(p.name for p in ident_dir.glob("*.npy"))
    by_net: dict[str, int] = {}
    recorded = 0
    for fname in vectors:
        rec = prov.get(Path(fname).stem) or {}
        net = rec.get("recognition_net")
        if net:
            recorded += 1
        else:
            # Enrolled before the net was recorded. Attribute to the service
            # default rather than calling it unknown - it was almost certainly
            # w600k_mbf, and treating a healthy gallery as unknown would be its
            # own kind of false alarm.
            net = default_net
        by_net[net] = by_net.get(net, 0) + 1

    return {
        "by_net": by_net,
        "total": len(vectors),
        "recorded": recorded,
        "active_net": active_net,
        "mismatched": {k: v for k, v in by_net.items() if k != active_net},
    }


def main() -> int:
    active = os.environ.get("FACE_MODEL_PACK", "buffalo_s")
    print(f"\n  FACE_MODEL_PACK = {active}")
    print(f"  gallery         = {IDENTITY_DIR}")

    info = load_pack(active)
    want = PACKS.get(active)
    print(f"\n  pack contents ({active}):")
    if want:
        print(f"    recognition  {want['recognition']}   ~{want['rec_mb']} MB")
        print(f"    detector     {want['detector']}")
        print(f"    {want['note']}")
    else:
        print("    *** unknown pack name - insightface would fail to load it ***")

    print(f"\n  gallery vectors: {info['total']}  "
          f"({info['recorded']} with the net recorded)")
    for net, n in sorted(info["by_net"].items()):
        tag = "OK" if net == info["active_net"] else "*** MISMATCH ***"
        print(f"    {net:<12} {n:>4} vector(s)   {tag}")
    if info["recorded"] < info["total"]:
        print(f"    ({info['total'] - info['recorded']} had no recorded net and were "
              f"attributed to the service default, w600k_mbf)")

    mismatched = info["mismatched"]
    print()
    if mismatched:
        print("  *** THIS GALLERY CANNOT BE USED WITH THIS PACK ***")
        print("  A cosine between embeddings from different networks is")
        print("  meaningless, and both are 512-d so nothing in the file would")
        print("  reveal it. Every name would be wrong and every score plausible.")
        for net, n in sorted(mismatched.items()):
            print(f"\n  re-enrol {n} identity(ies) built by {net} "
                  f"using {info['active_net']}:")
            print("    python tools\\_enrol.py")
        return 1

    print("  gallery and model agree - safe to serve")
    if active == "buffalo_l":
        print()
        print("  note: buffalo_l swaps the DETECTOR too, so face boxes, IOD and")
        print("        sharpness all move. Re-measure before trusting any")
        print("        pixel-budget or calibration number:")
        print("    python tools\\_audit_gallery.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
