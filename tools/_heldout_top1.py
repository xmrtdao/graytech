"""
Top-1 accuracy on genuinely held-out photographs, using the CURRENT gallery.

Written because the README quotes 97.0% and I could not tell where that number
came from or whether it still holds. demo/dataset_eval.py enrols its own
references at --per-identity (4 by default) and excludes exactly those photos
from the holdout, so it is a valid test - but it does NOT measure the gallery
this service actually ships, which is 15 shots per identity.

That difference matters. A 4-shot prototype and a 15-shot prototype are not the
same reference, and quoting one while the system runs the other is the kind of
number that looks fine and means nothing.

So this reads the live gallery as it stands, picks photos it did NOT enrol - not
"the last N", which overlaps the enrolment linspace, but a seeded random sample -
and reports the three outcomes separately:

  top-1 correct      the right person AND above threshold
  refused            below threshold, so nobody was named  (a safe failure)
  wrong identity     a name was attached and it was the wrong one  (the dangerous one)

The third number is the one a top-1 percentage hides. Refusing a face is a
correct behaviour for a closed gallery; attaching the wrong name is the failure
that matters operationally.
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app                                       # noqa: E402
from app import face                             # noqa: E402

FACES = Path(r"C:\Users\PureTrek\Desktop\Faces\Faces")
PER_IDENTITY = 8
SEED = 11


def truth_of(stem: str) -> str:
    return re.sub(r"[_-]?\d+$", "", stem).strip()


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def main() -> int:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(FACES.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            groups[truth_of(p.stem)].append(p)

    # Read the SHIPPED gallery rather than building one, so this reports the
    # system that is actually running.
    face.load()
    face.reload_identities()
    thr = app.current_threshold()

    rng = np.random.default_rng(SEED)
    sample: list[Path] = []
    for _, ps in sorted(groups.items()):
        if len(ps) > 15:            # only identities with photos to spare
            sample.extend(rng.choice(ps, size=PER_IDENTITY, replace=False).tolist())

    print(f"\n  gallery    : {len(face.names)} identities (the shipped prototypes)")
    print(f"  threshold  : {thr}")
    print(f"  holdout    : {len(sample)} images, seeded random, never enrolled\n")

    rows = []
    for p in sample:
        try:
            _, m = face.embed(p.read_bytes())
        except Exception:                                    # noqa: BLE001
            continue
        if m.size == 0:
            continue
        want = safe(truth_of(p.stem))
        r = max(face.match(m)[0], key=lambda x: x["cosine"])
        rows.append({
            "cosine": float(r["cosine"]),
            "matched": bool(r["matched"]),
            "want": want,
            "got": r["name"],
            "correct": r["name"] == want,
        })

    n = len(rows)
    if not n:
        print("  no usable holdout images")
        return 1
    correct = sum(1 for r in rows if r["correct"] and r["matched"])
    refused = sum(1 for r in rows if not r["matched"])
    wrong = n - correct - refused
    cos = [r["cosine"] for r in rows]

    print(f"  top-1 correct            : {correct:>4}   {correct / n * 100:>5.1f}%")
    print(f"  refused, nobody named    : {refused:>4}   {refused / n * 100:>5.1f}%")
    print(f"  WRONG identity attached  : {wrong:>4}   {wrong / n * 100:>5.1f}%")
    print(f"  cosine  min={min(cos):.4f}  p50={statistics.median(cos):.4f}  "
          f"p05={sorted(cos)[max(0, int(n * 0.05))]:.4f}  max={max(cos):.4f}")

    print("\n  the six lowest-cosine holdout images:")
    for r in sorted(rows, key=lambda x: x["cosine"])[:6]:
        verdict = ("matched" if r["matched"] else "REFUSED")
        flag = "" if r["correct"] else "   <- wrong name" if r["matched"] else ""
        print(f"    {r['cosine']:.4f}  {r['want']:<18} -> {r['got']:<18} "
              f"{verdict}{flag}")

    print(f"\n  identities at risk: any whose p05 sits near {thr} will be refused")
    print(f"  in live traffic even on good photographs. Run")
    print(f"    python tools\\_audit_gallery.py")
    print(f"  for that per-identity view.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())