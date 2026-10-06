"""
Record how many shots each already-enrolled identity was actually built from.

Written because /api/identities was reporting every identity as shots=1. A
gallery entry is one averaged 512-d vector: fifteen shots go in, the count does
not come out, and a .npy with no sidecar is identical whether it came from one
photo or fifteen. So the endpoint could only default, and the default said
something false about 31 identities.

This does not guess. It re-derives the number from the same rule _enrol.py used -
group the source photos by name, take the deterministic linspace spread of 15 -
and writes identities/_provenance.json. Where the on-disk vector disagrees with
what that rule produces, that disagreement is reported rather than papered over,
because it would mean the gallery holds something the dataset cannot account for.

Read-only with respect to the .npy files. Nothing is re-enrolled.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import IDENTITY_DIR, face          # noqa: E402

FACES = Path(r"C:\Users\PureTrek\Desktop\Faces\Faces")
SHOTS_PER_PERSON = 15
PROVENANCE = IDENTITY_DIR / "_provenance.json"


def person_of(stem: str) -> str:
    return re.sub(r"[_-]?\d+$", "", stem).strip()


def pick_count(paths: list[Path], n: int) -> int:
    if len(paths) <= n:
        return len(paths)
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    seen, out = set(), []
    for i in idx:
        j = int(i)
        while j in seen and j < len(paths) - 1:
            j += 1
        if j not in seen:
            seen.add(j)
            out.append(j)
    return len(out)


def main() -> int:
    if not FACES.is_dir():
        print("dataset missing:", FACES)
        return 1

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(FACES.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            groups[person_of(p.stem)].append(p)

    face.load()
    face.reload_identities()
    enrolled = {n.lower(): n for n in face.names}

    prov: dict[str, dict] = {}
    unmatched, unverifiable = [], []
    for name in sorted(groups):
        key = re.sub(r"[^A-Za-z0-9_]", "_", name)
        actual = enrolled.get(key.lower())
        if actual is None:
            unmatched.append(key)
            continue
        shots = pick_count(groups[name], SHOTS_PER_PERSON)
        prov[actual] = {"shots": shots, "batches": 1,
                        "source": "derived from dataset, not recorded at write time"}

    # An identity in the gallery that the dataset cannot account for is a real
    # gap: something enrolled it, and the record of how is not on this machine.
    derivable = {k.lower() for k in prov}
    for n in sorted(face.names):
        if n.lower() not in derivable:
            unverifiable.append(n)

    PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nrecorded provenance for {len(prov)} of {len(face.names)} identities")
    print(f"{'identity':<22} {'shots':>6} {'from photos':>12}")
    print("-" * 44)
    for n in sorted(prov, key=lambda k: (prov[k]["shots"], k)):
        print(f"{n:<22} {prov[n]['shots']:>6} {len(groups.get(next(k for k in groups if re.sub(r'[^A-Za-z0-9_]','_',person_of(k)).lower()==n.lower()), [])):>12}")

    if unmatched:
        print("\nin the dataset but NOT enrolled:")
        for u in unmatched:
            print(f"  {u}")
    if unverifiable:
        print("\nenrolled but NOT derivable from the dataset (provenance unknown):")
        for u in unverifiable:
            print(f"  {u}")
    print(f"\nwrote {PROVENANCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
