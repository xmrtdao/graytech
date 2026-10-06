"""
Does the pack check actually fail when the gallery disagrees with the model?

Written because the first version of _model_pack_check.py passed against a
gallery deliberately marked as half-built-from-r50. It read a `pack` field that
nothing ever writes, so its counter was always empty, every vector fell into the
"no sidecar" branch, and it reported a healthy gallery whatever the sidecar
actually said.

That is the same failure the repo has hit twice already - a check that cannot
fail is worse than no check - so this proves the current one can, by writing
sidecars in three states and demanding the right verdict for each. If someone
later changes the field name on either side of this contract, this fails.

    python tools\\_check_pack_check.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

spec = importlib.util.spec_from_file_location("mpc", HERE / "_model_pack_check.py")
mpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mpc)

MBF = "w600k_mbf"
R50 = "w600k_r50"


def scenario(label: str, sidecar: dict | None, active: str, want_ok: bool) -> bool:
    """Write a sidecar into a scratch gallery, run the check, demand a verdict."""
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "identities"
        g.mkdir()
        for n in ("Alpha", "Bravo", "Charlie", "Delta"):
            (g / f"{n}.npy").write_bytes(b"\0" * 16)
        if sidecar is not None:
            (g / "_provenance.json").write_text(json.dumps(sidecar), encoding="utf-8")

        old_pack = os.environ.get("FACE_MODEL_PACK")
        os.environ["FACE_MODEL_PACK"] = active
        try:
            info = mpc.load_pack(active, ident_dir=g,
                                 prov_path=g / "_provenance.json")
            ok = not info["mismatched"]
        finally:
            if old_pack is None:
                os.environ.pop("FACE_MODEL_PACK", None)
            else:
                os.environ["FACE_MODEL_PACK"] = old_pack

    verdict = "OK" if ok else "MISMATCH"
    want = "OK" if want_ok else "MISMATCH"
    good = ok == want_ok
    print(f"  {'PASS' if good else 'FAIL'}  {label}")
    print(f"        active={active}  got={verdict}  want={want}  "
          f"by_net={info['by_net']}")
    return good


def main() -> int:
    print("\n  proving _model_pack_check.py discriminates\n")
    results = [
        # A gallery built by the loaded pack. Healthy.
        scenario("matching gallery (all mbf) on buffalo_s",
                 {"Alpha": {"recognition_net": MBF},
                  "Bravo": {"recognition_net": MBF},
                  "Charlie": {"recognition_net": MBF},
                  "Delta": {"recognition_net": MBF}},
                 "buffalo_s", want_ok=True),

        # No sidecar at all - 33 vectors enrolled before the net was recorded.
        # Attributed to the service default, so still healthy on buffalo_s.
        scenario("no sidecar on buffalo_s (pre-dates the field)",
                 None, "buffalo_s", want_ok=True),

        # The dangerous one: pack switched, gallery not rebuilt.
        scenario("mbf gallery served as buffalo_l",
                 {"Alpha": {"recognition_net": MBF},
                  "Bravo": {"recognition_net": MBF},
                  "Charlie": {"recognition_net": MBF},
                  "Delta": {"recognition_net": MBF}},
                 "buffalo_l", want_ok=False),

        # A half-rebuilt gallery - the case the old check missed entirely.
        scenario("MIXED gallery: 2 mbf + 2 r50 on buffalo_s",
                 {"Alpha": {"recognition_net": MBF},
                  "Bravo": {"recognition_net": MBF},
                  "Charlie": {"recognition_net": R50},
                  "Delta": {"recognition_net": R50}},
                 "buffalo_s", want_ok=False),

        # The reverse direction, to prove the comparison is not one-sided.
        scenario("r50 gallery served as buffalo_s",
                 {"Alpha": {"recognition_net": R50},
                  "Bravo": {"recognition_net": R50},
                  "Charlie": {"recognition_net": R50},
                  "Delta": {"recognition_net": R50}},
                 "buffalo_s", want_ok=False),

        # Partial sidecar: some recorded, some not, all consistent.
        scenario("partial sidecar, all mbf, on buffalo_s",
                 {"Alpha": {"recognition_net": MBF},
                  "Bravo": {"recognition_net": MBF}},
                 "buffalo_s", want_ok=True),

        # Partial sidecar where the RECORDED ones disagree - the unrecorded ones
        # must not paper over the recorded ones.
        scenario("partial sidecar, recorded ones disagree",
                 {"Alpha": {"recognition_net": MBF},
                  "Charlie": {"recognition_net": R50}},
                 "buffalo_s", want_ok=False),
    ]

    print(f"\n  {sum(results)}/{len(results)} scenarios behaved correctly")
    if all(results):
        print("  the check can fail, which is the only property that matters")
        return 0
    print("  *** the check does not discriminate - do not trust it ***")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
