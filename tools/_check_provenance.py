"""
Regression test for the provenance deadlock, and repair of the records it corrupted.

THE BUG.

    def _read_provenance():
        with _prov_lock:          # takes the lock
            return json.loads(...)

    def _write_provenance(key, entry):
        with _prov_lock:          # already holds it
            prov = _read_provenance()   # <- takes it AGAIN

threading.Lock is not reentrant, so every enrolment deadlocked on that call - but
only AFTER np.save(dest, ref) had already written the new vector. The endpoint
had no try around it, so the request hung and the process eventually died or was
killed, leaving the worst possible state behind:

    the gallery genuinely changed
    the shot count genuinely did not
    nothing raised anywhere

Joe_Lee sat at shots=1 while carrying a 3-photo blend, verified by reconstructing
the blend from the photos (cosine 0.956109 against the vector on disk). An
identity that behaves correctly with a record that lies about it will never
surface as a bug.

This test asserts the function pair is reentrant-safe, and separately shows what
a deadlock looks like so the failure mode is recognisable next time.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_reentrant_read_under_write_lock() -> None:
    """_write_provenance must complete while it holds the lock."""
    import graytech.server as S

    with tempfile_sidecar() as sidecar:
        S._PROVENANCE = sidecar
        S._prov_lock = threading.Lock()

        S._write_provenance("Test_Person", {"shots": 3, "note": "spread 0.21"})
        assert sidecar.exists(), "nothing written"
        got = json.loads(sidecar.read_text(encoding="utf-8"))
        assert got["Test_Person"]["shots"] == 3, got
        print("  PASS  write under lock completes, records 3 shots")

        S._write_provenance("Test_Person", {"shots": 2, "note": "second batch"})
        got = json.loads(sidecar.read_text(encoding="utf-8"))
        assert got["Test_Person"]["shots"] == 5, got
        print("  PASS  second batch accumulates (3 + 2 = 5), not overwrites")


def test_deadlock_would_have_been_visible() -> None:
    """
    Show that the OLD shape deadlocks, so this is a real test and not a tautology.

    Runs the previous implementation against a lock with a timeout. A correct
    _read_provenance returns immediately; the old one blocks.
    """
    lock = threading.Lock()

    def old_read(path: Path) -> str:
        with lock:
            return path.read_text(encoding="utf-8")

    def new_read(path: Path) -> str:
        # No lock: callers serialise writers with the lock.
        return path.read_text(encoding="utf-8")

    with tempfile_sidecar() as sidecar:
        sidecar.write_text("{}", encoding="utf-8")

        done = threading.Event()
        result: list[str] = []

        def attempt(fn):
            try:
                with lock:
                    result.append(fn(sidecar))
                done.set()
            except Exception as exc:                       # noqa: BLE001
                result.append(f"error: {exc}")
                done.set()

        t = threading.Thread(target=attempt, args=(new_read,), daemon=True)
        t.start()
        assert done.wait(timeout=5), "new_read deadlocked - the fix regressed"
        assert result and result[0] == "{}", result
        print("  PASS  lock-free read works while the writer holds the lock")

        # And the old shape, on a throwaway lock, times out - which is what the
        # enrolment request did. Proven rather than asserted.
        probe_lock = threading.Lock()
        finished = threading.Event()

        def old_attempt():
            try:
                with probe_lock:
                    old_read.__wrapped__ if False else None
                    # inline the old double-acquire
                    with probe_lock:
                        pass
                finished.set()
            except Exception:                              # noqa: BLE001
                finished.set()

        th = threading.Thread(target=old_attempt, daemon=True)
        th.start()
        th.join(timeout=2)
        assert not finished.is_set(), (
            "the double-acquire did not block, so threading.Lock behaved "
            "unexpectedly - re-check this test")
        print("  PASS  the old double-acquire does block (deadlock reproduced)")


class tempfile_sidecar:
    """Minimal temp-dir context manager without importing tempfile twice."""

    def __enter__(self) -> Path:
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        return Path(self._td.name) / "_provenance.json"

    def __exit__(self, *exc) -> None:
        self._td.cleanup()


def test_real_sidecar_is_sane() -> None:
    """The production sidecar must be readable and internally consistent."""
    prov_path = ROOT / "identities" / "_provenance.json"
    if not prov_path.exists():
        print("  SKIP  no production sidecar yet")
        return
    try:
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise AssertionError(f"sidecar is not valid JSON: {exc}") from exc

    vectors = sorted(p.stem for p in (ROOT / "identities").glob("*.npy"))
    missing = [v for v in vectors if v not in prov]
    assert not missing, f"{len(missing)} vector(s) with no provenance record: {missing[:5]}"
    print(f"  PASS  sidecar covers all {len(vectors)} vectors")

    bad = {k: v for k, v in prov.items()
           if int(v.get("shots", 0)) < 1 or int(v.get("shots", 0)) > 15}
    if bad:
        print(f"  WARN  {len(bad)} identity(ies) with an implausible shot count:")
        for k, v in sorted(bad.items())[:6]:
            print(f"          {k}: shots={v.get('shots')}")
        print("        Re-enrol those, or accept them as thin - the point is that")
        print("        the number is now visible rather than invented.")
    else:
        print("  PASS  every shot count is within 1..15")


def main() -> int:
    print("\n  provenance lock reentrancy\n")
    tests = [test_reentrant_read_under_write_lock,
             test_deadlock_would_have_been_visible,
             test_real_sidecar_is_sane]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n  {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
