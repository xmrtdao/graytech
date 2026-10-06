"""
Enrol from inside the serving process. No HTTP client involved.

Four separate attempts to drive POST /api/enrol over the wire all failed, none of
them server-side:

  * curl 7.55 stalls on the multipart handshake here. It sends
    `Expect: 100-continue`, uvicorn answers 100 Continue, the transfer never
    completes, and curl reports HTTP 000 or a bare 100 with no body. Disabling
    Expect and forcing HTTP/1.0 did not help either - the verbose trace shows the
    headers going out and the body never routing.
  * Invoke-WebRequest -Form needs PowerShell 7; there is no pwsh on this box.
    A hand-built multipart body got as far as "the underlying connection was
    closed".
  * Calling the ASGI app in-process from a second python is worse than useless:
    it writes .npy files that the running service has already loaded and will
    never see, and it needs face.load() by hand because lifespan never ran. That
    is how the gallery was left in a state nobody could vouch for.

So this takes the third option: run inside the service process itself, where the
live gallery matrix lives. Importing graytech.server gives the same `recogniser`
object the request handler uses, so a blend here is a blend there - same
L2-normalised average, same backup rotation, same provenance write, same
reload. Calling S.enrol(...) directly is exactly what FastAPI would have done
after parsing the form, minus the transport that keeps dying.

Not a substitute for the HTTP path, which is what a browser uses and which still
has to work. It exists so enrolment is not blocked on a curl bug. If this
succeeds and the HTTP path still fails for a real browser, that is a server bug
and this file is the evidence of it.

Run it with the service STOPPED, or accept that a concurrently-running service
will overwrite provenance on its next write.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET = Path(r"C:\Users\PureTrek\Desktop\Faces\Faces")


class _Upload:
    """Stands in for fastapi.UploadFile. enrol() only ever calls read()."""

    def __init__(self, path: Path):
        self.filename = path.name
        self._data = path.read_bytes()

    async def read(self) -> bytes:
        return self._data


async def _main(pattern: str) -> int:
    from app import face
    import graytech.server as S

    if getattr(face, "app", None) is None:
        print("  loading the recognition model (~70s)")
        face.load()
        face.reload_identities()
    print(f"  gallery before: {len(face.names)} identities")

    shots = sorted(DATASET.glob(pattern))
    if not shots:
        print(f"  no files matched {pattern!r} in {DATASET}")
        return 1
    print(f"  {len(shots)} shot(s):")
    for p in shots:
        print(f"    {p.name}  {p.stat().st_size} bytes")

    name = pattern.split("*")[0]
    uploads = [_Upload(p) for p in shots]
    print(f"\n  enrolling as {name!r} ...\n")
    res = await S.enrol(name=name, files=uploads)

    for k in ("ok", "name", "key", "updated_existing", "enrolled", "spread",
              "shot_quality", "shot_note", "reason", "identities"):
        if k in res:
            print(f"  {k:<17} {res[k]}")
    for r in res.get("rejected") or []:
        print(f"    REJECTED {r.get('file')}: {r.get('why')}")
    for d in res.get("similar_to") or []:
        print(f"    similar_to {d.get('name')} @ {d.get('cosine')}")

    print(f"\n  gallery after:  {len(face.names)} identities")
    prov = S._read_provenance()
    for key in sorted(prov):
        if key.lower() in {p.stem.lower() for p in shots} or \
           key.lower() == name.replace("-", "_").replace(" ", "_").lower():
            print(f"  provenance {key:<14} shots={prov[key]['shots']} "
                  f"batches={prov[key]['batches']}")
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "joe-lee*"
    raise SystemExit(asyncio.run(_main(pat)))
