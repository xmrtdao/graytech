# AstraGaze by Gray Tech Solutions

**Live face recognition for physical security — find, check, name.**

Three stages on one frame: a person detector gates a face detector, and the face
is matched against an enrolled gallery. A name only ever comes from a reference
somebody enrolled. Nobody is guessed at.

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-5C3EE8?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![insightface 0.7.3](https://img.shields.io/badge/insightface-0.7.3-7B4FA?logo=python&logoColor=white)](https://github.com/deepinsight/insightface)

- **Live console** — <https://graytech.mobilemonero.com>
- **Public brochure** — <https://astragaze.mobilemonero.com/graytech>

---

## Contents

- [What it does](#what-it-does)
- [The pipeline](#the-pipeline)
- [Measured accuracy](#measured-accuracy)
- [Capabilities](#capabilities)
- [How it compares](#how-it-compares)
- [Quick start](#quick-start)
- [API](#api)
- [Hardware](#hardware)
- [Architecture](#architecture)
- [Operating it](#operating-it)
- [Limitations](#limitations)

---

## What it does

A camera frame arrives. The system finds people, checks whether their face is
usable, and — only if it is — attaches a name it has been given permission to
attach.

The console renders all three live: the frame, a reticle per person that carries
the stage it has reached (cyan for a body, gold while the face is being checked,
green once a name is attached), and an event log where each sighting resolves
`FIND → CHECK → NAME`.

The browser owns the camera. The viewer picks their own device, nothing starts
without a click, and the server never has to guess what is plugged in.

---

## The pipeline

```
        ┌─ Stage 1 ──┐      ┌─ Stage 2 ──┐      ┌─ Stage 3 ──┐
frame → │ YOLO person │  →   │  SCRFD face │  →   │ ArcFace    │  →  named, or
        │  yolov8n    │      │  det_500m   │      │  w600k_mbf │      anonymous
        └─────────────┘      └─────────────┘      └─────────────┘
          gates everything      pixel budget        cosine vs gallery
          after it              gates it
```

| Stage | Model | Pack | Job |
|---|---|---|---|
| 1 · Find | `yolov8n.onnx` | — | Person on a wide frame. ~12 MB |
| 2 · Check | `det_500m.onnx` | `buffalo_s` | Face inside the person box. SCRFD |
| 3 · Name | `w600k_mbf.onnx` | `buffalo_s` | 512-d embedding. ArcFace / MobileFaceNet |

Stage 2 gates on a **pixel budget** before spending anything on recognition:
face width, inter-ocular distance, and Laplacian sharpness. Below ~20 px across
there is nothing to identify; a confident cosine computed off a 14 px crop is a
wrong answer rather than a hard one, so it is refused.

**Matching is arithmetic, not AI.** Cosine similarity between two 512-d vectors
is a dot product. The index is a plain NumPy matrix, not FAISS — at tens of
identities a matmul beats FAISS's call overhead. FAISS earns its place past ~50k.

**No language model is involved anywhere.**

---

## Measured accuracy

248 held-out photographs, seeded random so none was enrolled, run against the
gallery this service actually ships (15 shots per identity, 33 identities):

| Outcome | Count | Share |
|---|---|---|
| Top-1 correct — right person, above threshold | 247 | **99.6%** |
| Refused — nobody named, face below threshold | 1 | 0.4% |
| **Wrong identity attached** | **0** | **0.0%** |

Cosine distribution: `min 0.3813 · p05 0.5635 · p50 0.7064 · max 0.8675`, against
a live threshold of `0.45`.

The third row is the one a top-1 percentage hides. Refusing a face is correct
behaviour for a closed gallery. Attaching the **wrong** name is the failure that
matters operationally, and it is zero here.

Reproduce with `python tools/_heldout_top1.py`.

### Read this before quoting the number

The evaluation set is **studio portraits** — controlled lighting, cooperative
subjects, one face centred in frame. **99.6% describes that set and nothing
wider.** It is not evidence about performance on real people, off-angle, in poor
light, at distance, or partially occluded, where accuracy is typically materially
lower. Establishing a real number means testing on footage from the environment
the system will actually run in.

### Where it is genuinely weak

`tools/_audit_gallery.py` measures each identity against photos it never saw, at
webcam resolution. The shipped gallery:

| Identity | Held-out cosine | × threshold | Verdict |
|---|---|---|---|
| `Courtney_Cox` | 0.4729 | **1.05×** | will be refused regularly |
| `Andy_Samberg` | 0.5801 | 1.29× | thin margin |
| `Hugh_Jackman` | 0.5818 | 1.29× | thin margin |
| … | | | |
| `Amitabh_Bachchan` | 0.8159 | 1.83× | comfortable |

`Courtney_Cox` sits 0.02 above the bar. That is the number to act on, and the
console surfaces it: the enrolment panel lists identities weakest-first rather
than alphabetically, because alphabetical hides exactly this.

### Throughput

Measured on this machine (see [Hardware](#hardware)):

| | |
|---|---|
| Sustained, full pass, 60 images | **15.5 fps** |
| Total per image | 122 ms median, 141 ms mean |
| — detect + embed | 90 ms median |
| — read from disk | 6 ms median |
| Stage 1 · person find (live frame) | 90–225 ms |
| Cold start to answering `/health` | ~70 s |
| Gallery | 33 identities, 512-d each |

All figures from `python -m demo.bench` and `tools/_heldout_top1.py` on this
machine. The 17 fps target was for a CPU path this hardware does not quite
reach; the numbers above are measured, not inherited.

---

## Capabilities

### Recognition

- Face detection with scored boxes and 5-point landmarks
- 512-d embeddings, L2-normalised so cosine is a plain dot product
- Identity matching by cosine similarity against an enrolled gallery
- **Multi-shot enrolment** — 5–15 photographs averaged into one reference vector,
  which is the centroid of that person's embedding cloud and far more stable than
  any single frame
- Webcam frames, video frames and stills recognised identically

### Enrolment, from the browser

`POST /api/enrol` plus a panel on the console. This was previously a route no
page called, which by this project's own standard made it not a feature — adding
a face meant running a script by hand on one machine.

- Name, multiple photographs, one submit
- Blends into an existing profile rather than overwriting one that is matching
- **Per-file verdict**: a photo with no detectable face, or a face too small to
  trust, is named and explained rather than quietly dropped
- Flags a face that closely resembles someone already enrolled — two identities
  for one person would each match only their own photographs
- **Will not manufacture shots.** Averaging fourteen copies of one photograph
  raises self-similarity and teaches nothing about variation. Upload one and it
  says the identity is thin and will struggle.

### Self-calibration

- Every score logged — **a score and its correctness, never image bytes**
- False accepts and false rejects measured from observed traffic, not assumed
- A recommendation with a stated reason and an explicit caveat
- Strangers clearing the bar **outrank** enrolled people being missed: a false
  accept hands a real identity to someone not enrolled, which is the dangerous
  direction
- If no unknown faces have been seen, it says it cannot speak to false accepts
  rather than reading a clean gallery as proof a low threshold is safe

### Adaptive matching

- A correction enriches that person's profile immediately
- Stored as a running mean, so one bad correction is diluted rather than
  overwriting a good profile
- No retraining, no restart
- **Cannot create an identity.** There is no path by which an unrecognised face
  becomes enrolled through the adaptive path

### Pixel budget

A measured quality gate in front of recognition, from the rig spec expressed as
code so the gate, the console and the build plan cannot disagree:

| Band | Face width | Meaning |
|---|---|---|
| `reject` | < 20 px | nothing to identify |
| `detect` | 20–40 px | face found, not identified |
| `identify` | 40–80 px | IEC 62676-4 ≈ 40 px for identification |
| `robust` | ≥ 80 px | pose, motion and backlight tolerated |

Measured against the image that actually reached the model — never assumed from
the scene.

### Tracking

- IoU tracker with non-maximum suppression, so one person is not counted ten
  times in a crowd
- Independent per-session identity space
- Counting is of people **actually on screen**, not of people mid-approach
  outside the frame

### Console

- Live frame with per-person reticles carrying their current stage
- Event log resolving `FIND → CHECK → NAME` per sighting
- Calibration panel: threshold, observations, FA/FR rates, accuracy
- **Enrolment panel** — gallery sorted weakest-first, each identity showing its
  own shot count
- Pause / resume; works on desktop and phone
- Styling aligned to [graytechsolutions.dev](https://graytechsolutions.dev) —
  olive-ink surfaces, 1 px rules, square corners, gold accent, Saira Condensed /
  Barlow / Space Mono. The five **state** colours are the deliberate exception:
  green, red, amber, cyan and violet carry meaning here (matched, unknown,
  calibration warning, entry, exit), so they are retuned warm to sit on olive
  rather than collapsed into the gold. The canvas reads its palette back out of
  the stylesheet rather than hard-coding hex.

---

## How it compares

Judged against the four things someone actually reaches for when they want face
recognition that runs somewhere they control. **This section is deliberately
unflattering where the comparison is unflattering.**

| | **AstraGaze** | InsightFace 2.x | DeepFace | dlib | commercial VMS |
|---|---|---|---|---|---|
| What you get | whole pipeline: find, gate, track, enrol, calibrate, console | toolkit + server + GUI + CLI | one-liner API | models + C++ | appliance + licence |
| Pixel/quality gate | **yes**, measured, spec-derived | no | no | no | rarely exposed |
| Multi-shot enrolment | **yes**, 15-shot averaged | reference-photo selection (2.0) | manual | manual | via enrolment UI |
| Threshold from observed errors | **yes**, with FA/FR | no | no | no | opaque |
| Won't name a stranger | **yes**, enforced in the matcher | depends on you | depends on you | depends on you | not the point |
| Person tracking | **yes**, NMS + IoU | PrivateFrame 2.0 | no | no | yes |
| Liveness / PAD | **no** | **yes**, optional RGB (2.0) | no | no | usually |
| Scale / search | 33 identities, NumPy matmul | 50M+ images, INT8 quantised | no | no | vendor scale |
| Deployment | single process, CPU-only | container, CPU or GPU | Python lib | C++/Python | appliance |
| Operator console | **yes** | server Web UI, GUI | no | no | yes |
| Accuracy on this gallery | **99.6%** top-1, 0 wrong | not measured here | wrapper over others | weaker on modern sets | not published |
| Code licence | MIT | MIT (code), models non-commercial | MIT + model terms | MIT | proprietary |
| You own the data | **yes** | yes | yes | yes | vendor cloud |

**Where this genuinely wins.** Not raw accuracy — the embedding comes from
InsightFace, so against a bare InsightFace install the recognition quality is
*identical by construction*. What this adds is everything around the model: the
pixel budget that refuses a confident score computed from 14 pixels, the
calibration that sets the operating point from observed errors instead of
inherited defaults, NMS so a crowd is not counted twice, and an operator console
where a human can see *why* something was refused.

**Where it genuinely loses.** This list is the reason to read the table:

- **Against InsightFace 2.x, which is now ahead on several axes.** The upstream
  project has shipped optional RGB liveness detection, PrivateFrame for local
  face blur and reference-photo selection, PersonAnalysis with body ReID, and a
  self-hosted server with INT8 quantised embedding search over 50M+ images on one
  GPU. AstraGaze is built on 0.7.3 and has **none** of that. A new deployment
  should evaluate InsightFace 2.x before writing glue around 0.7.x — the
  upstream project is actively maintained and this is not.
- **No liveness detection.** A printed photograph held to the camera is a face.
  For a physical-security deployment that is a material gap, not a caveat.
- **Scale.** 33 identities in a NumPy matrix. InsightFace's server does 50M on
  one GPU. This has not been tested at 10,000.
- **Against a GPU box**, `buffalo_l` (ResNet-50/100) is materially more accurate.
  It cannot run fast here — no NVIDIA device — and swapping it invalidates every
  stored vector, since `w600k_r50` and `w600k_mbf` embeddings are different
  spaces. A guard detects that mismatch and reports `degraded` rather than
  silently returning wrong names.
- **Against a team.** One machine's work. There is no test suite covering the
  recognition path end to end, and every check in `tools/` exists because a
  defect once passed it.

**Deliberately not attempted:** telling you who an unknown person *is*. That
requires a watchlist, not an enrolment gallery, and that line is exactly where
this project refuses to go.

---

## Quick start

```bash
git clone https://github.com/xmrtdao/graytech.git
cd graytech

uv venv --python 3.12
uv pip install insightface==0.7.3 onnxruntime==1.19.2 \
             fastapi "uvicorn[standard]" numpy pillow python-multipart

# run the console
uvicorn graytech.server:app --host 127.0.0.1 --port 8090 --workers 1
```

Open <http://127.0.0.1:8090> and enrol from the panel on the page.

**`--workers 1` is not optional.** Each worker loads its own copy of the model;
several on one CPU is slower at best and an out-of-memory at worst. Concurrency
comes from the thread pool inside ONNX Runtime.

**Python 3.10–3.12.** Python 3.13 has no insightface wheels.

### Verify what you just installed

```bash
python tools/_check_pack_check.py     # the model/gallery guard can fail
python tools/_heldout_top1.py         # top-1 on photos the gallery never saw
python tools/_audit_gallery.py        # per-identity margin against the threshold
python tools/check_console_js.py      # the console's inline script actually runs
```

Every one of these exists because the thing it checks once passed while being
wrong. See [Limitations](#limitations).

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model pack, **gallery/model agreement**, threshold |
| `GET /api/state` | full snapshot + calibration |
| `GET /api/stream` | SSE: live state, events, calibration |
| `GET /api/events` | recent event log |
| `POST /api/live/frame` | a browser camera frame through all three stages |
| `POST /api/live/stop` | end a live session |
| `POST /api/pause` | pause / resume |
| `POST /api/persons` | stage 1 on demand, one uploaded scene frame |
| `GET /api/persons/status` | whether stage 1 is loaded |
| `GET /api/identities` | who is enrolled, and how many shots each has |
| `POST /api/enrol` | enrol from one or more photographs |
| `DELETE /api/identities/{key}` | remove an identity (kept as `.npy.1`) |
| `GET /api/profiles` | identities that have absorbed corrections |
| `GET /calibration` | calibration state |
| `POST /calibration/apply` | apply a threshold |
| `GET /` | the console |

---

## Hardware

Built and measured on:

| | |
|---|---|
| CPU | Intel Core i5-8250U @ 1.60 GHz, 8 threads |
| RAM | 6.3 GB |
| GPU | Intel UHD 620 **integrated** |
| ONNX providers | `AzureExecutionProvider`, `CPUExecutionProvider` |

**There is no accelerator here, by hardware or by OS version:**

- **CUDA — unreachable.** No NVIDIA device, so `onnxruntime-gpu` cannot load and
  TensorRT FP-16 is not available on this machine at all.
- **DirectML — unavailable.** *"DXCore is not available on this platform."*

So **15.5 fps** measured is short of the 17 fps CPU target. Both are properties
of the hardware, not of the code.

**On a CUDA box** — not tested here, treat as a guide:

```bash
pip uninstall -y onnxruntime           # they share an import path
uv pip install onnxruntime-gpu==1.19.2 # silently clobbers and breaks CPU
```

Then `providers=['CUDAExecutionProvider']`. Switching to `buffalo_l` requires
**re-enrolling every identity** — see the guard in `tools/_model_pack_check.py`.

---

## Architecture

```
graytech/server.py      console, live ingest, SSE stream, enrolment endpoint
app/__init__.py         the recognition service — model loaded once
app/personfind.py       stage 1, NMS, IoU tracker, per-session identity space
app/calibration.py      threshold calibration + adaptive matching
demo/dataset_eval.py    enrol, measure top-1 against ground truth
demo/bench.py           per-image throughput, decode and inference separated
demo/accel_probe.py     CPU vs DirectML
tools/                  the checks below, and the harnesses that proved them needed
```

**One process, one model.** The console imports the service rather than calling
its HTTP API — a frame crosses the scan line every few seconds, and a round trip
plus a second model load per frame is wasteful. Same code path, so a change to
`app/` changes the console too.

**Blocking work stays off the event loop.** Recognition runs on its own thread,
so ~40 ms of inference never stalls the UI.

**SSE over WebSocket.** One-directional — the server pushes, the browser never
sends — and SSE reconnects on its own.

**Supervised.** Root `supervisor.mjs`, not `relay/supervisor.mjs`, which is a
stale duplicate. Measured recovery after a kill: **56–75 s**, dominated by model
loading. `tcpPort` is declared on the definition because a service missing from
the supervisor's port map silently loses the port-kill and port-wait on restart,
and then loses every spawn to `EADDRINUSE`.

---

## Operating it

**When the console is down**, <https://astragaze.mobilemonero.com/graytech> serves
a brochure from the relay — a separate process with no dependency on the
recognition model. Cloudflare Pages would be the natural home but is blocked:
the token in `relay/.env` verifies and reads DNS zones while 403ing at account
scope, and the other token on the machine belongs to an account that does not
serve this hostname. See `relay/README.md`.

**Identities are derived data.** Rebuild from the dataset with
`tools/_enrol.py`; they are gitignored because a stale shot count committed to
git is worse than none, since it looks authoritative.

---

## Limitations

**What is real:** face detection, the 512-d embeddings, every identity match,
every confidence score, the pixel gate, the calibration, the tracking.

**What is not.** There is no liveness or presentation-attack detection — a
printed photograph held to the camera is a face. There is no multi-camera fusion
and no watchlist matching: a name only comes from an enrolled gallery, so the
system cannot tell you who an unknown person *is*. The network's weights are
frozen; retraining would need thousands of labelled images per identity and would
overfit a gallery of tens. What adapts is the operating point and the enrolled
profiles.

**Scale is unproven.** 33 identities in a plain matrix. InsightFace 2.x ships
INT8-quantised search over 50M images on a single GPU; nothing here has been
tested past two digits, and the NumPy index is the obvious ceiling.

**InsightFace 2.x is ahead of this.** Upstream now ships optional RGB liveness,
PrivateFrame, PersonAnalysis with body ReID, and a self-hosted server with a Web
UI. This is built on 0.7.3 and has none of it. For a new deployment, evaluate
InsightFace 2.x before writing glue around 0.7.x.

**Calibration is provisional.** Derived from observed traffic, not a held-out set.
A recommendation from a few dozen observations is a starting point.

**Licensing is load-bearing.** The InsightFace *code* is MIT. The trained *models*
are not: upstream states the training data and models trained on it are available
for **non-commercial research purposes only**, and that applies whether you
download the pack by hand or let the Python library fetch it for you. For
commercial deployment, upstream directs you to
`recognition-oss-pack@insightface.ai` for licensing. Anyone building on this
needs to resolve that before procurement, not after.

### Checks that exist because they once passed while wrong

Every one of these was a green check reporting a broken system:

- **`node --check` validates syntax only.** It is perfectly happy with a reference
  to a name that does not exist. Two production breaks shipped through it — a
  temporal-dead-zone read of `isStill`, then the same for `onFace` and `idrec` and
  `lw`, each thrown on every live frame. `tools/check_console_js.py` now runs the
  console's script against a DOM and calls the entry points, because syntax is
  not behaviour.
- **A check that reads a field nobody writes cannot fail.** The model-pack guard
  read `pack` while the writer stored `recognition_net`, and read it from a
  module global a caller had reassigned — so it measured the real gallery and
  called it healthy against a gallery deliberately marked half-`r50`. Now 7/7
  scenarios discriminate, via `tools/_check_pack_check.py`.
- **A non-reentrant lock, silently.** `_read_provenance` took `_prov_lock` while
  `_write_provenance` held it, so every enrolment deadlocked *after* `np.save`
  had written the vector. The gallery changed, the shot count did not, and nothing
  raised. `tools/_check_provenance.py` reproduces the deadlock and asserts the fix.
- **A stub that invents what it is asked for cannot detect a typo.** The console
  harness's `getElementById` fabricated an element for any id, so
  `enrolCountTYPO` wrote into the void and the harness passed. It now derives the
  real id list from the page HTML and rejects anything else.
- **A self-match is not an accuracy measurement.** An early robustness harness
  reported 0.97–0.99 under degradation by matching the one photograph a
  single-shot identity was built from against itself.
- **`pick_shots` includes the last photo in its linspace,** so "held out" testing
  against it was inside the prototype. Fixed; the real numbers are lower and more
  useful.

### Known issues, fixed but recorded because they were instructive

- A hoisted timer above a loop reported cumulative time as per-image latency,
  producing a "median" of 61 s beside a 10.2 fps figure from the same pass.
- `cv2.bitwise_join` does not exist in OpenCV 5. Swallowed by a bare
  `except: pass`, it presented as "0 faces detected" rather than a crash.
- The SSE generator built its payload from `scene.snapshot()` instead of the
  enriched state, so the live stream silently lacked every field the REST endpoint
  returned.
- Dataset names (`Akshay Kumar`) and gallery keys (`Akshay_Kumar`) differ;
  comparing them naively marks correct matches incorrect.
- `object-fit: cover` cropped 776 px of a 720×1280 portrait feed out of a
  landscape HUD. The subject was never drawn.

---

## Licence

MIT for this code.

The **models are not MIT.** `buffalo_s` is InsightFace's own pack, downloaded at
runtime and not vendored here. Upstream states that its *training data, and the
models trained on that data*, are available for **non-commercial research
purposes only** — including when fetched automatically by the Python library. For
commercial use, upstream directs licensing requests to
`recognition-oss-pack@insightface.ai`.

That is a procurement question, not a technical obstacle, and it is the single
most consequential line in this README for anyone considering deployment.

Part of the [XMRT DAO](https://github.com/xmrtdao) ecosystem.