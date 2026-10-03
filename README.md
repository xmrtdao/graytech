# GrayTech Security

**Face detection and recognition over a monitored scene.**

A standalone [insightface](https://github.com/deepinsight/insightface) + ONNX Runtime
service, and a live dashboard that visualises people crossing a scene and
identifies them one by one.

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-5C3EE8?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![insightface 0.7.3](https://img.shields.io/badge/insightface-0.7.3-7B4FA?logo=python&logoColor=white)](https://github.com/deepinsight/insightface)
[![License: MIT](https://img.shields.io/badge/license-MIT-3ddc84.svg)](LICENSE)
[![Code size](https://img.shields.io/badge/code-13%20files-8b97a6)](https://github.com/xmrtdao/graytech)
[![Commits](https://img.shields.io/badge/commits-2-8b97a6)](https://github.com/xmrtdao/graytech/commits/main)

---

## Table of contents

- [What it does](#what-it-does)
- [The AI](#the-ai)
- [Capabilities](#capabilities)
- [Quick start](#quick-start)
- [Accuracy](#accuracy)
- [Hardware](#hardware)
- [Architecture](#architecture)
- [API](#api)
- [Limitations and honesty](#limitations-and-honesty)

---

## What it does

A scene is monitored. People cross a scan line. Each one is detected, embedded
into a 512-number vector, and matched against enrolled references by cosine
similarity — arriving at a named identity with a confidence score, or
`UNKNOWN` if nobody clears the bar.

The dashboard renders it live: stick figures walking the scene, real portraits on
their heads, a scan line they cross, a ring that turns **green** on a match or
**red** on a miss, and a streaming event log.

---

## The AI

Two convolutional networks, and nothing else:

| Role | Model | Input | Output |
|---|---|---|---|
| Detection — finds faces | SCRFD `det_500m` | 127.5 × 128 | boxes + landmarks |
| Recognition — identity vector | `w600k_mbf` (mobilefacenet-style) | 112 × 112 | **512-d embedding** |

Both from the **buffalo_s** pack, running on ONNX Runtime's `CPUExecutionProvider`.

`allowed_modules=['detection','recognition']` means only **two of buffalo_s's six
heads ever load**. The genderage and both landmark heads are discarded — roughly
40% of the pack's compute for outputs nothing here consumes:

```
find model:  det_500m.onnx    detection
model ignore: genderage.onnx   genderage        <- dropped
model ignore: 1k3d68.onnx      landmark_3d_68   <- dropped
model ignore: 2d106det.onnx    landmark_2d_106  <- dropped
find model:  w600k_mbf.onnx   recognition
```

**Matching is not AI.** Cosine similarity between two 512-d vectors is a dot
product. No model is involved — which is exactly why the index is a plain NumPy
matrix rather than FAISS. At tens of identities, a matmul is faster than FAISS's
call overhead; FAISS earns its place past ~50k.

**No language model is involved anywhere.** The wider XMRT relay uses Zen,
Ollama and cloud models — none of it touches this.

---

## Capabilities

### Recognition

- Face detection with confidence-scored bounding boxes and 5-point landmarks
- 512-d embeddings, L2-normalised so cosine similarity is a plain dot product
- Identity matching by cosine similarity with a configurable threshold
- Enrolment from a labelled folder, **averaging multiple photos per person** so
  the stored vector is representative rather than one arbitrary frame
- Webcam stills and video frames recognised identically to stored images

### Self-calibration

- Every score logged — **a score and its correctness, never image bytes**
- False accepts and false rejects measured from observed traffic, not assumed
- Threshold recommendation with a stated reason and an explicit caveat
- Strangers clearing the bar **outrank** enrolled people being missed: a false
  accept hands a real identity to someone not enrolled, which is the dangerous
  direction
- If no unknown faces have been seen, it says it cannot speak to false accepts
  rather than reading a clean gallery as proof a low threshold is safe

### Adaptive matching

- Correcting a misidentification enriches that person's profile **immediately**
- Stored as a running mean, so one bad correction is diluted rather than
  overwriting a good profile
- No retraining, no restart — the next sighting uses the blended vector
- **Cannot create an identity.** The endpoint rejects names not already in the
  gallery; there is no path by which an unrecognised face becomes enrolled

### Dashboard

- Live scene canvas at 960 × 540 with a scan zone
- Real cropped portraits on the stick figures, feathered into a circular mask
- Colour-coded recognition state: green matched, red unknown, grey scanning
- Streaming event log over SSE — entries, scans, exits, errors
- Live calibration panel with threshold, observations, FA/FR rates, accuracy
- Pause / resume
- Works on desktop and phone

### Camera sources

- **Folder of stills** — demo source #1, no permissions needed
- **Live webcam** — off by default so it never opens a camera by accident
- Camera capture uses **the same ffmpeg / DirectShow path as the relay's
  `vex-vision` tool**, because `cv2.VideoCapture` on Windows builds its own
  capture graph, ignores DirectShow device naming, and would open a second
  competing handle on one camera

### Engineering

- Model loaded **once** in a lifespan handler — not per request, not via a lazy
  singleton that rebuilds on race
- **One worker, always.** Each worker loads its own copy of the model; several
  on one CPU or GPU is slower at best and an out-of-memory at worst. Concurrency
  comes from the thread pool inside ONNX Runtime
- Per-image timing with decode and inference separated
- Identities stored as plain `.npy` — inspectable, diffable, no database

---

## Quick start

```bash
git clone https://github.com/xmrtdao/graytech.git
cd graytech

uv venv --python 3.12
uv pip install insightface==0.7.3 onnxruntime==1.19.2 \
             fastapi "uvicorn[standard]" numpy pillow python-multipart

# enrol identities from a folder of labelled photos
python -m demo.dataset_eval --folder "/path/to/photos" --per-identity 4 --reset

# run the dashboard
uvicorn graytech.server:app --host 127.0.0.1 --port 8090 --workers 1
```

Open <http://127.0.0.1:8090>.

**`--workers 1` is not optional.** The model is ~20 MB and each worker loads its
own copy.

**Python 3.10–3.12.** Python 3.13 has no insightface wheels.

### Measure it

```bash
python -m demo.bench --folder "/path/to/photos" --count 200
python -m demo.accel_probe          # CPU vs DirectML on this machine
```

---

## Accuracy

Measured on **2,562 images / 31 identities**, held out from enrolment:

| Metric | Value |
|---|---|
| Faces detected | **1156 / 1156** (100%) |
| Top-1 correct | **1121** |
| **Top-1 accuracy** | **97.0%** |
| Throughput | **14.8 fps** sustained, 65.8 ms median/image |
| — detect + embed | 53.5 ms |
| — disk read | 5.3 ms |
| Enrolment | 31 identities in 12.7 s |

Scan time on the live dashboard runs **30–80 ms**.

### Read this before quoting the 97%

The evaluation set is **studio portraits** — controlled lighting, cooperative
subjects, one face centred in frame. **97.0% describes that set and nothing
wider.** It is not evidence about performance on real people, off-angle, in poor
light, at distance, or partially occluded, where accuracy is typically
materially lower.

Establishing a real number means testing on footage from the actual environment.

---

## Hardware

Built and measured on:

| | |
|---|---|
| CPU | Intel Core i5-8250U @ 1.60 GHz, 8 threads |
| RAM | 6.3 GB |
| GPU | **Intel UHD 620 integrated** |
| ONNX providers | `AzureExecutionProvider`, `CPUExecutionProvider` |

**There is no accelerator on this machine, by hardware or by OS version:**

- **CUDA — unreachable.** No NVIDIA device. `onnxruntime-gpu` cannot load.
- **DirectML — unavailable.** Falls back with *"DXCore is not available on this
  platform. This is expected on older versions of Windows."*
  (`demo/accel_probe.py` demonstrates this.)

So **14.8 fps is short of the 17 fps CPU target**, and the GPU-first
architecture this was originally specified against is unreachable here. Both are
properties of the hardware, not of the code.

**GPU path, for a CUDA box** — not tested, treat as a guide:

```bash
pip uninstall -y onnxruntime          # they share an import path; both installed
uv pip install onnxruntime-gpu==1.19.2  # silently clobbers and breaks CPU
```

Then `providers=['CUDAExecutionProvider']` and `FACE_MODEL_PACK=buffalo_l`.

---

## Architecture

```
graytech/server.py    dashboard, scene simulation, SSE event stream
app/__init__.py       the recognition service — FastAPI, model loaded once
app/calibration.py    threshold calibration + adaptive matching
demo/dataset_eval.py  enrol identities, measure top-1 against ground truth
demo/bench.py         per-image throughput, decode and inference separated
demo/stills.py        folder-of-stills demo
demo/webcam.py        live webcam via the vex-vision capture path
demo/accel_probe.py   CPU vs DirectML
```

**One process, one model.** The dashboard imports the service rather than
calling its HTTP API — a frame crosses the scan line every few seconds, and a
round trip plus a second model load per frame is wasteful. Same code path, so a
change to `app/` changes the dashboard too.

**Blocking work stays off the event loop.** Recognition runs on the simulation
thread, so ~40 ms of inference never stalls the UI.

**SSE over WebSocket.** This is one-directional — the server pushes, the browser
never sends — and SSE reconnects on its own.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | model, identities, live threshold, observation count |
| `GET /api/state` | full scene snapshot + calibration |
| `GET /api/stream` | SSE: live scene, events and calibration |
| `GET /api/events` | recent event log |
| `POST /api/pause` | pause / resume the simulation |
| `GET /api/profiles` | identities that have absorbed corrections |
| `GET /calibration` | current calibration state |
| `GET /calibration/recommend` | recommended threshold, with reasoning |
| `POST /calibration/threshold` | apply a threshold |
| `POST /calibration/learn` | enrich an **existing** profile |
| `POST /detect` | detect + embed every face in an image |
| `POST /register` | store one 512-d embedding |

The service's own endpoints (`/detect`, `/register`, `/identities`) live on
`app:app` — a separate ASGI app from the dashboard.

---

## Limitations and honesty

**What is real:** face detection, the 512-d embeddings, every identity match,
every confidence score. The threshold calibration and the profile adaptation.

**What is simulated:** the movement. People cross the scene because a scheduler
moves their images along a path. There is no camera tracking real motion unless
live camera mode is switched on. The frame analysed is a still photograph.

**The network does not learn.** Its weights are frozen and nothing updates them.
Retraining would need thousands of labelled images per identity; with a gallery
of tens it would overfit — excellent on the set it trained on, worse on new
arrivals. What adapts is the operating point and the enrolled profiles. Say that
plainly rather than claiming the model gets smarter on its own.

**Calibration is provisional.** It is derived from observed traffic, not a
held-out set, and a recommendation from a few dozen observations is a starting
point, not a final answer.

**Known issues, all fixed but recorded here because they were instructive:**

- A hoisted `t0 = time.perf_counter()` above a loop reported *cumulative* time
  as per-image latency, producing a "median" of 61 s alongside a 10.2 fps figure
  measured over the same pass. Accuracy was unaffected; the latency line was
  meaningless. The timer is now per-image and inside the loop.
- `cv2.bitwise_join` does not exist in OpenCV 5. Swallowed by a bare
  `except Exception: pass`, it presented as *"0 faces detected"* rather than a
  crash.
- The SSE generator built its payload from `scene.snapshot()` instead of the
  enriched `state()`, so the live stream silently lacked every field the REST
  endpoint returned.
- Dataset names (`Akshay Kumar`) and gallery keys (`Akshay_Kumar`) differ;
  comparing them naively marks correct matches incorrect.

---

## Licence

MIT. Models carry their own licences — `buffalo_s` ships with insightface under
its own terms and is **downloaded at runtime, not vendored here**.

## Related

Part of the [XMRT DAO](https://github.com/xmrtdao) ecosystem. See
[sea-hampton-house](https://github.com/xmrtdao/sea-hampton-house) for the
agency site.