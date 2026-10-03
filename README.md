# GrayTech Security — start here

The face-service lives in this repository. `face-service/` in the parent
workspace is the live working copy this was copied from; run the demo from
either, they are the same source.

## Quick start

```bash
uv venv --python 3.12
uv pip install insightface==0.7.3 onnxruntime==1.19.2 fastapi "uvicorn[standard]" numpy pillow python-multipart
uvicorn graytech.server:app --host 127.0.0.1 --port 8090 --workers 1
```

Open <http://127.0.0.1:8090>.

**`--workers 1` is not optional.** Each worker loads its own copy of the model;
several workers on one CPU or GPU is slower at best and an out-of-memory at
worst. Concurrency comes from the thread pool inside onnxruntime.

Python 3.12 is required (3.10–3.12 works; **3.13 has no insightface wheels**).

## Enrol identities first

The dashboard identifies people by matching against enrolled embeddings. With
an empty index everyone reads `UNKNOWN`. Build it from a folder of photos:

```bash
python -m demo.dataset_eval --folder "C:\Users\PureTrek\Desktop\Faces\Faces" \
  --per-identity 4 --verify-limit 1200 --reset
```

Filenames carry the ground truth (`Akshay Kumar_0.jpg` → identity "Akshay
Kumar"), so this prints a real top-1 accuracy figure as well as enrolling.

## What the demo does and does not do

**Real:** face detection, 512-d embeddings, identity matches, confidence
scores. insightface `buffalo_s` with `allowed_modules=['detection','recognition']`
so the genderage and extra landmark heads never load.

**Simulated:** the movement. People cross the scene because a scheduler moves
their images along a path. The frame analysed is a still photograph.

**Live camera** (`POST /api/camera`, off by default so it never opens a camera
by accident) replaces the dataset source with frames from the laptop webcam,
captured through the same ffmpeg/DirectShow path the relay's `vex-vision` tool
uses. Webcam visitors show their live frame inside their head.

## Accuracy caveat

The evaluation set is studio portraits — controlled lighting, cooperative
subjects, one face centred. A percentage measured on it describes that set only.
It is not evidence about real people, off-angle, or in poor light, where
accuracy is typically materially lower.

## Hardware this was measured on

Intel i5-8250U, 8 threads, 6.3 GB RAM, **Intel UHD 620 integrated — no NVIDIA,
no CUDA**. `onnxruntime` reports `CPUExecutionProvider` only.

- **CUDA is unreachable** here — no NVIDIA device.
- **DirectML is unavailable** — onnxruntime falls back with *"DXCore is not
  available on this platform"*.
- So there is no accelerator on this machine, by hardware or OS version.

Measured: **14.8 fps** sustained, 65.8 ms median per image (53.5 ms
detect+embed), 97.0% top-1 over 31 identities on held-out images. That is
short of the 17 fps CPU target in the original brief, and the GPU architecture
the brief headlined was never reachable here.

## Layout

```
graytech/server.py     dashboard + simulation + SSE event stream
app/__init__.py        the recognition service (FastAPI, model loaded once)
demo/dataset_eval.py   enrol identities + measure top-1 accuracy
demo/bench.py          per-image throughput, decode and inference separated
demo/stills.py         folder-of-stills demo
demo/webcam.py         live webcam demo via the vex-vision capture path
demo/accel_probe.py    CPU vs DirectML comparison
```

## Not in this repository

`.venv/`, `models/` (a ~280 MB downloaded pack), and `identities/*.npy` are all
gitignored — large, reproducible, and the embeddings are derived data. The model
re-downloads on first load; identities are rebuilt with `demo.dataset_eval`.