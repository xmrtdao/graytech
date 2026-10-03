"""
Experiment: does DirectML beat plain CPU on this box?

Question: the machine has an Intel UHD 620 and no NVIDIA GPU. CUDA is therefore
unreachable. DirectML is the one accelerator path available on Windows without
CUDA. This measures whether it is worth using.

Method: identical model, identical input, identical session config except the
execution provider. Reports median over N runs, not mean, because a single GC
pause or scheduler hiccup skews a mean badly at this sample size.

Honest framing going in: the UHD 620 is a 2018 integrated part. DirectML
partitions a graph across a GPU and a CPU, and any op the shader compiler
cannot map falls back to CPU. So the realistic outcomes are "some speedup",
"a wash", or "slower because of partitioning overhead". Only one of those makes
DirectML worth the extra moving part in production.
"""

import statistics
import sys
import time

MODEL_ROOT = "./models"
MODEL_PACK = "buffalo_s"
DET_SIZE = 320
RUNS = 8


def synth_face_image():
    """
    A synthetic image. Not a real face - it is here to make the timing loop
    measure the pipeline, not the accuracy. Accuracy needs demo.stills with
    real photographs; this cannot tell you whether recognition works.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    img = (rng.random((480, 640, 3)) * 60).astype("uint8")
    # a rough head-shaped blob, so the detector has something face-like to chew
    cy, cx = 240, 320
    yy, xx = np.ogrid[:480, :640]
    mask = ((yy - cy) ** 2 / 150 ** 2 + (xx - cx) ** 2 / 110 ** 2) < 1
    img[mask] = (150, 120, 105)
    return img


def build(provider: str):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name=MODEL_PACK, root=MODEL_ROOT,
        allowed_modules=["detection", "recognition"],
        providers=[provider],
    )
    app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE), det_thresh=0.5)
    return app


def time_it(app, img, runs=RUNS):
    """Warm up, then time. The first call includes lazy allocation and is dropped."""
    app.get(img)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        app.get(img)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def report(name, samples):
    med = statistics.median(samples)
    print(f"  {name:<26} median {med:8.1f} ms   min {min(samples):8.1f}   "
          f"max {max(samples):8.1f}   -> {1000/med:5.1f} fps")
    return med


def main() -> int:
    import onnxruntime as ort
    print("\n  onnxruntime", ort.__version__)
    print("  available providers:", ort.get_available_providers())
    print()

    img = synth_face_image()
    results = {}

    # ── CPU baseline ──
    print("  building CPU session...")
    cpu = build("CPUExecutionProvider")
    results["CPUExecutionProvider"] = report("CPU (1 intra-op thread)",
                                            time_it(cpu, img))

    # Reconfigure for a fair comparison: let ORT pick its own thread count,
    # which is what a default deployment would do.
    print("\n  re-measuring CPU with default threading...")
    cpu_def = build("CPUExecutionProvider")
    for s in cpu_def.app.models.values() if hasattr(cpu_def.app, "models") else []:
        pass
    results["CPU (default threads)"] = report("CPU (default threads)",
                                             time_it(cpu_def, img))
    del cpu, cpu_def

    # ── DirectML ──
    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("\n  DmlExecutionProvider not present - onnxruntime-directml not installed.")
        print("  install it with:")
        print("    .venv\\Scripts\\python.exe -m pip uninstall -y onnxruntime")
        print("    uv pip install onnxruntime-directml")
        print()
        print("  NOTE: onnxruntime and onnxruntime-directml cannot coexist - same")
        print("  import path. Uninstalling onnxruntime breaks the CPU build.")
        return 2

    print("\n  building DirectML session...")
    dml = build("DmlExecutionProvider")
    results["DmlExecutionProvider"] = report("DirectML", time_it(dml, img))

    # ── verdict ──
    print("\n  --- verdict ---")
    cpu_med = min(v for k, v in results.items() if k.startswith("CPU"))
    dml_med = results["DmlExecutionProvider"]
    ratio = cpu_med / dml_med if dml_med > 0 else 0
    print(f"  fastest CPU {cpu_med:.1f} ms   DirectML {dml_med:.1f} ms   "
          f"speedup {ratio:.2f}x")
    if ratio >= 1.25:
        print("  DirectML is meaningfully faster. Worth wiring up, with the")
        print("  caveat that ops without a shader mapping still run on the CPU.")
    elif ratio >= 0.95:
        print("  A wash. DirectML adds a second execution path and a fallback")
        print("  surface for no measurable gain. Not worth it.")
    else:
        print("  DirectML is SLOWER. The partition/fallback overhead exceeds what")
        print("  the UHD 620 contributes. Stay on CPU.")
    print()
    print("  Reminder: this measures the pipeline on a SYNTHETIC image. It says")
    print("  nothing about recognition accuracy - that needs demo.stills on real")
    print("  photographs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())