"""Run the remaining pipeline steps strictly one after another.

Sequential on purpose: two GPU jobs on the 6 GB laptop GPU spill into system
RAM and slow each other ~10x. Every step is cached/resumable, so a re-run
skips finished work.

Usage:
    python tools/run_pipeline.py run2      # text features -> notebooks 04, 05
    python tools/run_pipeline.py final     # + segment states, adapter heads, 4-bit Qwen3-4B
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "artifacts" / "extract.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} [pipeline] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, name):
    log(f"start {name}")
    t0 = time.time()
    with open(ROOT / "artifacts" / f"{name}.log", "w", encoding="utf-8") as out:
        rc = subprocess.call(args, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    log(f"{'done' if rc == 0 else 'FAILED'} {name} (exit {rc}, {(time.time() - t0) / 60:.1f} min)")
    if rc != 0:
        sys.exit(rc)


def notebook(nb):
    run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=shl",
         f"notebooks/{nb}.ipynb"], nb)


def qwen4b_ready() -> bool:
    snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots"
    files = list(snap.glob("*/model-*.safetensors")) if snap.exists() else []
    return len(files) >= 3 and all(f.stat().st_size > 0 for f in files)


def main(mode):
    run([PY, "tools/extract_text.py"], "text_pipeline")       # cached steps are skipped
    for b in ("build_nb04", "build_nb05", "build_nb06"):
        run([PY, f"tools/{b}.py"], b)
    if mode == "final":
        run([PY, "tools/extract_segments.py"], "segments")
        if qwen4b_ready():
            run([PY, "tools/extract_llm.py", "--model", "Qwen/Qwen3-4B", "--tag", "4b", "--four-bit"], "llm4b")
        else:
            log("Qwen3-4B not fully downloaded: skipping the 4B view")
    notebook("04_heads")
    if mode == "final":
        notebook("06_adapter_heads")
    notebook("05_stack_submit")
    log(f"{mode} pipeline complete")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run2")
