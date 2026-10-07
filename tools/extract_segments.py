"""Extract 1-second segment states of each encoder's best layer band (band A)
for the trained layer-adapter heads. One forward pass per full clip; crops
are cut from the segments later, so they need no extra GPU work.

Usage: python tools/extract_segments.py [--enc whisper_v3 wavlm_large]
Output: artifacts/features/seg_{key}.npy (n_clips, 4, max_s, d) float16 and
        seg_{key}_lens.npy (number of real 1-s segments per clip).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SHL_DATA", str(ROOT / "shl-hiring-assessment-2026"))
from shl import audio, embed, heads  # noqa: E402

OUT = ROOT / "artifacts"
FEAT = OUT / "features"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "extract.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", nargs="*", default=["whisper_v3", "wavlm_large"])
    args = ap.parse_args()
    meta = pd.read_csv(OUT / "meta.csv")
    prof = pd.read_csv(OUT / "layer_profiles.csv")
    for key in args.enc:
        out = FEAT / f"seg_{key}.npy"
        if out.exists():
            log(f"segments {key}: cached")
            continue
        s, e = heads.bands_for(prof[prof.encoder == key])[0]
        log(f"segments {key}: layers {s}-{e - 1}")
        ext = embed.LayerStats(key)
        # WavLM's relative-position attention on a full 60 s clip sits at the
        # 6 GB limit and ran out of memory on one clip: encode 30 s windows.
        ext.ssl_window_s = 30
        segs, t0 = [], time.time()
        for k, p in enumerate(meta.path, 1):
            segs.append(ext.segments(audio.load_audio(p), range(s, e)))
            if k % 200 == 0 or k == len(meta):
                rate = k / (time.time() - t0)
                log(f"segments {key}: {k}/{len(meta)}  {rate:.2f} clips/s  ETA {(len(meta) - k) / rate / 60:.1f} min")
        ext.close()
        lens = np.array([x.shape[1] for x in segs])
        arr = np.zeros((len(segs), segs[0].shape[0], lens.max(), segs[0].shape[2]), np.float16)
        for i, x in enumerate(segs):
            arr[i, :, : x.shape[1]] = x
        np.save(out, arr)
        np.save(FEAT / f"seg_{key}_lens.npy", lens)
        log(f"segments {key}: saved {arr.shape}")
    log("segments done")


if __name__ == "__main__":
    main()
