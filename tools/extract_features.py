"""Extract per-layer pooled encoder features with visible progress.

Runs as a plain script, so progress (clips done, clips/s, ETA) is written to
artifacts/extract.log, and it is resumable (checkpoint every ``--ckpt`` clips).
Unlike the first notebook-01 run (used for Whisper and WavLM), each clip is
encoded once and crop statistics are pooled from frame slices
(``LayerStats.with_crops``), about 2.5x faster.

Usage:
    python tools/extract_features.py                 # all encoders
    python tools/extract_features.py --enc wavlm_large
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
from shl import audio, embed  # noqa: E402

OUT = ROOT / "artifacts"
FEAT = OUT / "features"
LOG = OUT / "extract.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def crop_plan(meta: pd.DataFrame, n_crops=2, lo=40.0, hi=50.0) -> pd.DataFrame:
    """Deterministic 40-50 s crops of every train clip (same as notebook 01)."""
    path = FEAT / "crops.csv"
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(0)
    rows = []
    for uid, p, dur in meta.query("split=='train'")[["uid", "path", "duration_s"]].itertuples(index=False):
        for c in range(n_crops):
            length = min(rng.uniform(lo, hi), dur)
            start = rng.uniform(0, max(dur - length, 0))
            rows.append((uid, p, c, round(start, 3), round(length, 3)))
    crops = pd.DataFrame(rows, columns=["uid", "path", "crop", "start_s", "len_s"])
    crops.to_csv(path, index=False)
    return crops


def run_single_pass(ext, meta: pd.DataFrame, crops: pd.DataFrame, key: str, ckpt: int) -> None:
    """One forward pass per clip: full-clip stats plus stats for each crop
    (pooled from the matching frame slice). Checkpointed and resumable."""
    f_full, f_crop = FEAT / f"{key}_full.npy", FEAT / f"{key}_crops.npy"
    p_full, p_crop, p_idx = (FEAT / f"{key}_full.part.npy", FEAT / f"{key}_crops.part.npy",
                             FEAT / f"{key}.part.idx")
    crop_rows = crops.groupby("uid").indices                    # uid -> crop row ids
    full_out = crop_out = None
    start = 0
    if p_idx.exists():
        start = int(p_idx.read_text())
        full_out, crop_out = np.load(p_full), np.load(p_crop)
        log(f"{key}: resuming at {start}/{len(meta)}")
    t0 = time.time()
    for i in range(start, len(meta)):
        uid, path = meta.uid.iloc[i], meta.path.iloc[i]
        rows = crop_rows.get(uid, [])
        wav = audio.load_audio(path)
        full, cr = ext.with_crops(wav, [(crops.start_s.iloc[r], crops.len_s.iloc[r]) for r in rows])
        if full_out is None:
            full_out = np.zeros((len(meta),) + full.shape, np.float16)
            crop_out = np.zeros((len(crops),) + full.shape, np.float16)
        full_out[i] = full
        for r, c in zip(rows, cr):
            crop_out[r] = c
        k = i + 1
        if k % ckpt == 0 or k == len(meta):
            np.save(p_full, full_out); np.save(p_crop, crop_out); p_idx.write_text(str(k))
            rate = (k - start) / (time.time() - t0)
            log(f"{key}: {k}/{len(meta)} clips  {rate:.2f} clips/s  ETA {(len(meta) - k) / max(rate, 1e-9) / 60:.1f} min")
    np.save(f_full, full_out); np.save(f_crop, crop_out)
    for f in (p_full, p_crop, p_idx):
        f.unlink(missing_ok=True)
    log(f"{key}: finished in {(time.time() - t0) / 60:.1f} min")


def reencode_crops(ext, crops: pd.DataFrame, key: str, ckpt: int) -> None:
    """Encode every crop on its own audio (true short context, like a test
    clip) and overwrite {key}_crops.npy. Checkpointed and resumable."""
    out_path = FEAT / f"{key}_crops.npy"
    part, idx = FEAT / f"{key}_crops_re.part.npy", FEAT / f"{key}_crops_re.part.idx"
    start, out = 0, None
    if idx.exists():
        start, out = int(idx.read_text()), np.load(part)
        log(f"{key} re-encode: resuming at {start}/{len(crops)}")
    t0 = time.time()
    for i in range(start, len(crops)):
        r = crops.iloc[i]
        wav = audio.load_audio(r.path)
        a = int(r.start_s * audio.SR)
        feat = ext(wav[a:a + int(r.len_s * audio.SR)])
        if out is None:
            out = np.zeros((len(crops),) + feat.shape, np.float16)
        out[i] = feat
        k = i + 1
        if k % ckpt == 0 or k == len(crops):
            np.save(part, out); idx.write_text(str(k))
            rate = (k - start) / (time.time() - t0)
            log(f"{key} re-encode: {k}/{len(crops)}  {rate:.2f} crops/s  ETA {(len(crops) - k) / max(rate, 1e-9) / 60:.1f} min")
    np.save(out_path, out)
    part.unlink(missing_ok=True); idx.unlink(missing_ok=True)
    log(f"{key} re-encode: finished in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", nargs="*", default=list(embed.ENCODERS))
    ap.add_argument("--ckpt", type=int, default=50)
    ap.add_argument("--reencode-crops", action="store_true",
                    help="re-encode crops of already-extracted encoders on their own audio")
    args = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(OUT / "meta.csv")
    crops = crop_plan(meta)
    log(f"start: {len(meta)} clips + {len(crops)} crops, encoders {args.enc}, device {embed.device()}")
    if args.reencode_crops:
        for key in args.enc:
            ext = embed.LayerStats(key)
            reencode_crops(ext, crops, key, args.ckpt)
            ext.close()
        log("all done")
        return
    for key in args.enc:
        if (FEAT / f"{key}_full.npy").exists() and (FEAT / f"{key}_crops.npy").exists():
            log(f"{key}: cached, skipping")
            continue
        t = time.time()
        ext = embed.LayerStats(key)
        log(f"{key}: model loaded in {time.time() - t:.0f}s")
        run_single_pass(ext, meta, crops, key, args.ckpt)
        ext.close()
    log("all done")


if __name__ == "__main__":
    main()
