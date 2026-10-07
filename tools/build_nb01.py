"""Generate notebooks/01_audio_encoders.ipynb."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# 01 · Frozen speech-encoder features

**Why frozen encoders?** With 732 scorable clips, fine-tuning a 300M–600M parameter model overfits. Public write-ups and the Speak & Improve 2025 papers agree that *frozen* self-supervised / ASR encoders carry the strongest signal for spoken proficiency: OOF RMSE ≈ 0.52–0.57, against ≈ 0.87 for transcript-only models.

**What is extracted:** for each encoder and **every layer**, the mean and standard deviation of the frame states over **speech frames only** (energy VAD). Information sits at different depths: acoustic and prosodic in the lower and middle layers, lexical and syntactic higher up. Notebook 04 therefore picks layer *bands* by grouped CV instead of assuming the last layer.

| key | model | layers | dim | why |
|---|---|---|---|---|
| `whisper_v3` | Whisper-large-v3 encoder | 33 | 1280 | ASR-trained on 5M h of speech; robust to accents. The best single view in public solutions |
| `wavlm_large` | WavLM-large | 25 | 1024 | Denoising SSL; strong on paralinguistics and fluency |
| `w2vbert2` | w2v-BERT 2.0 | 25 | 1024 | 4.5M h, 143 languages: good coverage of L2 accents |
| `hubert_large` | HuBERT-large (LL60k) | 25 | 1024 | Different training objective, so its errors differ (useful in the ensemble) |

**Crops:** test clips are ~45 s and train clips ~60 s. Each train clip therefore also gets **2 random 40–50 s crops**. They serve as augmentation, and as a check that features are not driven by clip length."""),
code("""import os, sys, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
for p in Path("/kaggle/input").glob("*/src"):
    sys.path.insert(0, str(p))
os.environ.setdefault("SHL_DATA", str(ROOT / "shl-hiring-assessment-2026"))
from shl import audio, embed

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT / "artifacts"
FEAT = OUT / "features"; FEAT.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(OUT / "meta.csv")
print(meta.split.value_counts().to_dict(), "| device:", embed.device(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")"""),
md("## Crop plan\nDeterministic: the seed is fixed, so every encoder sees exactly the same crops."),
code("""N_CROPS, CROP_MIN, CROP_MAX = 2, 40.0, 50.0
rng = np.random.default_rng(0)
rows = []
for uid, path, dur in meta.query("split=='train'")[["uid", "path", "duration_s"]].itertuples(index=False):
    for c in range(N_CROPS):
        length = min(rng.uniform(CROP_MIN, CROP_MAX), dur)
        start = rng.uniform(0, max(dur - length, 0))
        rows.append((uid, path, c, round(start, 3), round(length, 3)))
crops = pd.DataFrame(rows, columns=["uid", "path", "crop", "start_s", "len_s"])
crops.to_csv(FEAT / "crops.csv", index=False)
print(len(crops), "crops;", crops.len_s.describe().round(1).to_dict())"""),
md("""## Extraction
Each clip is encoded **once**. Statistics for its crops are pooled from the matching slice of the full-clip frames (`LayerStats.with_crops`), ~2.5× cheaper than encoding each crop again.

Memory notes for a 6 GB laptop GPU, models in fp16:
- **Whisper:** sees fixed 30 s windows.
- **w2v-BERT 2.0:** encoded in 20 s windows. Its relative-position attention builds a (T × T × 64) tensor per layer, ~1.2 GB for a 60 s clip, which overflowed into system RAM and ran ~10× slower.
- **WavLM / HuBERT:** take the whole clip.

The same logic lives in `tools/extract_features.py`, which logs progress/ETA and can resume. On the first local run, Whisper and WavLM crops were made by re-encoding each crop; the other two used frame slicing."""),
code("""def extract(key, ckpt=100):
    f_full, f_crop = FEAT / f"{key}_full.npy", FEAT / f"{key}_crops.npy"
    if f_full.exists() and f_crop.exists():
        print(key, "cached"); return
    t0 = time.time()
    ext = embed.LayerStats(key)
    crop_rows = crops.groupby("uid").indices
    full_out = crop_out = None
    for i, (uid, path) in enumerate(tqdm(meta[["uid", "path"]].itertuples(index=False), total=len(meta), desc=key)):
        rows = crop_rows.get(uid, [])
        full, cr = ext.with_crops(audio.load_audio(path), [(crops.start_s.iloc[r], crops.len_s.iloc[r]) for r in rows])
        if full_out is None:
            full_out = np.zeros((len(meta),) + full.shape, np.float16)
            crop_out = np.zeros((len(crops),) + full.shape, np.float16)
        full_out[i] = full
        for r, c in zip(rows, cr):
            crop_out[r] = c
    np.save(f_full, full_out); np.save(f_crop, crop_out)
    ext.close(); del ext; gc.collect(); torch.cuda.empty_cache()
    print(f"{key}: full {full_out.shape}, crops {crop_out.shape}, {time.time() - t0:.0f}s")

for key in embed.ENCODERS:
    extract(key)"""),
md("## Sanity checks\nNo NaN/inf values. Feature norms should be similar between train and test; a large gap would point to a domain or length effect."),
code("""is_test = (meta.split == "test").values
summary = []
for key in embed.ENCODERS:
    X = np.load(FEAT / f"{key}_full.npy", mmap_mode="r")
    finite = np.isfinite(X).all()
    norms = np.linalg.norm(np.asarray(X[:, -1], np.float32), axis=1)
    summary.append({"encoder": key, "shape": X.shape, "finite": bool(finite),
                    "last-layer norm train": norms[~is_test].mean().round(1), "test": norms[is_test].mean().round(1)})
pd.DataFrame(summary)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "01_audio_encoders.ipynb"
nbf.write(nb, out)
print("wrote", out)
