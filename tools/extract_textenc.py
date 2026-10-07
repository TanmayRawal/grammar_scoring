"""Frozen text-encoder views of the transcripts (full clips + crops).

For each encoder and transcript source, saves the mean-pooled hidden state of
every layer: text/enc_{tag}_{source}_states.npy, shape (n_items, n_layers, d)
float16, rows in text/transcripts.csv order. Notebook 04 picks a layer band
per file. The best public pipeline without leaderboard-derived steps (0.3355)
stacked RoBERTa / DeBERTa / ELECTRA embeddings of transcripts; ELECTRA's
discriminator is trained to spot replaced tokens, which overlaps with
spotting ungrammatical words.

Usage: python tools/extract_textenc.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "artifacts" / "text"
ENCODERS = {
    "deberta": "microsoft/deberta-v3-large",
    "roberta": "FacebookAI/roberta-large",
    # plain local folder: the hub cache's symlinks fail on this machine (WinError 1314)
    "electra": str(Path.home() / "hf_models" / "electra-large-discriminator"),
}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(ROOT / "artifacts" / "extract.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.inference_mode()
def encode(name: str, texts: list[str], batch: int = 16) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(dev).eval()
    if dev == "cuda":
        model.half()
    out = []
    order = np.argsort([len(t) for t in texts])          # length-sorted batches: less padding
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] or "(silence)" for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = torch.stack([(h.float() * m).sum(1) / m.sum(1) for h in hs], 1)   # (B, L, d)
        out.append((idx, pooled.cpu().numpy().astype(np.float16)))
    res = np.zeros((len(texts),) + out[0][1].shape[1:], np.float16)
    for idx, arr in out:
        res[idx] = arr
    del model
    torch.cuda.empty_cache()
    return res


def main():
    tr = pd.read_csv(TXT / "transcripts.csv").fillna("")
    for tag, name in ENCODERS.items():
        for src in ("whisper", "ctc"):
            path = TXT / f"enc_{tag}_{src}_states.npy"
            if path.exists():
                log(f"textenc {tag}/{src}: cached")
                continue
            t0 = time.time()
            np.save(path, encode(name, tr[src].tolist()))
            log(f"textenc {tag}/{src}: done in {(time.time() - t0) / 60:.1f} min")
    log("textenc done")


if __name__ == "__main__":
    main()
