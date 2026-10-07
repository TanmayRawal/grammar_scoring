"""Add another LLM view (e.g. Qwen3-4B in 4-bit) from existing transcripts.

Reads artifacts/text/transcripts.csv (written by extract_text.py), writes
text/llm{tag}_states.npy and text/llm{tag}_rubric.npy, and adds
llm{tag}_rubric_ev / _entropy columns to text/hand.csv. Notebook 04 picks up
every llm*_states.npy file as its own view.

Usage: python tools/extract_llm.py --model Qwen/Qwen3-4B --tag 4b --four-bit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shl import text  # noqa: E402

TXT = ROOT / "artifacts" / "text"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(ROOT / "artifacts" / "extract.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--tag", default="4b")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--source", default="whisper", choices=["whisper", "ctc"],
                    help="transcript: fluent Whisper text or error-preserving CTC text")
    ap.add_argument("--no-rubric", action="store_true")
    a = ap.parse_args()
    tr = pd.read_csv(TXT / "transcripts.csv").fillna("")
    texts = tr[a.source].tolist()
    llm = text.LLMFeatures(a.model, four_bit=a.four_bit)
    kinds = (("states", llm.states),) if a.no_rubric else (("states", llm.states), ("rubric", llm.rubric))
    for kind, fn in kinds:
        path = TXT / f"llm{a.tag}_{kind}.npy"
        if path.exists():
            log(f"llm{a.tag} {kind}: cached")
            continue
        out, t0 = [], time.time()
        for k, t in enumerate(texts, 1):
            out.append(fn(t))
            if k % 250 == 0 or k == len(texts):
                log(f"llm{a.tag} {kind}: {k}/{len(texts)}  ETA {(len(texts) - k) / (k / (time.time() - t0)) / 60:.1f} min")
        np.save(path, np.stack(out))
    del llm
    torch.cuda.empty_cache()
    if a.no_rubric:
        log(f"llm{a.tag} done (states only)")
        return
    rub = np.load(TXT / f"llm{a.tag}_rubric.npy")
    hand = pd.read_csv(TXT / "hand.csv")
    assert len(hand) == len(rub)
    hand[f"llm{a.tag}_rubric_ev"], hand[f"llm{a.tag}_rubric_entropy"] = rub[:, 0], rub[:, 1]
    hand.to_csv(TXT / "hand.csv", index=False)
    log(f"llm{a.tag} done")


if __name__ == "__main__":
    main()
