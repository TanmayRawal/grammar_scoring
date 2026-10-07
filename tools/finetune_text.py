"""Fine-tune DeBERTa-v3-base as a transcript -> grammar-score regressor.

Recipe from a public 0.3351 solution (SakashSrivastava/Grammar-Scoring-Engine,
src/finetune_text.py: max 256 tokens, 6 epochs, lr 3e-5, batch 8, MSE on
standardised targets, cosine schedule), adapted to this project:
* folds: our speaker + prompt held-out folds (folds_joint.csv), 3 fold seeds;
* transcript source: Whisper (fluent) or wav2vec2-CTC (keeps learner errors).

Usage: python tools/finetune_text.py whisper|ctc
Output: artifacts/preds/text_debft_{source}__ft.npz (same keys as the other heads).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
MODEL = "microsoft/deberta-v3-base"
EPOCHS, LR, BS, MAXLEN = 6, 3e-5, 8, 256
MICRO = 4   # 6 GB GPU: micro-batches + gradient checkpointing (effective batch stays 8)
MEAN, STD = 3.5, 1.0


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "extract.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.no_grad()
def predict(model, enc):
    model.eval()
    out = []
    for i in range(0, len(enc["input_ids"]), 32):
        b = {k: v[i:i + 32].cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**b).logits.float().squeeze(-1).cpu())
    return torch.cat(out).numpy() * STD + MEAN


def main(source: str):
    meta = pd.read_csv(OUT / "meta.csv")
    folds = pd.read_csv(OUT / "folds_joint.csv")
    tr = pd.read_csv(OUT / "text" / "transcripts.csv").fillna("")
    full = tr[tr.kind == "full"].set_index("uid")
    X = full.loc[folds.uid, source].values
    Xt = full.loc[meta.uid[meta.split == "test"], source].values
    y = meta.set_index("uid").label.loc[folds.uid].values
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = lambda t: dict(tok(list(t), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt"))
    E, Et = enc(X), enc(Xt)
    cols = [c for c in folds.columns if c.startswith("fold_")][:1]   # 1 fold seed: ~3.5 min/fold on the laptop GPU
    oofs, te, ins = [], np.zeros(len(Xt)), np.zeros(len(y))
    for s, col in enumerate(cols):
        oof = np.zeros(len(y))
        for k in range(5):
            t0 = time.time()
            a, b = np.where(folds[col] != k)[0], np.where(folds[col] == k)[0]
            torch.manual_seed(100 * s + k)
            # dtype=float32 is essential: transformers 5 otherwise keeps the checkpoint's fp16
            # weights, and AdamW on fp16 weights turns every parameter into NaN after one step
            model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.float32).cuda()
            model.gradient_checkpointing_enable()   # without it activations spill out of 6 GB VRAM (~4x slower)
            tgt = torch.tensor((y - MEAN) / STD, dtype=torch.float32)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            n_steps = (len(a) + BS - 1) // BS
            sch = get_cosine_schedule_with_warmup(opt, n_steps // 2, n_steps * EPOCHS)
            g = torch.Generator().manual_seed(100 * s + k)
            for ep in range(EPOCHS):
                model.train()
                perm = a[torch.randperm(len(a), generator=g).numpy()]
                for i in range(0, len(perm), BS):
                    idx = perm[i:i + BS]
                    for m in range(0, len(idx), MICRO):
                        sub = idx[m:m + MICRO]
                        batch = {kk: v[sub].cuda() for kk, v in E.items()}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = model(**batch).logits.float().squeeze(-1)
                        loss = torch.nn.functional.mse_loss(pred, tgt[sub].cuda()) * len(sub) / len(idx)
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad()
            sub = {kk: v[b] for kk, v in E.items()}
            oof[b] = predict(model, sub)
            te += predict(model, Et) / (5 * len(cols))
            ins += predict(model, E) / (5 * len(cols))
            log(f"debft/{source} {col} fold {k}: RMSE {np.sqrt(np.mean((oof[b] - y[b]) ** 2)):.4f} ({(time.time() - t0) / 60:.1f} min)")
            del model, opt
            torch.cuda.empty_cache()
        oofs.append(oof)
    oofs = np.vstack(oofs)
    o = oofs.mean(0)
    log(f"debft/{source} OOF RMSE {np.sqrt(np.mean((o - y) ** 2)):.4f} r {np.corrcoef(o, y)[0, 1]:.4f}")
    np.savez(OUT / "preds" / f"text_debft_{source}__ft.npz", oof=o, oof_per_seed=oofs, test=te, insample=ins,
             oof_short=o, oof_short_per_seed=oofs)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "whisper")
