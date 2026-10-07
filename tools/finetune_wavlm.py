"""End-to-end fine-tuning of WavLM-base+ as a grammar-score regressor.

Recipe from a public 0.3351 solution (SakashSrivastava/Grammar-Scoring-Engine,
kaggle/finetune_wavlm.py), adapted to this project:
* folds: our speaker + prompt held-out folds (artifacts/folds_joint.csv, seed 0)
  instead of plain stratified K-fold, so the OOF score is comparable to our other heads;
* noise-masked clips: our zero gate (they are excluded from training as before).

Model: WavLM encoder (CNN front-end frozen) -> softmax-weighted mix of all
layers -> mean over time -> linear score. Trained on random 15 s crops (MSE on
standardised targets, AdamW, cosine schedule with warm-up, fp16). A clip is
predicted as the mean over consecutive 15 s windows.

Output: artifacts/preds/wavlm_ft__ft.npz with the same keys as the other heads.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import WavLMModel, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SHL_DATA", str(ROOT / "shl-hiring-assessment-2026"))
from shl import audio  # noqa: E402

OUT = ROOT / "artifacts"
MODEL = "microsoft/wavlm-base-plus"
SEED, EPOCHS, BS, CROP, SR = 42, 15, 8, 15, audio.SR
MICRO = 4   # 6 GB GPU: micro-batches of 4 with gradient accumulation (effective batch 8)
LR_ENC, LR_HEAD = 3e-5, 1e-3
MEAN, STD = 3.5, 1.0


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "extract.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(MODEL, layerdrop=0.0, dtype=torch.float32)   # fp32 master weights
        self.wavlm.feature_extractor._freeze_parameters()
        self.layer_w = torch.nn.Parameter(torch.zeros(self.wavlm.config.num_hidden_layers + 1))
        self.head = torch.nn.Linear(self.wavlm.config.hidden_size, 1)

    def forward(self, x):
        hs = self.wavlm(x, output_hidden_states=True).hidden_states
        w = self.layer_w.softmax(0)
        h = sum(w[i] * hs[i].mean(1) for i in range(len(hs)))   # pool each layer first: far less memory
        return self.head(h).squeeze(-1)


def norm(w):
    return (w - w.mean()) / (w.std() + 1e-7)


def crop(w, rng):
    n = CROP * SR
    if len(w) <= n:
        return np.pad(w, (0, n - len(w)))
    s = rng.integers(0, len(w) - n)
    return w[s:s + n]


@torch.no_grad()
def predict(model, wavs):
    model.eval()
    n, out = CROP * SR, []
    for w in wavs:
        chunks = [w[s:s + n] for s in range(0, max(len(w) - n // 2, 1), n)]
        x = torch.tensor(np.stack([np.pad(c, (0, n - len(c))) for c in chunks])).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.cat([model(x[i:i + MICRO]).float() for i in range(0, len(x), MICRO)]).mean().item())
    return np.array(out) * STD + MEAN


def main():
    meta = pd.read_csv(OUT / "meta.csv")
    folds = pd.read_csv(OUT / "folds_joint.csv")
    row = {u: i for i, u in enumerate(meta.uid)}
    fit = np.array([row[u] for u in folds.uid])
    test = np.where(meta.split == "test")[0]
    y = meta.label.values[fit]
    fold = folds.fold_s0.values
    log("wavlm-ft: loading audio")
    wav_tr = [norm(audio.load_raw(meta.path.iloc[i])) for i in fit]
    wav_te = [norm(audio.load_raw(meta.path.iloc[i])) for i in test]
    oof, te, ins = np.zeros(len(y)), np.zeros(len(test)), np.zeros(len(y))
    for k in range(5):
        t0 = time.time()
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]
        torch.manual_seed(SEED + k)
        rng = np.random.default_rng(SEED + k)
        model = Regressor().cuda()
        head = [model.layer_w, *model.head.parameters()]
        enc = [p for p in model.wavlm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENC}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
        steps = (len(a) // BS) * EPOCHS
        sch = get_cosine_schedule_with_warmup(opt, steps // 10, steps)
        scaler = torch.amp.GradScaler("cuda")
        tgt = (y - MEAN) / STD
        for ep in range(EPOCHS):
            model.train()
            perm = rng.permutation(a)
            for i in range(0, len(perm) - BS + 1, BS):
                idx = perm[i:i + BS]
                for m in range(0, BS, MICRO):
                    sub = idx[m:m + MICRO]
                    x = torch.tensor(np.stack([crop(wav_tr[j], rng) for j in sub])).cuda()
                    t = torch.tensor(tgt[sub], dtype=torch.float32).cuda()
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred = model(x).float()
                    loss = torch.nn.functional.mse_loss(pred, t) * len(sub) / BS
                    scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
        oof[b] = predict(model, [wav_tr[j] for j in b])
        te += predict(model, wav_te) / 5
        ins += predict(model, wav_tr) / 5
        log(f"wavlm-ft fold {k}: RMSE {np.sqrt(np.mean((oof[b] - y[b]) ** 2)):.4f} ({(time.time() - t0) / 60:.1f} min)")
        del model, opt
        torch.cuda.empty_cache()
    r = np.corrcoef(oof, y)[0, 1]
    log(f"wavlm-ft OOF RMSE {np.sqrt(np.mean((oof - y) ** 2)):.4f} r {r:.4f}")
    np.savez(OUT / "preds" / "wavlm_ft__ft.npz", oof=oof, oof_per_seed=oof[None], test=te, insample=ins,
             oof_short=oof, oof_short_per_seed=oof[None])


if __name__ == "__main__":
    main()
