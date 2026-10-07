"""Trained layer-adapter heads on frozen encoder segment states.

Design follows the Speak & Improve 2025 winner (Cai et al., "perezoso"): a
frozen speech encoder, a small adapter per layer, concatenation, temporal
mean+std pooling, then a linear output. It has ~0.6M trainable parameters,
so it is safe to train on 732 clips, unlike fine-tuning the encoder.

Adapted to this competition:
* **Input:** 1-second segment means of a 4-layer band (``LayerStats.segments``).
* **Duration shift:** every training step sees a random 38–52 s window, the
  test clips' length (~45 s). The model never learns 60 s statistics.
* **Loss:** MSE + λ·(1 − Pearson over the batch), aimed at both halves of
  the metric. Pearson ignores the shrinkage that MSE alone pushes toward.
* **Early stopping:** on a held-out 15% of the *training speakers*,
  evaluated on centred 45 s windows. The outer validation fold is never
  touched.
* **Test mixture:** with importance weights ``w`` (``shl.population``),
  each epoch samples training clips with probability proportional to ``w``,
  and early stopping uses the test-weighted composite.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from .folds import iter_folds
from .metrics import composite


@dataclass
class AdapterConfig:
    hidden: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.05
    epochs: int = 60
    batch: int = 32
    pearson_weight: float = 0.5
    win_min_s: int = 38
    win_max_s: int = 52
    eval_win_s: int = 45
    patience: int = 12
    model_seeds: int = 1   # models per fold (different init / windows / early-stop split), averaged


class AdapterHead(nn.Module):
    def __init__(self, n_layers: int, d: int, cfg: AdapterConfig, y_mean: float):
        super().__init__()
        h = cfg.hidden
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.adapt = nn.ModuleList([
            nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h))
            for _ in range(n_layers)])
        self.mix = nn.Sequential(nn.Linear(n_layers * h, h), nn.GELU())
        self.out = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(2 * h, 1))
        nn.init.zeros_(self.out[1].weight)
        nn.init.constant_(self.out[1].bias, y_mean)

    def forward(self, x, mask):
        # x: (B, L, T, d); mask: (B, T) True for real segments
        z = torch.cat([a(n(x[:, i])) for i, (n, a) in enumerate(zip(self.norm, self.adapt))], -1)
        z = self.mix(z)                                       # (B, T, h)
        m = mask.unsqueeze(-1).float()
        cnt = m.sum(1).clamp(min=1)
        mu = (z * m).sum(1) / cnt
        sd = (((z - mu.unsqueeze(1)) ** 2 * m).sum(1) / cnt).clamp(min=1e-6).sqrt()
        return self.out(torch.cat([mu, sd], -1)).squeeze(-1)


def _pearson_loss(pred, y):
    p, t = pred - pred.mean(), y - y.mean()
    return 1 - (p * t).sum() / (p.norm() * t.norm() + 1e-8)


def _windows(S, lens, idx, rng, cfg, fixed=None):
    """Batch of windows: random 38-52 s (training) or fixed (start, len) pairs."""
    xs, ms = [], []
    for k, i in enumerate(idx):
        n = int(lens[i])
        if fixed is not None:
            a, w = fixed[k]
        else:
            w = int(rng.integers(cfg.win_min_s, cfg.win_max_s + 1))
            a = int(rng.integers(0, max(n - w, 0) + 1))
        a, b = min(a, max(n - 1, 0)), min(a + w, n)
        xs.append(S[i, :, a:b])
        ms.append(b - a)
    T = max(ms)
    X = np.zeros((len(idx), S.shape[1], T, S.shape[3]), np.float32)
    M = np.zeros((len(idx), T), bool)
    for k, (x, m) in enumerate(zip(xs, ms)):
        X[k, :, :m] = x
        M[k, :m] = True
    return torch.from_numpy(X), torch.from_numpy(M)


def _centre(lens, idx, w):
    return [(max(0, (int(lens[i]) - w) // 2), w) for i in idx]


@torch.no_grad()
def _predict(model, S, lens, idx, fixed, dev, bs=64):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        X, M = _windows(S, lens, idx[s:s + bs], None, None, fixed=fixed[s:s + bs])
        out.append(model(X.to(dev), M.to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_one(S, lens, y, tr, groups, cfg, seed, dev, w=None):
    """Train on clips ``tr`` with speaker-grouped early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    fit_i, es_i = next(gss.split(tr, groups=groups[tr]))
    fit, es = tr[fit_i], tr[es_i]
    model = AdapterHead(S.shape[1], S.shape[3], cfg, float(y[fit].mean())).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    yt = torch.tensor(y, dtype=torch.float32)
    es_fixed = _centre(lens, es, cfg.eval_win_s)
    best, best_state, bad = np.inf, None, 0
    if w is not None:
        p_fit = w[fit] / w[fit].sum()
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(fit) if w is None else rng.choice(fit, size=len(fit), replace=True, p=p_fit)
        for s in range(0, len(order), cfg.batch):
            b = order[s:s + cfg.batch]
            if len(b) < 4:
                continue
            X, M = _windows(S, lens, b, rng, cfg)
            pred = model(X.to(dev), M.to(dev))
            tgt = yt[b].to(dev)
            loss = nn.functional.mse_loss(pred, tgt) + cfg.pearson_weight * _pearson_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        score = composite(y[es], _predict(model, S, lens, es, es_fixed, dev), None if w is None else w[es])
        if score < best - 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def run_adapter_cv(S, lens, y, groups, folds: pd.DataFrame, S_test, lens_test,
                   crop_src, crop_win, cfg: AdapterConfig | None = None, log=print, w=None) -> dict:
    """Repeated grouped CV with the same outputs as ``heads.CVResult``.

    ``S``: (n, L, T, d) segment states of the scorable train clips (fold order).
    ``crop_src`` / ``crop_win``: source clip and (start_s, len_s) of each crop,
    used for the test-like short-clip OOF, exactly as for the other heads.
    """
    cfg = cfg or AdapterConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(y)
    oofs, shorts, tests, ins = [], [], [], []
    for si, col in enumerate(folds.columns):
        oof, short = np.zeros(n), np.full(n, np.nan)
        for k, (tr, va) in enumerate(iter_folds(folds, col)):
            m = np.isin(crop_src, va)
            o_va, o_cr, o_te, o_in = [], [], [], []
            for ms in range(cfg.model_seeds):          # bag several models per fold
                model, es_score, n_ep = train_one(S, lens, y, tr, groups, cfg,
                                                  seed=1000 * ms + 100 * si + k, dev=dev, w=w)
                o_va.append(_predict(model, S, lens, va, [(0, int(lens[i])) for i in va], dev))
                o_cr.append(_predict(model, S, lens, crop_src[m], [tuple(c) for c in crop_win[m]], dev))
                o_te.append(_predict(model, S_test, lens_test, np.arange(len(lens_test)),
                                     [(0, int(l)) for l in lens_test], dev))
                o_in.append(_predict(model, S, lens, np.arange(n), [(0, int(l)) for l in lens], dev))
                del model
                torch.cuda.empty_cache()
            oof[va] = np.mean(o_va, 0)
            tot, cnt = np.zeros(n), np.zeros(n)
            np.add.at(tot, crop_src[m], np.mean(o_cr, 0))
            np.add.at(cnt, crop_src[m], 1)
            short[va] = (tot / np.maximum(cnt, 1))[va]
            tests.append(np.mean(o_te, 0))
            ins.append(np.mean(o_in, 0))
        oofs.append(oof)
        shorts.append(short)
        log(f"  {col}: full {composite(y, oof):.4f} | test-weighted {composite(y, oof, w):.4f}")
    oofs, shorts = np.vstack(oofs), np.vstack(shorts)
    return dict(oof=oofs.mean(0), oof_per_seed=oofs, oof_short=shorts.mean(0), oof_short_per_seed=shorts,
                test=np.mean(tests, 0), insample=np.mean(ins, 0))
