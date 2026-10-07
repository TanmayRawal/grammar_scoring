"""Transcribe all clips and build transcript-based features (full clips + crops).

Steps (each cached, so the script can be re-run after a failure):
  1. wav2vec2 CTC (no LM) on every full clip -> error-preserving words + timings
  2. Whisper-turbo on every full clip -> punctuated text + segment times
  3. crop transcripts = CTC words / Whisper segments inside the crop window
  4. fluency features (CPU), cross-ASR disagreement
  5. GEC edit rates (Whisper text and CTC text), CoLA acceptability
  6. Qwen3 layer states + rubric expected score
Outputs go to artifacts/text/; progress goes to artifacts/extract.log.
Run alone on the GPU: sharing a 6 GB card with another job spills into
system RAM and slows both jobs ~10x.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("SHL_DATA", str(ROOT / "shl-hiring-assessment-2026"))
from shl import asr, audio, text  # noqa: E402

OUT = ROOT / "artifacts"
TXT = OUT / "text"
LOG = OUT / "extract.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free():
    gc.collect()
    torch.cuda.empty_cache()


def run_asr(meta, name, make_model, every=100):
    """Run an ASR model over every full clip; cache results as JSON lines."""
    path = TXT / f"asr_{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["uid"]] = r
    todo = meta[~meta.uid.isin(done)]
    if len(todo) == 0:
        log(f"{name}: cached")
        return done
    model = make_model()
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for k, (uid, p) in enumerate(todo[["uid", "path"]].itertuples(index=False), 1):
            wav = audio.load_audio(p)
            r = model(wav)
            r.update(uid=uid, duration_s=len(wav) / audio.SR)
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[uid] = r
            if k % every == 0 or k == len(todo):
                rate = k / (time.time() - t0)
                log(f"{name}: {len(done)}/{len(meta)}  {rate:.2f} clips/s  ETA {(len(todo) - k) / rate / 60:.1f} min")
    del model
    free()
    return done


def build_items(meta, crops, ctc, wh):
    """One row per text item: every full clip, then every crop (crops.csv order)."""
    items = []
    for uid in meta.uid:
        c, w = ctc[uid], wh[uid]
        items.append(dict(kind="full", uid=uid, crop_row=-1, duration_s=c["duration_s"],
                          text=w["text"], ctc_words=c["words"], ctc_text=c["text"]))
    for i, cr in crops.iterrows():
        a, b = cr.start_s, cr.start_s + cr.len_s
        cw = asr.slice_words(ctc[cr.uid]["words"], a, b)
        segs = [s for s in wh[cr.uid]["segments"] if a <= 0.5 * (s[1] + s[2]) < b]
        items.append(dict(kind="crop", uid=cr.uid, crop_row=i, duration_s=cr.len_s,
                          text=" ".join(s[0] for s in segs), ctc_words=cw, ctc_text=asr.words_to_text(cw)))
    return items


def main():
    import jiwer

    TXT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(OUT / "meta.csv")
    crops = pd.read_csv(OUT / "features" / "crops.csv")
    log(f"text pipeline start: {len(meta)} clips, {len(crops)} crops")

    ctc = run_asr(meta, "ctc", asr.CTCTranscriber)
    wh = run_asr(meta, "whisper", lambda: asr.WhisperSegments(batch_size=4), every=50)
    items = build_items(meta, crops, ctc, wh)
    texts = [it["text"] for it in items]
    ctc_texts = [it["ctc_text"] for it in items]

    # pauses and rates from CTC word timings; sentence stats from Whisper punctuation
    feats = pd.DataFrame([text.fluency_features(it["ctc_words"], it["text"], it["duration_s"]) for it in items])
    norm = lambda s: " ".join(text.tokens(s)) or "x"
    feats["asr_disagreement"] = [min(jiwer.wer(norm(a), norm(b)), 2.0) for a, b in zip(texts, ctc_texts)]
    feats["ctc_words_per_min"] = [len(text.tokens(c)) / max(it["duration_s"], 1) * 60 for c, it in zip(ctc_texts, items)]
    log("fluency features done")

    nn_path = TXT / "nn_feats.csv"
    if nn_path.exists():
        nn = pd.read_csv(nn_path)
    else:
        nn = pd.DataFrame(index=range(len(items)))
        gec = text.GEC()
        nn["gec_whisper"] = gec.edit_rates(texts, punctuated=True); log("GEC (whisper) done")
        nn["gec_ctc"] = gec.edit_rates(ctc_texts, punctuated=False); log("GEC (ctc) done")
        del gec; free()
        cola = text.CoLA()
        nn[["cola_mean", "cola_min", "cola_bad_frac"]] = cola.scores(texts); log("CoLA (whisper) done")
        nn[["cola_ctc_mean", "cola_ctc_min", "cola_ctc_bad_frac"]] = cola.scores(ctc_texts); log("CoLA (ctc) done")
        del cola; free()
        nn.to_csv(nn_path, index=False)
    feats = pd.concat([feats, nn], axis=1)

    llm_path, rub_path = TXT / "llm_states.npy", TXT / "llm_rubric.npy"
    if not (llm_path.exists() and rub_path.exists()):
        llm = text.LLMFeatures()
        for path, fn, label in ((llm_path, llm.states, "llm states"), (rub_path, llm.rubric, "llm rubric")):
            if path.exists():
                continue
            out, t0 = [], time.time()
            for k, t in enumerate(texts, 1):
                out.append(fn(t))
                if k % 250 == 0 or k == len(texts):
                    log(f"{label}: {k}/{len(texts)}  ETA {(len(texts) - k) / (k / (time.time() - t0)) / 60:.1f} min")
            np.save(path, np.stack(out))
        del llm; free()
    rub = np.load(TXT / "llm_rubric.npy")
    feats["llm_rubric_ev"], feats["llm_rubric_entropy"] = rub[:, 0], rub[:, 1]

    index = pd.DataFrame([{k: it[k] for k in ("kind", "uid", "crop_row", "duration_s")} for it in items])
    pd.concat([index, feats], axis=1).to_csv(TXT / "hand.csv", index=False)
    pd.DataFrame({"uid": index.uid, "kind": index.kind, "crop_row": index.crop_row,
                  "whisper": texts, "ctc": ctc_texts}).to_csv(TXT / "transcripts.csv", index=False)
    log("text pipeline done")


if __name__ == "__main__":
    main()
