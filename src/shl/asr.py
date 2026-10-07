"""Speech recognition: two complementary transcript views.

* **Whisper-large-v3-turbo** (segment timestamps): a fluent, punctuated
  transcript, good for sentence-level and LLM features. Whisper's strong
  language model tends to *repair* learner errors, a known blind spot for
  grammar scoring. Word-level timestamps were tried but cost ~150 s per clip
  (cross-attention alignment), so only segment times are used.
* **wav2vec2-large CTC, greedy, no language model**: an acoustic transcript
  that cannot "fix" grammar. Its character offsets give exact **word
  timings** for free (one frame = 20 ms). These drive the pause and fluency
  features and let crop transcripts be cut without re-running ASR.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from .audio import SR

WHISPER = "openai/whisper-large-v3-turbo"
CTC = "facebook/wav2vec2-large-960h-lv60-self"


class WhisperSegments:
    """Whisper transcription returning text plus (segment_text, start_s, end_s)."""

    def __init__(self, model_name: str = WHISPER, batch_size: int = 4):
        from transformers import pipeline

        self.batch_size = batch_size
        self.pipe = pipeline(
            "automatic-speech-recognition", model=model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"attn_implementation": "sdpa"},  # memory-efficient attention
        )

    def __call__(self, wav: np.ndarray) -> dict:
        out = self.pipe(
            {"raw": wav, "sampling_rate": SR}, chunk_length_s=30, batch_size=self.batch_size,
            return_timestamps=True,
            generate_kwargs={"language": "en", "task": "transcribe"},
        )
        segs, last = [], 0.0
        for c in out.get("chunks", []):
            s, e = c["timestamp"]
            s = float(s) if s is not None else last
            e = float(e) if e is not None else len(wav) / SR
            if c["text"].strip():
                segs.append((c["text"].strip(), s, e))
            last = e
        return {"text": out["text"].strip(), "segments": segs}


class CTCTranscriber:
    """Greedy CTC decoding (no LM) with word timings from character offsets."""

    FRAME_S = 0.02  # wav2vec2 output stride: 320 samples at 16 kHz

    def __init__(self, model_name: str = CTC):
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.dev).eval()
        if self.dev == "cuda":
            self.model.half()

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray, win_s: float = 30.0) -> dict:
        words = []
        win = int(win_s * SR)
        for s in range(0, len(wav), win):
            seg = wav[s:s + win]
            if len(seg) < SR // 2:
                continue
            x = self.proc(seg, sampling_rate=SR, return_tensors="pt").input_values.to(self.dev)
            if self.dev == "cuda":
                x = x.half()
            ids = self.model(x).logits.argmax(-1)[0].cpu().numpy()
            dec = self.proc.decode(ids, output_word_offsets=True)
            off = s / SR
            for w in dec.word_offsets:
                words.append((w["word"].lower(), off + w["start_offset"] * self.FRAME_S,
                              off + w["end_offset"] * self.FRAME_S))
        return {"text": " ".join(w[0] for w in words), "words": words}


def slice_words(words: list, start_s: float, end_s: float) -> list:
    """(word, start, end) items whose midpoint falls inside [start_s, end_s)."""
    return [w for w in words if start_s <= 0.5 * (w[1] + w[2]) < end_s]


def words_to_text(words: list) -> str:
    return re.sub(r"\s+", " ", " ".join(w[0] for w in words)).strip()
