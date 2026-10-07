"""Frozen-model embedding extraction (speaker embeddings and encoder layers).

Every extractor returns one row per clip. Results are cached as .npy by the
notebooks, so the GPU work runs once.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .audio import SR, load_audio


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def speaker_embeddings(paths, max_seconds: float = 30.0,
                       model_name: str = "microsoft/wavlm-base-plus-sv") -> np.ndarray:
    """x-vector speaker embeddings (L2-normalised), used for pseudo-speakers.

    The model was trained for speaker verification, so its embedding space
    captures voice identity rather than proficiency.
    """
    from transformers import AutoFeatureExtractor, WavLMForXVector

    dev = device()
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name).to(dev).eval()
    out = []
    for p in tqdm(paths, desc="speaker emb"):
        wav = load_audio(p)[: int(max_seconds * SR)]
        inputs = fe(wav, sampling_rate=SR, return_tensors="pt").to(dev)
        emb = model(**inputs).embeddings
        out.append(torch.nn.functional.normalize(emb, dim=-1)[0].float().cpu().numpy())
    return np.vstack(out)


# --------------------------------------------------------------------------
# Layer-wise pooled statistics from frozen speech encoders
# --------------------------------------------------------------------------
ENCODERS = {
    # key: (Hugging Face id, family)
    "whisper_v3": ("openai/whisper-large-v3", "whisper"),
    "wavlm_large": ("microsoft/wavlm-large", "ssl"),
    "w2vbert2": ("facebook/w2v-bert-2.0", "w2vbert"),
    "hubert_large": ("facebook/hubert-large-ll60k", "ssl"),
}


def speech_mask(wav: np.ndarray, n_frames: int, floor_db: float = 35.0) -> np.ndarray:
    """Energy-based voice activity per encoder frame.

    The waveform is split into ``n_frames`` equal chunks (one per encoder
    frame). Frames more than ``floor_db`` below the loudest one count as
    silence. Pooling over speech only keeps long pauses from diluting the
    statistics; pause behaviour is measured separately by fluency features.
    """
    chunks = np.array_split(wav[: len(wav) - len(wav) % max(n_frames, 1)] if len(wav) >= n_frames else wav, n_frames)
    db = np.array([10 * np.log10(np.mean(c ** 2) + 1e-10) if len(c) else -100 for c in chunks])
    mask = db > db.max() - floor_db
    return mask if mask.sum() >= 10 else np.ones(n_frames, bool)


class LayerStats:
    """Frozen encoder -> per-layer [mean, std] over speech frames.

    ``__call__`` returns an array of shape (n_layers, 2 * hidden) in float16.
    """

    def __init__(self, key: str, dtype=torch.float16):
        from transformers import AutoFeatureExtractor, AutoModel

        self.key = key
        self.name, self.family = ENCODERS[key]
        self.dev, self.dtype = device(), dtype
        self.fe = AutoFeatureExtractor.from_pretrained(self.name)
        if self.family == "whisper":
            from transformers import WhisperModel

            full = WhisperModel.from_pretrained(self.name, dtype=dtype)
            self.model = full.get_encoder().to(self.dev).eval()
            del full
        else:
            self.model = AutoModel.from_pretrained(self.name, dtype=dtype).to(self.dev).eval()

    @torch.inference_mode()
    def hidden_states(self, wav: np.ndarray) -> torch.Tensor:
        """All layers' frame states, shape (n_layers, T, d), on the GPU."""
        if self.family == "whisper":
            # Whisper sees fixed 30 s windows; encode each window and keep only
            # the frames that cover real audio (50 frames per second).
            win = 30 * SR
            outs = []
            for s in range(0, len(wav), win):
                seg = wav[s:s + win]
                if len(seg) < SR:  # ignore a sub-second tail
                    continue
                feats = self.fe(seg, sampling_rate=SR, return_tensors="pt").input_features
                hs = self.model(feats.to(self.dev, self.dtype), output_hidden_states=True).hidden_states
                keep = int(np.ceil(len(seg) / SR * 50))
                outs.append(torch.stack(hs)[:, 0, :keep])
            return torch.cat(outs, dim=1)
        # w2v-BERT's relative-position attention builds a (T, T, 64) tensor per
        # layer: ~1.2 GB for a 60 s clip, which overflows a 6 GB GPU into
        # system RAM (~10x slower). Encode it in 20 s windows instead, like
        # Whisper's 30 s windows. The other SSL models take the whole clip.
        if self.family == "w2vbert":
            win = 20 * SR
        elif getattr(self, "ssl_window_s", None):    # optional cap, e.g. WavLM segments on 6 GB
            win = int(self.ssl_window_s * SR)
        else:
            win = len(wav)
        outs = []
        for s in range(0, len(wav), win):
            seg = wav[s:s + win]
            if len(seg) < SR and outs:  # drop a sub-second tail
                continue
            inputs = self.fe(seg, sampling_rate=SR, return_tensors="pt")
            inputs = {k: (v.to(self.dev, self.dtype) if v.is_floating_point() else v.to(self.dev))
                      for k, v in inputs.items()}
            hs = self.model(**inputs, output_hidden_states=True).hidden_states
            outs.append(torch.stack(hs)[:, 0])
        return torch.cat(outs, dim=1)

    @staticmethod
    def _pool(h: torch.Tensor, wav: np.ndarray) -> np.ndarray:
        m = torch.from_numpy(speech_mask(wav, h.shape[1])).to(h.device)
        h = h[:, m]
        return torch.cat([h.mean(1), h.std(1)], dim=-1).cpu().numpy().astype(np.float16)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        return self._pool(self.hidden_states(wav).float(), wav)  # (L, 2d)

    def with_crops(self, wav: np.ndarray, crops: list[tuple[float, float]]):
        """One forward pass; stats for the full clip and for each (start_s, len_s) crop.

        Crop stats are pooled from the matching slice of the full-clip frames,
        so a crop keeps context from outside its edges. That is a small
        difference from encoding the crop alone, and ~2.5x cheaper.
        """
        h = self.hidden_states(wav).float()
        T, n = h.shape[1], len(wav)
        full = self._pool(h, wav)
        out = []
        for start, length in crops:
            a, b = int(start * SR), min(int((start + length) * SR), n)
            fa, fb = int(a / n * T), max(int(b / n * T), int(a / n * T) + 1)
            out.append(self._pool(h[:, fa:fb], wav[a:b]))
        return full, out

    def segments(self, wav: np.ndarray, layers: range, seg_s: float = 1.0) -> np.ndarray:
        """Frame states of ``layers`` averaged per ``seg_s`` second.

        Returns (n_layers, n_segments, d) float16: the input of the trained
        layer-adapter heads (``shl.adapter``). Segments keep temporal
        structure at ~1/50 of the frame-level storage.
        """
        h = self.hidden_states(wav)[list(layers)].float()       # (L, T, d)
        T = h.shape[1]
        n_seg = max(1, int(np.ceil(len(wav) / SR / seg_s)))
        edges = np.linspace(0, T, n_seg + 1).round().astype(int)
        segs = [h[:, a:b].mean(1) if b > a else h[:, min(a, T - 1)] for a, b in zip(edges[:-1], edges[1:])]
        return torch.stack(segs, 1).cpu().numpy().astype(np.float16)

    def close(self):
        del self.model
        torch.cuda.empty_cache()
