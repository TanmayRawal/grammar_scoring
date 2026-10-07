"""Audio loading, preprocessing and quality-control features.

Includes the "zero gate": the 37 training clips labelled 0 are noise-masked
recordings with an almost flat loudness envelope (tiny dynamic range) and
near-clipping peaks. A simple rule separates them, so they are scored 0
directly and excluded from regression training.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

SR = 16_000


def load_audio(path, sr: int = SR, trim_db: float | None = 40.0,
               peak_norm: bool = True) -> np.ndarray:
    """Load a clip as float32 mono at ``sr``.

    Optionally trims leading/trailing silence and peak-normalises. The QC
    features must use ``load_raw`` instead, because normalising would hide
    the clipping signature of the noise-masked clips.
    """
    import librosa

    wav = load_raw(path, sr)
    if trim_db is not None:
        trimmed, _ = librosa.effects.trim(wav, top_db=trim_db)
        if len(trimmed) > sr:  # never trim a clip down to nothing
            wav = trimmed
    if peak_norm:
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = 0.95 * wav / peak
    return wav.astype(np.float32)


def load_raw(path, sr: int = SR) -> np.ndarray:
    """Load mono audio resampled to ``sr`` without any other processing."""
    import librosa

    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav


def qc_features(wav: np.ndarray, sr: int = SR) -> dict:
    """Signal-level features used by the zero gate and the EDA."""
    import librosa

    frame, hop = 400, 160  # 25 ms / 10 ms
    rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + 1e-8)
    flat = librosa.feature.spectral_flatness(y=wav, n_fft=512, hop_length=hop)[0]
    return {
        "duration_s": len(wav) / sr,
        "peak": float(np.max(np.abs(wav))),
        # loudness spread between loud and quiet frames: speech is bursty
        # (large range); stationary masking noise is flat (small range)
        "dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)),
        "rms_db_mean": float(rms_db.mean()),
        "spectral_flatness": float(np.mean(flat)),
        "silence_ratio": float(np.mean(rms_db < rms_db.max() - 35)),
    }


def zero_gate(qc, min_flatness: float = 0.40, max_dynamic_range_db: float = 8.0):
    """Return a boolean mask of clips that look noise-masked (score 0).

    The noise-masked batch is both spectrally flat (noise-like, flatness
    0.48-0.53) and stationary (dynamic range <= 5.3 dB). Real speech fails
    at least one condition: the closest scorable clips have flatness <= 0.46
    or dynamic range >= 8 dB, but never both. The public rule (range < 5 dB
    and peak > 0.9) misses 4 of the 37 zeros. See notebook 00, section 3b.
    """
    return (np.asarray(qc["spectral_flatness"]) > min_flatness) & (
        np.asarray(qc["dynamic_range_db"]) < max_dynamic_range_db
    )


def random_crops(wav: np.ndarray, n: int, min_s: float, max_s: float,
                 sr: int = SR, seed: int = 0) -> list[np.ndarray]:
    """Random contiguous crops, used to match the shorter test durations."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        length = int(rng.uniform(min_s, max_s) * sr)
        if length >= len(wav):
            out.append(wav)
            continue
        start = int(rng.integers(0, len(wav) - length))
        out.append(wav[start:start + length])
    return out
