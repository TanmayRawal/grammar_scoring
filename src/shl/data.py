"""Locating the competition data and building one metadata table.

Works both locally (set the ``SHL_DATA`` environment variable to the folder
holding ``train.csv``) and on Kaggle (searches ``/kaggle/input``).

Train and test reuse some file names for *different* recordings, so every
clip is keyed by ``uid = f"{split}/{file}"``, never by file name alone.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CANDIDATE_ROOTS = [
    os.environ.get("SHL_DATA", ""),
    "/kaggle/input/shl-hiring-assessment-2026",
    "/kaggle/input",
]


def find_data_dir() -> Path:
    """Return the folder that contains train.csv (searched recursively)."""
    for root in filter(None, CANDIDATE_ROOTS):
        root = Path(root)
        if not root.exists():
            continue
        hits = sorted(root.rglob("train.csv"), key=lambda p: len(p.parts))
        if hits:
            return hits[0].parent
    raise FileNotFoundError("train.csv not found; set the SHL_DATA environment variable")


def _id_and_label_cols(df: pd.DataFrame) -> tuple[str, str | None]:
    """Detect the file-name column and the label column of a CSV."""
    id_col = next((c for c in df.columns
                   if df[c].astype(str).str.contains(r"\.wav$|audio", case=False).any()
                   or c.lower() in {"filename", "file_name", "file", "id", "audio"}),
                  df.columns[0])
    label_col = next((c for c in df.columns
                      if c != id_col and pd.api.types.is_numeric_dtype(df[c])), None)
    return id_col, label_col


def _resolve_audio(data_dir: Path, split: str, name: str) -> Path:
    name = str(name)
    fname = name if name.lower().endswith(".wav") else f"{name}.wav"
    for cand in (data_dir / split / fname, data_dir / "audios" / split / fname,
                 data_dir / f"{split}_audio" / fname, data_dir / split / "audio" / fname):
        if cand.exists():
            return cand
    hits = [p for p in data_dir.rglob(fname) if split in str(p.parent).lower()]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"{split}/{fname} not found under {data_dir}")


def load_metadata(data_dir: Path | None = None) -> pd.DataFrame:
    """One row per clip: uid, split, file, path, label (NaN for test)."""
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(data_dir / f"{split}.csv")
        id_col, label_col = _id_and_label_cols(df)
        out = pd.DataFrame({"split": split, "file": df[id_col].astype(str)})
        # test.csv ships with random labels, so they are dropped deliberately
        out["label"] = df[label_col].astype(float) if (split == "train" and label_col) else float("nan")
        out["path"] = [str(_resolve_audio(data_dir, split, f)) for f in out.file]
        frames.append(out)
    meta = pd.concat(frames, ignore_index=True)
    meta.insert(0, "uid", meta.split + "/" + meta.file)
    assert meta.uid.is_unique
    return meta


def submission_frame(data_dir: Path, test_files, preds) -> pd.DataFrame:
    """Build the submission from test.csv (not sample_submission.csv, which
    reportedly lists stale IDs), using the sample's column names."""
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    test = pd.read_csv(data_dir / "test.csv")
    id_col, _ = _id_and_label_cols(test)
    sub = pd.DataFrame({sample.columns[0]: test[id_col].values})
    pred_map = dict(zip(map(str, test_files), preds))
    sub[sample.columns[1]] = sub[sample.columns[0]].astype(str).map(pred_map)
    assert sub[sample.columns[1]].notna().all(), "missing predictions"
    assert len(sub) == len(test)
    return sub
