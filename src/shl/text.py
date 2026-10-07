"""Transcript-based features: fluency, grammar-error proxies and LLM views.

All count-like quantities are expressed as **rates** (per word, per minute)
because test clips are ~45 s and train clips ~60 s; absolute counts would
encode clip length rather than proficiency.
"""
from __future__ import annotations

import re

import numpy as np
import torch

FILLERS = {"uh", "um", "er", "ah", "erm", "hmm", "mm", "uhm"}
SUBORDINATORS = {"because", "although", "though", "which", "that", "when", "while", "if",
                 "since", "unless", "whereas", "who", "whom", "whose", "where", "whether",
                 "however", "therefore", "moreover", "until", "before", "after"}
GEC_MODEL = "vennify/t5-base-grammar-correction"
COLA_MODEL = "textattack/roberta-base-CoLA"
LLM_MODEL = "Qwen/Qwen3-1.7B"

RUBRIC = """Grammar score rubric for a spoken English answer:
1 = struggles with sentence structure and syntax; limited control of simple structures; memorized patterns.
2 = limited understanding of sentence structure; consistent basic mistakes; may leave sentences incomplete.
3 = decent grasp of sentence structure but errors in grammar, or decent grammar but errors in syntax.
4 = strong understanding of sentence structure; good control of grammar; occasional minor errors.
5 = high grammatical accuracy and adept control of complex grammar; seldom noticeable mistakes."""


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_words(text: str, size: int = 15) -> list[str]:
    """Fixed-size word windows, used when a transcript has no punctuation (CTC)."""
    t = text.split()
    return [" ".join(t[i:i + size]) for i in range(0, len(t), size)] or [""]


def edit_distance(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def fluency_features(words: list, text: str, duration_s: float) -> dict:
    """Rate-based fluency and complexity features from timestamped words."""
    toks = tokens(text)
    n = max(len(toks), 1)
    minutes = max(duration_s, 1.0) / 60
    starts = np.array([w[1] for w in words]) if words else np.zeros(0)
    ends = np.array([w[2] for w in words]) if words else np.zeros(0)
    gaps = starts[1:] - ends[:-1] if len(words) > 1 else np.zeros(0)
    gaps = np.clip(gaps, 0, None)
    spoken = float(np.clip(ends - starts, 0, None).sum()) if len(words) else 0.0
    sents = sentences(text)
    sent_len = [len(tokens(s)) for s in sents] or [0]
    # moving-average type-token ratio (window 25) is length-robust
    win = 25
    mattr = np.mean([len(set(toks[i:i + win])) / win for i in range(max(1, len(toks) - win + 1))]) if len(toks) >= win \
        else len(set(toks)) / n
    rep = sum(a == b for a, b in zip(toks, toks[1:]))
    bigrams = list(zip(toks, toks[1:]))
    return {
        "words_per_min": len(toks) / minutes,
        "articulation_rate": len(words) / max(spoken, 1e-3),            # words per second of speaking
        "pause_per_min": float((gaps > 0.25).sum()) / minutes,
        "long_pause_per_min": float((gaps > 1.0).sum()) / minutes,
        "mean_pause_s": float(gaps[gaps > 0.25].mean()) if (gaps > 0.25).any() else 0.0,
        "pause_time_frac": float(gaps[gaps > 0.25].sum()) / max(duration_s, 1.0),
        "filler_rate": sum(t in FILLERS for t in toks) / n,
        "repeat_rate": rep / n,
        "bigram_repeat_rate": (len(bigrams) - len(set(bigrams))) / max(len(bigrams), 1),
        "mattr": float(mattr),
        "mean_word_len": float(np.mean([len(t) for t in toks])) if toks else 0.0,
        "long_word_rate": sum(len(t) > 6 for t in toks) / n,
        "subordinator_rate": sum(t in SUBORDINATORS for t in toks) / n,
        "mean_sent_len": float(np.mean(sent_len)),
        "max_sent_len": float(np.max(sent_len)),
        "sent_per_min": len(sents) / minutes,
    }


# --------------------------------------------------------------------------
# Neural text scorers (GPU). Each takes a list of texts and returns arrays.
# --------------------------------------------------------------------------
def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class GEC:
    """T5 grammar correction; feature = word edits per word (higher = more errors)."""

    def __init__(self, model_name: str = GEC_MODEL):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_device()).eval()

    @torch.inference_mode()
    def correct(self, segs: list[str], batch: int = 32) -> list[str]:
        out = []
        for i in range(0, len(segs), batch):
            enc = self.tok(["grammar: " + s for s in segs[i:i + batch]], return_tensors="pt",
                           padding=True, truncation=True, max_length=128).to(_device())
            gen = self.model.generate(**enc, max_new_tokens=128, num_beams=1)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out

    def edit_rates(self, texts: list[str], punctuated: bool = True) -> np.ndarray:
        segs, owner = [], []
        for k, t in enumerate(texts):
            parts = sentences(t) if punctuated else chunk_words(t)
            segs += parts
            owner += [k] * len(parts)
        corr = self.correct(segs)
        edits, words = np.zeros(len(texts)), np.zeros(len(texts))
        for k, s, c in zip(owner, segs, corr):
            a, b = tokens(s), tokens(c)
            edits[k] += edit_distance(a, b)
            words[k] += len(a)
        return edits / np.maximum(words, 1)


class CoLA:
    """RoBERTa CoLA acceptability per sentence -> mean, min, share below 0.5."""

    def __init__(self, model_name: str = COLA_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(_device()).eval()

    @torch.inference_mode()
    def scores(self, texts: list[str], batch: int = 64) -> np.ndarray:
        segs, owner = [], []
        for k, t in enumerate(texts):
            parts = sentences(t)
            if len(parts) <= 1:          # unpunctuated / one long run -> windows
                parts = chunk_words(t)
            segs += parts
            owner += [k] * len(parts)
        probs = []
        for i in range(0, len(segs), batch):
            enc = self.tok(segs[i:i + batch], return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(_device())
            probs.append(torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
        probs = np.concatenate(probs) if probs else np.zeros(0)
        owner = np.array(owner)
        out = np.zeros((len(texts), 3))
        for k in range(len(texts)):
            p = probs[owner == k]
            out[k] = (p.mean(), p.min(), (p < 0.5).mean()) if len(p) else (0.5, 0.5, 0.0)
        return out


class LLMFeatures:
    """Qwen3 views of a transcript.

    * ``states``: mean-pooled hidden states of every layer over the
      transcript tokens. Middle layers often beat the last one for probing.
    * ``rubric``: expected score E[s] = sum_k p(k) * k over the next-token
      probabilities of "1".."5" after a rubric prompt, plus the entropy.
      Expected-value decoding is smoother than parsing a generated digit.
    """

    def __init__(self, model_name: str = LLM_MODEL, four_bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if four_bit:  # NF4 weights: Qwen3-4B fits a 6 GB GPU (~2.7 GB)
            import os

            from transformers import BitsAndBytesConfig

            # transformers 5.x otherwise copies all full-precision tensors to the
            # GPU in parallel before quantising (8 GB for Qwen3-4B -> OOM on 6 GB)
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=q,
                                                              device_map={"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(_device()).eval()
        self.digit_ids = [self.tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 6)]

    @torch.inference_mode()
    def states(self, text: str) -> np.ndarray:
        """Mean-pooled hidden states of every layer over the transcript tokens."""
        dev = _device()
        prefix = "Transcript of a spoken answer by an English learner:\n"
        ids_p = self.tok(prefix, return_tensors="pt").input_ids
        ids_t = self.tok(text or "(silence)", return_tensors="pt", add_special_tokens=False).input_ids[:, :512]
        ids = torch.cat([ids_p, ids_t], 1).to(dev)
        hs = self.model(ids, output_hidden_states=True).hidden_states
        span = slice(ids_p.shape[1], ids.shape[1])
        return torch.stack([h[0, span].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16)

    @torch.inference_mode()
    def rubric(self, text: str) -> np.ndarray:
        """[E[score], entropy, p(1)..p(5)] from next-token probabilities."""
        prompt = (f"{RUBRIC}\n\nTranscript (automatic speech recognition):\n\"{(text or '')[:2500]}\"\n\n"
                  "Rate only the grammar of the speaker on the 1-5 scale.\nGrammar score: ")
        pid = self.tok(prompt, return_tensors="pt").input_ids.to(_device())
        logits = self.model(pid).logits[0, -1].float()
        # Qwen tokenises "score: 3" as [":", " ", "3"]. The prompt therefore
        # ends with the space token, so the next token is the bare digit.
        p = torch.softmax(logits[self.digit_ids], -1).cpu().numpy()
        ev = float((p * np.arange(1, 6)).sum())
        ent = float(-(p * np.log(p + 1e-12)).sum())
        return np.array([ev, ent, *p], np.float32)

    def __call__(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        return self.states(text), self.rubric(text)
