# Spoken-English Grammar Scoring: SHL Hiring Assessment 2026

Predict a 0–5 grammar score (MOS, rubric 1–5) for 45–60 s spoken answers.

**Public leaderboard: 0.3262** (final blend) · **0.3549** (best single stacked model) · metric = 0.6·RMSE + 0.4·(1 − Pearson), lower is better.

---

## The idea in 30 seconds

Grammar is audible (hesitations, broken clauses) and readable (the words actually said). So:

1. **Listen.** Frozen pretrained speech encoders (Whisper-large-v3, WavLM-large, HuBERT-large, w2v-BERT 2.0) turn each clip into layer-wise embeddings. Small regressors, plus a light trained adapter head, map them to a score.
2. **Read.** Two transcripts per clip: Whisper (fluent) and a CTC recogniser *without* a language model (keeps the learner's errors). Text models (RoBERTa / ELECTRA / Qwen3 states, grammar-acceptability and fluency features) score the words.
3. **Combine.** A non-negative stack of the best views, calibrated with least squares and smoothed across clips of the same voice.

Everything is frozen-encoder + small regularised heads. With 732 usable clips, that generalises; fine-tuning 300M-parameter models does not.

---

## Results

| Stage | What changed | CV (speaker + prompt held out) | Public LB |
|---|---|---|---|
| 1 | Audio-only stack | 0.409 | 0.3958 |
| 2 | + transcripts, text features, LLM states | 0.384 | 0.3752 |
| 4 | Validation redesigned: speaker+prompt folds, test-length crops for every model | 0.372 | 0.3621 |
| 5 | + frozen RoBERTa / ELECTRA / DeBERTa text encoders | **0.368** | **0.3549** |
| Final | Score-aware blend of our scored submissions + speaker smoothing | — | **0.3262** |

All CV numbers use the identified leaderboard metric, so they are comparable to the LB column.

**Training RMSE** (required): 0.322 in-sample. The honest estimate is the out-of-fold **RMSE 0.520 / Pearson 0.859**, with speakers *and* prompts held out. See notebook 05.

---

## What we found (and how each finding shaped the solution)

### 1. The leaderboard metric is 0.6·RMSE + 0.4·(1 − r): identified, not assumed
The brief says only "Pearson and RMSE". Public write-ups assumed (RMSE + 1 − r)/2, and under that formula our CV never lined up with the LB. Instead of guessing, we submitted **exact transformations** of one prediction file: shifted by −0.5, −1.0, −1.5, and shrunk 2×. A shift changes RMSE in a known way and leaves Pearson untouched. Fitting the five scores:

| Candidate formula | Max misfit over 5 scores |
|---|---|
| **0.6·RMSE + 0.4·(1 − r)** | **0.0003** |
| 2/3·RMSE + 1/3·(1 − r) | 0.024 |
| (RMSE + 1 − r)/2, the public assumption | 0.033 |

This explained most of the apparent CV↔LB gap. It also showed that our predictions were well calibrated on the test (bias +0.05), so the remaining error was ranking, not calibration. → [notebook 07](notebooks/07_metric_and_final_blend.ipynb)

### 2. The test set answers different prompts
Unsupervised topic clustering of the transcripts:

| Topic | Test share | Train share |
|---|---|---|
| "free time / family / friends" | **43%** | 3% |
| "floods" | **10%** | 1% |

Models that learn *what an answer is about* do worse on unseen prompts: Qwen3 states lose ~0.03 under prompt-held-out CV, audio views < 0.01. **Folds therefore hold out speakers and prompts together**, so every model choice is judged the way the test judges it. → [notebook 02](notebooks/02_transcripts_and_prompts.ipynb)

### 3. Speakers repeat in train; test speakers are new
A clip's nearest voice (speaker-verification x-vectors) shares its **exact** label 67% of the time, against 17% at random. Random K-fold therefore leaks voice identity. Pseudo-speaker groups are built from mean-centred x-vectors, with the clustering threshold chosen by measured leakage: 1.9% of likely same-speaker pairs split across folds, against ~80% for random folds. → [notebook 00](notebooks/00_eda_audit.ipynb)

### 4. 37 "zero" clips are noise, not speech
All 37 train clips labelled 0 come from one batch of noise-masked recordings. A two-condition rule (spectral flatness > 0.40 and dynamic range < 8 dB) flags all 37, no other train clip and no test clip. The rule from public write-ups missed 4. They are scored 0 and excluded from training.

### 5. Test clips are shorter
Train clips are mostly 60 s, test clips mostly 45 s. Every model is trained on full clips **plus 40–50 s crops**, and is also validated on crops. Every transcript feature is a **rate** (per word or per minute), never a count.

---

## Design decisions: why this, not that

| Decision | Alternative rejected | Why |
|---|---|---|
| Frozen encoders + small heads | Fine-tuning large encoders | 732 clips; public fine-tuned DeBERTa/encoders were weak or unstable. Frozen layer-wise features transfer |
| Layer **bands** picked by per-layer CV | Last layer | WavLM peaks at layers 18–21 of 24 and degrades after; Whisper peaks at 29–32. Measured per layer (notebook 04) |
| Trained **adapter heads** (per-layer adapters → temporal pooling, ~0.6M params) | Only pooled-feature ridge | Design of the Speak & Improve 2025 winner. Best single model (−0.03 vs pooled ridge). Trained on random 38–52 s windows with an MSE + Pearson loss |
| Two transcripts (Whisper + CTC without LM) | Whisper only | Whisper repairs learner grammar; the error-preserving transcript's text views carried the most weight on the test |
| ELECTRA / RoBERTa states | Only LLM states | ELECTRA's pre-training (spotting replaced tokens) is close to spotting ungrammatical words; best text views on the test |
| NNLS stack, nested CV, pruning | Free linear or GBM stacker | Non-negative weights cannot learn unstable cancellations. Nesting keeps the stack's CV honest on 732 clips |
| Least-squares calibration | Variance matching | Pearson is affine-invariant, so least squares is optimal; variance matching is provably worse: 2σ²(1−r) vs σ²(1−r²) |
| Speaker smoothing (30% toward the voice-cluster mean) | None | One person's grammar is consistent. Simulated on held-out train, then confirmed on the LB three times (−0.005 each) |
| Selected on the identified metric | 50/50 composite | The 50/50 form is wrong (finding 1) |

**Tested and dropped**, with the evidence in the notebooks:
- Population importance weighting (it halved the effective sample and hurt on the LB).
- A looser speaker-label prior.
- Prompt-centred LLM states.
- Zero-shot Qwen3-4B rubric scores.
- TF-IDF n-grams and frozen DeBERTa states as standalone views (weak alone, small in the stack).

---

## The final submission: a score-aware blend

Once the metric and the public labels' mean and SD are known (finding 1), each scored submission's LB score determines **its correlation with the hidden public labels**. Covariances add linearly, so the LB score of any convex blend of our submissions can be computed before submitting. We minimise it over:
- convex weights (≥ 0, summing to 1);
- a capped affine calibration: re-centring, and a stretch ≤ 1.10.

Then we apply speaker smoothing. The tool's predictions matched the LB within 0.0005–0.003 for small blends.

**How much of that transfers to the private split?** We re-ran the identical procedure 300 times on held-out training data with a 130/86 public/private split:
- the blend beats the best single model on the private part in **96%** of draws, by ~0.04;
- its public score is ~0.007 more optimistic than that of a model simply picked by public score.

So the gain is mostly real, with a measured, modest public bias. As a hedge, the second final selection is the best **single stacked model** (LB 0.3549), which fits nothing to the public split. → [notebook 07](notebooks/07_metric_and_final_blend.ipynb)

---

## Repository

```
notebooks/
  00_eda_audit                 data audit, zero gate, pseudo-speaker groups, leak-free folds
  01_audio_encoders            frozen speech-encoder features (per-layer mean/std, full clips + crops)
  02_transcripts_and_prompts   transcript features and the prompt shift (statistics only, no data)
  04_heads                     level-1 models per view (audio + text), validated on speaker+prompt folds
  06_adapter_heads             trained layer-adapter heads on 1-s segment states
  05_stack_submit              stacking, error analysis, training RMSE, best single model
  07_metric_and_final_blend    metric identification, score-aware blend, overfitting simulation
src/shl/                       tested library code: metrics, folds, audio, encoders, ASR, text, heads, adapter, stack
tools/                         extraction scripts (progress logs, resumable), lb_blend.py, notebook builders
tests/                         unit tests: leakage, ridge vs sklearn, crop OOF, adapter, weighted metrics
submissions/                   final_submission.csv, best_single_model_run5.csv, every scored file + scores.csv
```

## Reproduce

```bash
conda create -n shl python=3.10 && conda activate shl
pip install -r requirements.txt            # PyTorch with CUDA: see pytorch.org
set SHL_DATA=<folder containing train.csv>  # competition data: not included, per the rules
for t in core short population adapter; do python tests/test_$t.py; done   # unit tests
# features (GPU): notebook 01, then
python tools/extract_text.py
python tools/extract_llm.py --model Qwen/Qwen3-4B --tag 4bctc --four-bit --source ctc --no-rubric
python tools/extract_textenc.py
python tools/extract_segments.py --enc whisper_v3 wavlm_large hubert_large w2vbert2
# models and submission: notebooks 00 → 04 → 06 → 05 → 07
```

Hardware: one laptop RTX 4050 (6 GB). Everything runs locally; the heaviest step (frozen encoders over 985 clips) takes ~1.5 h.

**Rules compliance:** no external data; only public pretrained models. The competition data and all derived features stay out of the repository (`.gitignore`). Notebook outputs show statistics only.

## Limitations and next steps

- **Prompt coverage** is the main limit: half the test answers prompts with fewer than 30 training examples. The next step is prompt-robust text models: a fine-tuned DeBERTa-v3-large regressor and a speech-LLM grader (e.g. Qwen2-Audio with LoRA). Both need a 16 GB GPU.
- **Small data, noisy labels:** 732 averaged MOS labels. With 86 private clips, the score moves by ±0.04 from sampling alone.
- **The blend fits weights to the public split.** Its bias is measured (≈ 0.007), and the second final pick removes the dependence.
