# Arabic ASR Fine-Tuning 

**Nassima Ould Ouali** · April 2026 · 

---

## Overview

This repository documents the end-to-end fine-tuning of a lightweight Arabic Automatic Speech Recognition (ASR) model under strict compute constraints (single GPU, ≤16 GB VRAM).

**Arabic was chosen deliberately.** During the introductory meeting at the Institute of Foundation Models, Arabic ASR — and the challenge of building robust speech models across Modern Standard Arabic (MSA) and its dialects — was identified as the lab's primary speech research focus. Choosing Arabic aligns directly with this roadmap, and leverages native Arabic speaker knowledge for qualitative evaluation that goes beyond automatic metrics.

**Key results:**

| Condition | WER % | CER % |
|-----------|-------|-------|
| Whisper-small forced English (pathological baseline) | 99.34 | 99.18 |
| Whisper-small zero-shot Arabic | 42.60 | 15.59 |
| **Whisper-small + LoRA fine-tuned (5.72 h, 5 epochs)** | **30.39** | **8.76** |

→ **−12.2 WER points absolute (−28.7% relative)** over zero-shot, within 12.3 GB peak GPU memory.

---

## Repository Structure

```
mbzuai-asr-arabic/
├── src/
│   ├── 01_download.py          # Download & resample Common Voice 18 Arabic
│   ├── 02_preprocess.py        # Arabic-specific preprocessing pipeline
│   ├── 03_evaluate_baselines.py # Zero-shot baseline evaluation
│   ├── 04_train.py             # LoRA fine-tuning with WandB logging
│   ├── 05_evaluate_finetuned.py # Final evaluation on full test set
│   └── results/                # Evaluation summaries
├── report/
│   ├── asr_report_v5.tex       # Full technical report (LaTeX)
│   ├── references.bib          # BibTeX references
│   └── images/                 # WandB training curves
└── README.md
```

---

## Dataset

**Mozilla Common Voice 18 Arabic** — `MohamedRashad/common-voice-18-arabic`

| Split | Samples | Duration | Notes |
|-------|---------|----------|-------|
| Train | 28,410 | ~35.5 h | Subsampled to 5,000 for training |
| Validation | 10,471 | ~13.1 h | Full set used for eval |
| Test | 10,471 | ~13.1 h | Full set used for final eval |

**Why this dataset?** Mozilla Common Voice 17 was migrated away from HuggingFace in October 2025. This CC-0 Arabic-only extraction of CV-18 was the most accessible validated alternative without authentication requirements.

**Known limitation:** CV-18 Arabic is read MSA and Quranic text — not dialectal Arabic. This is the most important limitation for real-world deployment (see [Limitations](#limitations)).

### Preprocessing pipeline (`02_preprocess.py`)

All operations at PyArrow table level — no audio decoding required:

1. **Quality filter** — upvotes ≥ 2, transcription ≥ 2 characters (before and after normalisation)
2. **Diacritic removal** — strip Tashkeel (U+064B–U+065F, U+0670)
3. **Letter normalisation** — unify orthographic variants:
   - Alef: إأٱآا → ا
   - Ta Marbuta: ة → ه
   - Ya: ىئ → ي
   - Waw Hamza: ؤ → و
4. **Punctuation removal** — keep Arabic Unicode block only
5. **Subsampling** — 5,000 train samples (seed=42) → 5.72 h verified audio

**Why normalise?** Whisper zero-shot predictions carry diacritics absent from preprocessed references. Without normalisation, WER is inflated by ~10 points (Talafha et al., Interspeech 2023). The same `normalize_arabic()` function is applied consistently across all scripts.

---

## Model & Fine-Tuning Strategy

### Model: `openai/whisper-small`

- 244M parameters — fits in 16 GB VRAM under LoRA
- Pre-trained on ~739 h of Arabic audio
- Native Arabic BPE tokeniser
- Zero-shot WER = 42.60% on CV-18 Arabic (after normalisation)

Whisper-medium (769M) and Whisper-large-v3 (1.5B) were ruled out — both exceed the 16 GB limit under full fine-tuning.

### Fine-tuning: LoRA

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Method | LoRA | 0.73% trainable params vs 100% for full FT |
| Rank r | 16 | Standard for Whisper LoRA |
| Alpha α | 32 | 2× rank, standard scaling |
| Target modules | q_proj, v_proj | All attention layers, encoder + decoder |
| Dropout | 0.05 | Regularisation on 5,000 samples |
| Trainable params | 1.77M / 244M (0.73%) | AdamW states: ~14 MB vs ~3 GB full FT |
| Learning rate | 1e-4 | Linear warmup (100 steps) + decay |
| Effective batch | 32 | batch=16 × grad_accum=2 |
| Epochs | 5 | Early stopping patience=3 on val WER |
| Mixed precision | fp16 | Memory efficiency |
| Peak GPU memory | **12.3 GB** | ✓ Within 16 GB constraint |
| Training time | ~30 min | On NVIDIA A100 |

### Training dynamics

| Epoch | Val WER % | Val CER % | Train Loss |
|-------|-----------|-----------|------------|
| 1 | 34.51 | 10.88 | 1.11 |
| 2 | 32.68 | 9.41 | 0.43 |
| 3 | 31.05 | 8.89 | 0.38 |
| 4 | 30.81 | 8.88 | 0.35 |
| **5** | **30.39** | **8.76** | **0.33** |

Monotonic WER decrease across all 5 epochs with no overfitting — train loss and val loss converge together.

---

## Evaluation

### Metrics

- **WER** (Word Error Rate) — primary metric: (S+D+I)/N
- **CER** (Character Error Rate) — important for Arabic where word boundaries are ambiguous and morphological errors can inflate WER disproportionately

**Critical methodological note:** all metrics are computed after applying `normalize_arabic()` to both predictions and references. Without this, zero-shot WER is artificially inflated by ~10 points due to diacritic mismatches.

### Baseline strategy

- **Baseline A (pathological):** Whisper-small forced to English. Latin output stripped by `normalize_arabic()` → empty strings → WER ≈ 100%. Quantifies the total gap Arabic adaptation must close.
- **Baseline B (zero-shot Arabic):** Whisper-small with `language="arabic"`, no fine-tuning. Isolates LoRA contribution from pre-trained Arabic knowledge.

### Final results (validation set, 1,000 samples)

| Model | WER % | CER % | Δ WER abs | Δ WER rel |
|-------|-------|-------|-----------|-----------|
| Forced English | 99.34 | 99.18 | — | — |
| Zero-shot Arabic | 42.60 | 15.59 | — | — |
| **LoRA fine-tuned** | **30.39** | **8.76** | **−12.21 pp** | **−28.7%** |

*Full test set results (10,471 samples) in `results/final_eval_*.txt`.*

---

## Limitations

These limitations are not incidental — they define the research agenda:

1. **MSA vs. dialectal Arabic** — CV-18 is read MSA. The lab's real targets (Darija, Egyptian, Gulf Arabic) differ significantly phonetically and lexically. MSA-fine-tuned models often generalise poorly to unseen dialects. This is the most important limitation for downstream deployment.

2. **Read speech bias** — Common Voice is scripted read speech. Spontaneous Arabic, conversational speech, code-switching (Arabic–French, Arabic–English) are absent. WER on read speech does not predict real-world performance.

3. **Diacritics for TTS** — after normalisation, transcriptions are undiacritised. A model trained on this data cannot reliably generate diacritized output — which is the key requirement for downstream TTS training. Fine-tuning on diacritized data (Tashkeela corpus) would be the critical next step.

4. **Speaker diversity** — 70% missing demographic metadata; corpus dominated by few contributors. Limited generalisation to unseen accents.

5. **LoRA vs. full fine-tuning** — recent work (arXiv:2604.06507) shows full fine-tuning can outperform LoRA on small corpora. Our choice is dictated by the 16 GB constraint.

6. **Subsampling** — 5,000 of 28,410 training samples sacrifices acoustic coverage for compute feasibility.

---

## Next Steps

Given more compute and dialectal data:

1. **Dialectal fine-tuning** — MGB-3 (Egyptian), MGB-2 (multi-dialect broadcast), Darija corpora
2. **Diacritization** — add Tashkeela corpus for diacritized ASR output → TTS-ready transcriptions
3. **Continued pre-training** — SSL-style continued pre-training on unlabeled Arabic audio before supervised fine-tuning
4. **Full fine-tuning** — compare LoRA vs. full fine-tuning on the same data budget
5. **End-to-end pipeline** — connect ASR → TTS for a speech-to-speech prototype

---

## Reproducing the Results

### Environment

```bash
conda create -n asr_env python=3.10
conda activate asr_env

pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.45.2
pip install peft==0.18.1
pip install accelerate==1.13.0
pip install datasets==2.21.0
pip install soundfile librosa jiwer wandb
pip install numpy"<2"
```

> **Note:** the cluster environment had pre-installed conflicting packages. Use `PYTHONNOUSERSITE=1` to isolate the conda environment:
> ```bash
> PYTHONNOUSERSITE=1 python script.py
> ```

### Running the pipeline

```bash
# 1. Download dataset
PYTHONNOUSERSITE=1 python src/01_download.py

# 2. Preprocess
PYTHONNOUSERSITE=1 python src/02_preprocess.py

# 3. Evaluate zero-shot baselines
PYTHONNOUSERSITE=1 python src/03_evaluate_baselines.py

# 4. Fine-tune (requires GPU ≤16 GB VRAM)
export WANDB_PROJECT="mbzuai-asr-arabic"
PYTHONNOUSERSITE=1 python src/04_train.py

# 5. Final evaluation
PYTHONNOUSERSITE=1 python src/05_evaluate_finetuned.py
```

### WandB

Training curves are logged at:
`https://wandb.ai/nassima-ouldouali-ecole-polytechnique/mbzuai-asr-arabic`

---

## References

- Radford et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Ardila et al. (2020). *Common Voice: A Massively-Multilingual Speech Corpus*. LREC.
- Talafha et al. (2023). *N-Shot Benchmarking of Whisper on Diverse Arabic Speech Recognition*. Interspeech. [arXiv:2306.02902](https://arxiv.org/abs/2306.02902)
- Ali et al. (2016). *The MGB-2 Challenge: Arabic Multi-Dialect Broadcast Media Recognition*. SLT. [arXiv:1609.05625](https://arxiv.org/abs/1609.05625)
- Grigoryan et al. (2025). *Open ASR Models for Classical and Modern Standard Arabic*. ICASSP. [arXiv:2507.13977](https://arxiv.org/abs/2507.13977)
- Dhouib et al. (2022). *Arabic ASR: A Systematic Literature Review*. Applied Sciences.
- Haboussi et al. (2025). *Arabic Speech Recognition Using Neural Networks*. J. Umm Al-Qura Univ.

---

## Technical Report

The full technical report is available in `report/asr_report_v5.tex` (LaTeX, compiled with XeLaTeX + Biber).



