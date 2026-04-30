# Arabic ASR Fine-Tuning — Whisper-small + LoRA

**Nassima OULD OUALI** · April 2026 · 

---

## 🤗 Model on Hugging Face

**[nassimaODL/whisper-small-arabic-cv18-lora](https://huggingface.co/nassimaODL/whisper-small-arabic-cv18-lora)**

The fine-tuned LoRA checkpoint is publicly available on Hugging Face Hub.
Use it directly in 5 lines of code — see the model card for full usage instructions.

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

processor  = WhisperProcessor.from_pretrained("nassimaODL/whisper-small-arabic-cv18-lora",
                                               language="arabic", task="transcribe")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model      = PeftModel.from_pretrained(base_model, "nassimaODL/whisper-small-arabic-cv18-lora")
model      = model.merge_and_unload()
```

---

## Overview

End-to-end fine-tuning of a lightweight Arabic ASR model under strict compute
constraints (single GPU, ≤16 GB VRAM).

**Arabic was chosen deliberately** — Arabic ASR, and the challenge of building
robust speech models across MSA and dialectal Arabic, is central to the lab's
current research roadmap. Choosing Arabic allows addressing a real research
challenge rather than a convenient one, and leverages native Arabic speaker
knowledge for qualitative evaluation beyond automatic metrics.

### Key results

| Condition | WER % | CER % |
|---|---|---|
| Whisper-small zero-shot Arabic (full test set) | 47.55 | 21.11 |
| **Whisper-small + LoRA fine-tuned (full test set)** | **37.39** | **12.14** |

→ **−10.2 pp WER absolute (−21.4% relative)** · **−9.0 pp CER absolute (−42.5% relative)**
→ Peak GPU memory: **12.3 GB** (within the 16 GB constraint)
→ Trainable parameters: **1.77M / 244M (0.73%)**

> **Evaluation protocol.** All WER/CER figures are computed on normalised
> undiacritised Arabic (after `normalize_arabic()` applied to both predictions
> and references). Not directly comparable to results on raw orthography.

---

## Repository Structure

```
asr-arabic-finetuning/
├── arabic_asr_finetuning.ipynb     # Complete pipeline notebook
├── requirements.txt                 # Full dependency stack
├── README.md                        # This file
├── src/
│   ├── 01_download.py              # Download & resample Common Voice 18 Arabic
│   ├── 02_preprocess.py            # Arabic normalisation pipeline (PyArrow level)
│   ├── 03_evaluate_baselines.py    # Zero-shot baseline evaluation
│   ├── 04_train.py                 # LoRA fine-tuning with WandB logging
│   └── 05_evaluate_finetuned.py    # Final evaluation on full test set
└── results/                         # Evaluation summaries (JSON + TXT)
```

---

## Dataset

**Mozilla Common Voice 18 Arabic** — `MohamedRashad/common-voice-18-arabic`

| Split | Samples | Duration | Used |
|---|---|---|---|
| Train | 28,410 | ≈ 35.5 h | Subsampled to 5,000 (5.72 h) |
| Validation | 10,471 | ≈ 13.1 h | Full set |
| Test | 10,471 | ≈ 13.1 h | Full set |

**Why CV-18?** Mozilla migrated Common Voice 17 away from Hugging Face in
October 2025. This CC-0 Arabic-only extraction was the most reliable alternative.

**Preprocessing pipeline** (`02_preprocess.py`) — all steps at PyArrow table level:
1. Quality filter — upvotes ≥ 2, transcription ≥ 2 characters
2. Diacritic removal — strip Tashkeel (U+064B–U+065F, U+0670)
3. Letter normalisation — Alef variants → ا · Ta Marbuta → ه · Ya → ي · Waw Hamza → و
4. Punctuation removal — keep Arabic Unicode block only
5. Subsample 5,000 train samples (seed=42)

---

## Model & Fine-Tuning Strategy

### Model: `openai/whisper-small`

- 244M parameters · fits in 16 GB VRAM under LoRA
- Pre-trained on ≈739 h of Arabic audio
- Zero-shot WER = 47.55% on CV-18 Arabic (full test set, after normalisation)

### LoRA configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Rank r | 16 | Standard for Whisper LoRA |
| Alpha α | 32 | 2× rank |
| Target modules | q_proj, v_proj | All layers, encoder + decoder |
| Trainable params | 1.77M / 244M (0.73%) | AdamW states: ~14 MB vs ~3 GB full FT |
| Dropout | 0.05 | Regularisation on 5,000 samples |

**Why q_proj + v_proj in encoder and decoder?**
Including `v_proj` alongside `q_proj` consistently outperforms `q_proj` alone
([LoRA-Whisper, Interspeech 2024](https://arxiv.org/abs/2406.07947);
[Pashto ASR, arXiv:2604.06507](https://arxiv.org/abs/2604.06507)).
The encoder is included because CV-18 Arabic's acoustic properties
(Quranic prosody, diacritised read speech) differ from Whisper's pre-training distribution.

### Training hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Effective batch size | 32 (batch=16 × grad_accum=2) |
| LR scheduler | Linear warmup (100 steps) + decay |
| Epochs | 5 (early stopping patience=3 on val WER) |
| Mixed precision | fp16 |
| Peak GPU memory | **12.3 GB** ✓ within 16 GB budget |
| Training time | ≈ 30 min (NVIDIA A100) |

### Training dynamics

| Epoch | Val WER % | Val CER % | Train Loss |
|---|---|---|---|
| 1 | 34.51 | 10.88 | 1.11 |
| 2 | 32.68 | 9.41 | 0.43 |
| 3 | 31.05 | 8.89 | 0.38 |
| 4 | 30.81 | 8.88 | 0.35 |
| **5** | **30.39** | **8.76** | **0.33** |

Monotonic WER decrease across all 5 epochs — no overfitting observed.

---

## Evaluation

### Official results — full 10,471-sample test set

| Model | WER % | CER % | Perfect % |
|---|---|---|---|
| Whisper-small zero-shot Arabic | 47.55 | 21.11 | 16.9 |
| **Whisper-small + LoRA (this work)** | **37.39** | **12.14** | **20.1** |

### Qualitative analysis (native Arabic speaker)

| Error type | Zero-shot | Fine-tuned |
|---|---|---|
| Lexical hallucination | ✗ frequent | ✓ corrected |
| Alef/Waw orthographic confusion | ✗ present | ✓ corrected |
| Dialectal expression (colloquial verb) | ✗ replaced by MSA | ✓ recovered |
| Repetition loop on short audio | — | ✗ ~1% of samples |
| Over-generation on single words | ✓ correct | ✗ occasional |

---

## Limitations

1. **MSA only** — CV-18 is read MSA. Generalises poorly to dialectal Arabic (Darija, Egyptian, Gulf). Most critical limitation for real-world deployment.
2. **Read speech bias** — scripted speech, no conversational Arabic or code-switching.
3. **Undiacritised output** — incompatible with downstream TTS training without Tashkeela corpus fine-tuning.
4. **Repetition loops** — affects ≈1% of short/low-energy samples. Fix: `no_repeat_ngram_size=3` at inference (zero cost).
5. **LoRA vs. full fine-tuning** — full FT may outperform on small corpora; constrained by 16 GB budget.

---

## Reproducing the Results

```bash
# 1. Clone and install
git clone https://github.com/NassimaOULDOUALI/asr-arabic-finetuning
cd asr-arabic-finetuning
pip install -r requirements.txt

# 2. Run the pipeline
python src/01_download.py
python src/02_preprocess.py
python src/03_evaluate_baselines.py
export WANDB_PROJECT="asr-arabic-finetuning"
python src/04_train.py
python src/05_evaluate_finetuned.py
```

**GPU requirement:** CUDA ≥ 11.8, VRAM ≥ 16 GB

> **Server note:** if `torchcodec` errors with `libnppicc.so.13 not found`,
> run `pip uninstall torchcodec -y` and set `DATASETS_AUDIO_BACKEND=soundfile`.
> See Appendix A of the technical report for the full workaround.

### WandB training logs
[https://wandb.ai/nassima-ouldouali-ecole-polytechnique/asr-arabic-finetuning](https://wandb.ai/nassima-ouldouali-ecole-polytechnique/asr-arabic-finetuning)

---

## Next Steps

1. **Repetition loop fix** — `no_repeat_ngram_size=3` at inference. Zero cost, eliminates ~1% catastrophic failures.
2. **Dialectal fine-tuning** — MGB-3 (Egyptian), MGB-2 (multi-dialect), Darija corpora.
3. **Diacritised ASR** — Tashkeela corpus for TTS-compatible output.
4. **Continued pre-training** — SSL on unlabelled Arabic audio before supervised FT.
5. **End-to-end pipeline** — ASR → TTS → speech-to-speech prototype.

---

## References

1. Radford et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
2. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. Ardila et al. (2020). *Common Voice: A Massively-Multilingual Speech Corpus*. LREC.
4. Talafha et al. (2023). *N-Shot Benchmarking of Whisper on Diverse Arabic Speech*. Interspeech. [arXiv:2306.02902](https://arxiv.org/abs/2306.02902)
5. Ali et al. (2016). *The MGB-2 Challenge: Arabic Multi-Dialect Broadcast Media Recognition*. SLT. [arXiv:1609.05625](https://arxiv.org/abs/1609.05625)
6. Song et al. (2024). *LoRA-Whisper: Parameter-Efficient Multilingual ASR*. Interspeech. [arXiv:2406.07947](https://arxiv.org/abs/2406.07947)
7. Hanif et al. (2025). *Fine-Tuning Whisper for Pashto ASR*. [arXiv:2604.06507](https://arxiv.org/abs/2604.06507)
