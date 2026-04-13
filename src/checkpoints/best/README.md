---
language:
- ar
license: mit
base_model: openai/whisper-small
tags:
- automatic-speech-recognition
- arabic
- whisper
- lora
- peft
- common-voice
datasets:
- MohamedRashad/common-voice-18-arabic
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
model-index:
- name: whisper-small-arabic-cv18-lora
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice 18 Arabic
      type: MohamedRashad/common-voice-18-arabic
      split: test
    metrics:
    - type: wer
      value: 37.39
      name: WER
    - type: cer
      value: 12.14
      name: CER
---

# whisper-small-arabic-cv18-lora

Fine-tuned **OpenAI Whisper-small** for Arabic ASR using **LoRA**, trained on
Common Voice 18 Arabic under a strict 16 GB VRAM constraint.

**MBZUAI — Institute of Foundation Models · Technical Test · April 2026**
**Author: Nassima OULD OUALI**

---

## Model description

This model is a parameter-efficient fine-tune of
[openai/whisper-small](https://huggingface.co/openai/whisper-small) (244M parameters)
using [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation).
Only **0.73% of parameters** are trainable (≈1.77M out of 244M),
enabling training within a **12.3 GB peak VRAM** footprint.

The model targets Arabic ASR across Modern Standard Arabic (MSA),
with awareness of the specific challenges of the Arabic language:
diglossia, morphological richness, diacritics, and data scarcity.

### Training data

| Split | Samples | Duration |
|---|---|---|
| Train (used) | 5,000 | 5.72 h |
| Full train available | 28,410 | ≈ 35.5 h |
| Validation | 10,471 | ≈ 13.1 h |
| Test | 10,471 | ≈ 13.1 h |

Source: `MohamedRashad/common-voice-18-arabic` — unofficial CC-0 Arabic extraction
of Mozilla Common Voice 18.0, in Parquet format.

---

## Evaluation results

All metrics computed on **normalised undiacritised Arabic** after applying a consistent
`normalize_arabic()` pipeline to both predictions and references.
These figures are not directly comparable to results reported on raw orthography.

| Model | Condition | WER (%) | CER (%) |
|---|---|---|---|
| Whisper-small | Zero-shot Arabic | 47.55 | 21.11 |
| **Whisper-small + LoRA** | **Fine-tuned (this model)** | **37.39** | **12.14** |
| Whisper-small | Full fine-tune, FLEURS Arabic† | 23.76 | — |

† Different benchmark (FLEURS), different normalisation — not directly comparable.

**Gain over zero-shot baseline:**
- WER: −10.16 pp absolute (−21.4% relative)
- CER: −9.0 pp absolute (−42.5% relative)
- Perfect transcriptions: 16.9% → 20.1%

### Training dynamics

| Epoch | Val WER (%) | Val CER (%) | Train Loss |
|---|---|---|---|
| 1 | 34.51 | 10.88 | 1.11 |
| 2 | 32.68 | 9.41 | 0.43 |
| 3 | 31.05 | 8.89 | 0.38 |
| 4 | 30.81 | 8.88 | 0.35 |
| **5** | **30.39** | **8.76** | **0.33** |

Monotonic WER decrease across all 5 epochs — no overfitting observed.

---

## Usage

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Load base model + LoRA adapter
processor = WhisperProcessor.from_pretrained(
    "nassimaODL/whisper-small-arabic-cv18-lora",
    language="arabic",
    task="transcribe"
)
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(
    base_model,
    "nassimaODL/whisper-small-arabic-cv18-lora"
)
model = model.merge_and_unload()  # merge adapters for faster inference
model.eval()

# Transcribe
import numpy as np

def transcribe(audio_array, sampling_rate=16000):
    forced_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )
    with torch.no_grad():
        ids = model.generate(
            inputs.input_features,
            forced_decoder_ids=forced_ids,
            no_repeat_ngram_size=3,   # prevents repetition loops on short audio
        )
    return processor.decode(ids[0], skip_special_tokens=True)
```

> **Note:** set `no_repeat_ngram_size=3` to suppress the repetition loop pathology
> that affects ≈1% of short or low-energy audio samples (known Whisper failure mode).

---

## LoRA configuration

| Parameter | Value |
|---|---|
| Base model | openai/whisper-small |
| Method | LoRA |
| Rank r | 16 |
| Alpha α | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, v_proj (all attention layers, encoder + decoder) |
| Trainable params | ≈1.77M / 244M (0.73%) |
| Peak GPU memory | 12.3 GB |

**Why q_proj + v_proj in encoder and decoder?**
Including `v_proj` alongside `q_proj` consistently outperforms `q_proj` alone
on low-resource languages ([LoRA-Whisper, Interspeech 2024](https://arxiv.org/abs/2406.07947);
[Pashto ASR, arXiv:2604.06507](https://arxiv.org/abs/2604.06507)).
The encoder is included because CV-18 Arabic's acoustic properties
(Quranic prosody, diacritised read speech) differ substantially from
Whisper's pre-training distribution.

---

## Training configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Effective batch size | 32 (batch=16 × grad_accum=2) |
| LR scheduler | Linear warmup (100 steps) + decay |
| Epochs | 5 |
| Early stopping | Patience = 3 on val WER |
| Mixed precision | fp16 |
| Optimizer | AdamW |
| Training time | ≈ 30 min (NVIDIA A100) |
| Primary metric | Normalised WER (not loss) |

---

## Limitations

- **MSA only.** Trained on read MSA + Quranic text. Generalises poorly to dialectal Arabic
  (Darija, Egyptian, Gulf). This is the most critical limitation for real-world deployment.
- **Read speech bias.** Common Voice is scripted read speech — not conversational Arabic.
- **Undiacritised output.** Cannot reliably produce diacritised transcriptions,
  which limits compatibility with downstream TTS pipelines.
- **Repetition loops.** Affects ≈1% of short/low-energy audio. Mitigated with
  `no_repeat_ngram_size=3` at inference time.
- **Short utterance regression.** Occasional over-generation on single-word inputs.

---

## Intended use

This model is intended for research and experimentation with Arabic ASR,
particularly in the context of:
- MSA speech transcription
- Parameter-efficient fine-tuning methodology on low-resource languages
- Baseline for dialectal Arabic adaptation

It is **not** recommended for production deployment without further evaluation
on dialectal speech and spontaneous conversation data.

---

## References

- Radford et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Talafha et al. (2023). *N-Shot Benchmarking of Whisper on Diverse Arabic Speech Recognition*. Interspeech. [arXiv:2306.02902](https://arxiv.org/abs/2306.02902)
- Song et al. (2024). *LoRA-Whisper: Parameter-Efficient and Generalizable Multilingual ASR*. Interspeech. [arXiv:2406.07947](https://arxiv.org/abs/2406.07947)
- Hanif et al. (2025). *Fine-Tuning Whisper for Pashto ASR*. [arXiv:2604.06507](https://arxiv.org/abs/2604.06507)
