"""
03_evaluate_baselines.py
========================
Zero-shot baseline evaluation for Arabic ASR fine-tuning project.
Institute of Foundation Models — MBZUAI

Evaluates two baselines on the preprocessed Common Voice 18 Arabic test set:

  Condition A — Pathological baseline
      Whisper-small forced to transcribe Arabic audio as English.
      Expected WER: ~100–150 % (Latin output vs. Arabic reference).

  Condition B — Zero-shot Arabic baseline
      Whisper-small with language="arabic", no fine-tuning.
      Expected WER: ~80–120 % on CV-18 (see Talafha et al., Interspeech 2023
      and Mdhaffar et al., arXiv:2604.02209).

Design decisions
----------------
* Predictions are normalised with the SAME normalize_arabic() function used
  in 02_preprocess.py before WER/CER computation.  This is critical: without
  it, Whisper zero-shot outputs carry diacritics and Alef variants that inflate
  WER against the normalised references, making the comparison unfair.

* Condition A predictions are also passed through normalize_arabic().
  Because the output is Latin text, the function strips it entirely, leaving
  empty strings.  WER against a non-empty Arabic reference therefore equals
  100 % by definition of the jiwer formula — which is the correct, honest
  result.

* Evaluation is run on a fixed random subsample of 500 test samples
  (SEED = 42) for speed.  The full 10,471-sample test set is reserved for
  the fine-tuned model evaluation in 05_evaluate_finetuned.py.

* Batch inference (BATCH_SIZE = 16) with fp16 and flash-attention when
  available to keep runtime reasonable on a single A100.

* All results are saved to JSON and a human-readable text summary.

Usage
-----
    python 03_evaluate_baselines.py

Requirements
------------
    pip install transformers datasets jiwer torch accelerate soundfile
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

import torch
import numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer as compute_wer, cer as compute_cer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "./data/processed"
RESULTS_DIR    = "./results"
MODEL_NAME     = "openai/whisper-small"
EVAL_SAMPLES   = 500          # subsample of test set for baseline speed
SEED           = 42
BATCH_SIZE     = 16
SAMPLING_RATE  = 16_000
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# Text normalisation  (identical to 02_preprocess.py — must stay in sync)
# ════════════════════════════════════════════════════════════════════════════

KEEP_ARABIC = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]"
)

def normalize_arabic(text: str) -> str:
    """
    Apply the same normalisation pipeline used in 02_preprocess.py:
      1. Strip Tashkeel diacritics (U+064B–U+065F, U+0670)
      2. Unify Alef variants  → ا
      3. Unify Ya  variants   → ي
      4. Ta Marbuta           → ه
      5. Waw Hamza            → و
      6. Remove all non-Arabic characters (including Latin)
      7. Collapse whitespace

    Applying this to Condition A predictions (English output) strips all
    Latin characters, yielding an empty string — which correctly produces
    WER = 100 % against any non-empty Arabic reference.
    """
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"[ىئ]",    "ي", text)
    text = re.sub(r"ة",        "ه", text)
    text = re.sub(r"ؤ",        "و", text)
    text = KEEP_ARABIC.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ════════════════════════════════════════════════════════════════════════════
# Dataset loading & subsampling
# ════════════════════════════════════════════════════════════════════════════

def load_test_subset(processed_dir: str, n: int, seed: int):
    """
    Load the preprocessed test split and draw a fixed random subsample.

    We use the preprocessed set (not raw) so that:
      - Reference transcriptions are already normalised
      - Audio is already at 16 kHz
      - The same subset can be reproduced exactly with the same seed
    """
    log.info("Loading preprocessed dataset from %s", processed_dir)
    ds = load_from_disk(processed_dir)
    test = ds["test"]
    log.info("Test split: %d samples total", len(test))

    random.seed(seed)
    indices = sorted(random.sample(range(len(test)), n))
    subset  = test.select(indices)
    log.info("Subsampled %d test samples (seed=%d)", len(subset), seed)
    return subset


# ════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: torch.device):
    """
    Load Whisper processor and model.
    Uses fp16 on CUDA for memory efficiency.
    """
    log.info("Loading model: %s", model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    log.info(
        "Model loaded — parameters: %dM  dtype: %s  device: %s",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
        dtype,
        device,
    )
    return processor, model


# ════════════════════════════════════════════════════════════════════════════
# Batch inference
# ════════════════════════════════════════════════════════════════════════════

def transcribe_batch(
    audio_arrays: list,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device,
    forced_decoder_ids,
) -> list:
    """
    Transcribe a single batch of audio arrays.

    Parameters
    ----------
    audio_arrays      : list of np.ndarray, each at SAMPLING_RATE
    forced_decoder_ids: output of processor.get_decoder_prompt_ids(...)
                        controls the language and task tokens injected at
                        decoding time — this is the key difference between
                        Condition A (english) and Condition B (arabic)
    """
    inputs = processor(
        audio_arrays,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device)
    if model.dtype == torch.float16:
        input_features = input_features.half()

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )

    transcriptions = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )
    return transcriptions


def run_inference(
    subset,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device,
    language: str,
    condition_label: str,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """
    Run full inference over the subset for a given language condition.

    Returns
    -------
    predictions_raw  : list[str]  — raw Whisper output
    predictions_norm : list[str]  — after normalize_arabic()
    references_norm  : list[str]  — reference transcriptions (already normalised)
    elapsed_sec      : float
    """
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe",
    )
    log.info(
        "── Condition %s  (language='%s')  forced_decoder_ids=%s",
        condition_label, language, forced_decoder_ids,
    )

    all_raw   = []
    all_norm  = []
    all_refs  = []
    n         = len(subset)
    t0        = time.time()

    for start in range(0, n, batch_size):
        end    = min(start + batch_size, n)
        batch  = subset.select(range(start, end))

        audio_arrays = [
            np.array(s["audio"]["array"], dtype=np.float32)
            for s in batch
        ]
        refs = [s["sentence"] for s in batch]

        preds_raw = transcribe_batch(
            audio_arrays, processor, model, device, forced_decoder_ids
        )

        preds_norm = [normalize_arabic(p) for p in preds_raw]

        all_raw.extend(preds_raw)
        all_norm.extend(preds_norm)
        all_refs.extend(refs)

        if (start // batch_size) % 5 == 0:
            log.info(
                "  [%s]  %d / %d samples processed",
                condition_label, end, n,
            )

    elapsed = time.time() - t0
    log.info(
        "  [%s]  Inference done — %.1f s  (%.2f s/sample)",
        condition_label, elapsed, elapsed / n,
    )
    return all_raw, all_norm, all_refs, elapsed


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    predictions_norm: list,
    references_norm: list,
    condition_label: str,
) -> dict:
    """
    Compute WER and CER.

    Both inputs are already normalised.  jiwer handles empty-string
    predictions correctly (counts as full substitution).

    WER formula  : (S + D + I) / N   where N = total reference words
    CER formula  : same at character level
    """
    # jiwer expects non-empty reference strings — filter degenerate pairs
    pairs = [
        (p, r) for p, r in zip(predictions_norm, references_norm)
        if len(r.strip()) > 0
    ]
    n_dropped = len(predictions_norm) - len(pairs)
    if n_dropped > 0:
        log.warning(
            "[%s]  %d pairs dropped (empty reference)", condition_label, n_dropped
        )

    preds = [p for p, _ in pairs]
    refs  = [r for _, r in pairs]

    word_error_rate = compute_wer(refs, preds)
    char_error_rate = compute_cer(refs, preds)

    # Per-sample WER for diagnostics
    per_sample_wer = [
        compute_wer([r], [p]) for p, r in zip(preds, refs)
    ]

    metrics = {
        "condition"        : condition_label,
        "n_samples"        : len(pairs),
        "wer"              : round(word_error_rate * 100, 2),
        "cer"              : round(char_error_rate * 100, 2),
        "wer_median_pct"   : round(float(np.median(per_sample_wer)) * 100, 2),
        "wer_p90_pct"      : round(float(np.percentile(per_sample_wer, 90)) * 100, 2),
    }
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Results persistence
# ════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, results_dir: str):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON (metrics + sample predictions for error analysis)
    json_path = os.path.join(results_dir, f"baseline_results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Full results saved → %s", json_path)

    # Human-readable summary
    txt_path = os.path.join(results_dir, f"baseline_summary_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_summary(results))
    log.info("Summary saved       → %s", txt_path)

    return json_path, txt_path


def format_summary(results: dict) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("  ARABIC ASR — ZERO-SHOT BASELINE EVALUATION SUMMARY")
    lines.append("=" * 65)
    lines.append(f"  Model        : {results['model']}")
    lines.append(f"  Dataset      : {results['dataset']}")
    lines.append(f"  Test samples : {results['eval_samples']} (seed={results['seed']})")
    lines.append(f"  Timestamp    : {results['timestamp']}")
    lines.append("")
    lines.append(f"  {'Condition':<40} {'WER %':>7}  {'CER %':>7}")
    lines.append("  " + "-" * 58)
    for cond in results["conditions"]:
        m = cond["metrics"]
        lines.append(
            f"  {cond['label']:<40} {m['wer']:>7.2f}  {m['cer']:>7.2f}"
        )
    lines.append("")
    lines.append("  Note: predictions normalised with normalize_arabic()")
    lines.append("  before WER/CER computation (same pipeline as 02_preprocess.py).")
    lines.append("=" * 65)
    lines.append("")

    # Sample predictions
    for cond in results["conditions"]:
        lines.append(f"  ── Sample predictions  [{cond['label']}]")
        for ex in cond["examples"][:5]:
            lines.append(f"    REF  : {ex['reference']}")
            lines.append(f"    PRED : {ex['prediction_raw']}")
            lines.append(f"    NORM : {ex['prediction_norm']}")
            lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 65)
    log.info("  Arabic ASR — Zero-Shot Baseline Evaluation")
    log.info("=" * 65)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.warning("No GPU found — inference will be slow on CPU.")

    # ── Data ──────────────────────────────────────────────────────────────
    subset = load_test_subset(PROCESSED_DIR, EVAL_SAMPLES, SEED)

    # ── Model ─────────────────────────────────────────────────────────────
    processor, model = load_model(MODEL_NAME, device)

    # ── Conditions ────────────────────────────────────────────────────────
    # Each condition is defined by a language token only.
    # Everything else (model weights, audio input, references) is identical.
    conditions_config = [
        {
            "label"   : "A — Whisper-small forced English (pathological)",
            "language": "english",
        },
        {
            "label"   : "B — Whisper-small zero-shot Arabic",
            "language": "arabic",
        },
    ]

    all_condition_results = []

    for cfg in conditions_config:
        log.info("")
        raw_preds, norm_preds, norm_refs, elapsed = run_inference(
            subset      = subset,
            processor   = processor,
            model       = model,
            device      = device,
            language    = cfg["language"],
            condition_label = cfg["label"],
        )

        metrics = compute_metrics(norm_preds, norm_refs, cfg["label"])

        # Keep 20 examples for qualitative inspection
        examples = [
            {
                "reference"       : norm_refs[i],
                "prediction_raw"  : raw_preds[i],
                "prediction_norm" : norm_preds[i],
            }
            for i in range(min(20, len(norm_refs)))
        ]

        all_condition_results.append({
            "label"    : cfg["label"],
            "language" : cfg["language"],
            "metrics"  : metrics,
            "examples" : examples,
            "elapsed_s": round(elapsed, 1),
        })

        log.info(
            "  ► WER = %.2f %%   CER = %.2f %%",
            metrics["wer"], metrics["cer"],
        )

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "model"       : MODEL_NAME,
        "dataset"     : PROCESSED_DIR,
        "eval_samples": EVAL_SAMPLES,
        "seed"        : SEED,
        "timestamp"   : datetime.now().isoformat(),
        "device"      : str(device),
        "conditions"  : all_condition_results,
    }

    json_path, txt_path = save_results(results, RESULTS_DIR)

    # ── Final console summary ──────────────────────────────────────────────
    log.info("")
    log.info(format_summary(results))
    log.info("Next step: run 04_train.py to fine-tune with LoRA.")


if __name__ == "__main__":
    main()
