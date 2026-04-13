"""
06_generate_figures.py
======================
Generates all figures for the technical report from real experimental data.
Saves to ./images/ for inclusion in asr_report_v6.tex.

Figures produced
----------------
Fig 1 : wer_progression.pdf    — WER/CER improvement across conditions (bar chart)
Fig 2 : learning_curves.pdf    — train loss + val loss + val WER per epoch
Fig 3 : audio_duration_dist.pdf — audio duration distribution (train/val/test)
Fig 4 : sentence_length_dist.pdf — sentence length before/after normalisation
Fig 5 : error_type_breakdown.pdf — error type breakdown (substitution/deletion/insertion)
Fig 6 : per_sample_wer_dist.pdf  — per-sample WER distribution (fine-tuned vs zero-shot)
Fig 7 : upvotes_distribution.pdf — community validation votes distribution

Usage
-----
    PYTHONNOUSERSITE=1 python 06_generate_figures.py
"""

import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.bbox"     : "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "english" : "#d62728",   # red   — pathological baseline
    "zeroshot": "#ff7f0e",   # orange — zero-shot
    "finetuned": "#2ca02c",  # green  — fine-tuned
    "train"   : "#1f77b4",   # blue
    "val"     : "#9467bd",   # purple
    "wer"     : "#d62728",
    "cer"     : "#ff7f0e",
    "neutral" : "#7f7f7f",
}

OUT_DIR = Path("./images")
OUT_DIR.mkdir(exist_ok=True)

print("Generating figures → ./images/")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — WER/CER progression across all conditions
# ══════════════════════════════════════════════════════════════════════════════

def fig_wer_progression():
    conditions = [
        "Forced\nEnglish\n(baseline A)",
        "Zero-shot\nArabic\n(baseline B)",
        "LoRA\nFine-tuned",
    ]
    wer = [99.34, 47.55, 37.39]
    cer = [99.18, 21.11, 12.14]
    colors_wer = [COLORS["english"], COLORS["zeroshot"], COLORS["finetuned"]]
    colors_cer = [COLORS["english"], COLORS["zeroshot"], COLORS["finetuned"]]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_wer = ax.bar(x - width/2, wer, width, label="WER (%)",
                      color=colors_wer, alpha=0.85, edgecolor="white")
    bars_cer = ax.bar(x + width/2, cer, width, label="CER (%)",
                      color=colors_cer, alpha=0.5,  edgecolor="white",
                      hatch="///")

    # Value labels
    for bar in bars_wer:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    for bar in bars_cer:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    # Improvement annotation
    ax.annotate("", xy=(2 - width/2, wer[2] + 3),
                xytext=(1 - width/2, wer[1] + 3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(1.5 - width/2, max(wer[1], wer[2]) + 12,
            "−10.2 pp WER\n(−21.4% rel.)",
            ha="center", fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("WER and CER Across Evaluation Conditions\n"
                 "(Full test set, 10,471 samples, CV-18 Arabic)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 120)
    ax.axhline(y=0, color="black", lw=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "wer_progression.pdf")
    plt.savefig(OUT_DIR / "wer_progression.png")
    plt.close()
    print("  ✓ wer_progression")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Learning curves (train loss + val loss + val WER per epoch)
# ══════════════════════════════════════════════════════════════════════════════

def fig_learning_curves():
    epochs     = [1, 2, 3, 4, 5]
    train_loss = [1.11, 0.43, 0.38, 0.35, 0.33]
    val_loss   = [0.994, 0.381, 0.360, 0.351, 0.350]
    val_wer    = [34.51, 32.68, 31.05, 30.81, 30.39]
    val_cer    = [10.88,  9.41,  8.89,  8.88,  8.76]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: loss curves
    ax1.plot(epochs, train_loss, "o-", color=COLORS["train"],
             label="Train loss", lw=2, markersize=6)
    ax1.plot(epochs, val_loss, "s--", color=COLORS["val"],
             label="Val loss", lw=2, markersize=6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.set_xticks(epochs)
    for e, tl, vl in zip(epochs, train_loss, val_loss):
        ax1.annotate(f"{tl:.2f}", (e, tl), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8,
                     color=COLORS["train"])
        ax1.annotate(f"{vl:.3f}", (e, vl), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8,
                     color=COLORS["val"])

    # Right: WER + CER
    ax2.plot(epochs, val_wer, "o-", color=COLORS["wer"],
             label="Val WER (%)", lw=2, markersize=6)
    ax2.plot(epochs, val_cer, "s--", color=COLORS["cer"],
             label="Val CER (%)", lw=2, markersize=6)

    # Zero-shot reference lines
    ax2.axhline(y=42.60, color=COLORS["wer"], lw=1, ls=":",
                label="Zero-shot WER (42.60%)", alpha=0.6)
    ax2.axhline(y=15.59, color=COLORS["cer"], lw=1, ls=":",
                label="Zero-shot CER (15.59%)", alpha=0.6)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error Rate (%)")
    ax2.set_title("Validation WER and CER per Epoch")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.set_xticks(epochs)
    for e, w in zip(epochs, val_wer):
        ax2.annotate(f"{w:.1f}", (e, w), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8,
                     color=COLORS["wer"])

    plt.suptitle("Training Dynamics — LoRA Fine-Tuning of Whisper-small on CV-18 Arabic",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "learning_curves.pdf")
    plt.savefig(OUT_DIR / "learning_curves.png")
    plt.close()
    print("  ✓ learning_curves")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Audio duration distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig_audio_duration():
    try:
        from datasets import load_from_disk
        ds = load_from_disk("./data/processed")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
        splits = ["train", "validation", "test"]
        split_colors = [COLORS["train"], COLORS["val"], COLORS["finetuned"]]
        stats = {}

        for ax, split, color in zip(axes, splits, split_colors):
            durations = [
                len(s["audio"]["array"]) / s["audio"]["sampling_rate"]
                for s in ds[split]
            ]
            durations = np.array(durations)
            stats[split] = durations

            ax.hist(durations, bins=40, color=color, alpha=0.75,
                    edgecolor="white", lw=0.5)
            ax.axvline(durations.mean(), color="black", lw=1.5, ls="--",
                       label=f"Mean: {durations.mean():.2f}s")
            ax.axvline(np.median(durations), color="gray", lw=1.5, ls=":",
                       label=f"Median: {np.median(durations):.2f}s")

            n = len(durations)
            total_h = durations.sum() / 3600
            ax.set_title(f"{split.capitalize()}\n"
                         f"n={n:,}  |  {total_h:.2f}h total")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Count" if split == "train" else "")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 12)

        plt.suptitle("Audio Duration Distribution by Split — CV-18 Arabic (preprocessed)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "audio_duration_dist.pdf")
        plt.savefig(OUT_DIR / "audio_duration_dist.png")
        plt.close()
        print("  ✓ audio_duration_dist")

    except Exception as e:
        print(f"  ✗ audio_duration_dist: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Sentence length before/after normalisation
# ══════════════════════════════════════════════════════════════════════════════

def fig_sentence_length():
    try:
        from datasets import load_from_disk

        # After normalisation (from processed)
        ds_proc = load_from_disk("./data/processed")
        after   = [len(s["sentence"]) for s in ds_proc["train"]]

        # Before normalisation (from raw)
        ds_raw  = load_from_disk("./data/raw")
        before  = [len(s["sentence"]) for s in ds_raw["train"]]
        # subsample to same size for fair comparison
        import random
        random.seed(42)
        before_sub = random.sample(before, len(after))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        bins = np.linspace(0, 160, 50)
        ax1.hist(before_sub, bins=bins, color=COLORS["zeroshot"],
                 alpha=0.7, edgecolor="white", label="Before normalisation")
        ax1.hist(after, bins=bins, color=COLORS["finetuned"],
                 alpha=0.7, edgecolor="white", label="After normalisation")
        ax1.axvline(np.mean(before_sub), color=COLORS["zeroshot"],
                    lw=2, ls="--", label=f"Mean before: {np.mean(before_sub):.1f}")
        ax1.axvline(np.mean(after), color=COLORS["finetuned"],
                    lw=2, ls="--", label=f"Mean after: {np.mean(after):.1f}")
        ax1.set_xlabel("Sentence length (characters)")
        ax1.set_ylabel("Count")
        ax1.set_title("Sentence Length Distribution\nBefore vs. After Normalisation")
        ax1.legend(fontsize=8.5)

        # Reduction breakdown as bar chart
        categories   = ["Mean length", "Max length"]
        before_vals  = [np.mean(before_sub), max(before_sub)]
        after_vals   = [np.mean(after),       max(after)]
        x = np.arange(len(categories))
        w = 0.35
        ax2.bar(x - w/2, before_vals, w, label="Before",
                color=COLORS["zeroshot"], alpha=0.8)
        ax2.bar(x + w/2, after_vals,  w, label="After",
                color=COLORS["finetuned"], alpha=0.8)
        for i, (b, a) in enumerate(zip(before_vals, after_vals)):
            ax2.text(i - w/2, b + 0.5, f"{b:.1f}", ha="center",
                     fontsize=9, color=COLORS["zeroshot"])
            ax2.text(i + w/2, a + 0.5, f"{a:.1f}", ha="center",
                     fontsize=9, color=COLORS["finetuned"])
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel("Characters")
        ax2.set_title("Key Statistics Before vs. After Normalisation")
        ax2.legend()

        plt.suptitle("Effect of Text Normalisation on Transcription Length",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "sentence_length_dist.pdf")
        plt.savefig(OUT_DIR / "sentence_length_dist.png")
        plt.close()
        print("  ✓ sentence_length_dist")

    except Exception as e:
        print(f"  ✗ sentence_length_dist: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Per-sample WER distribution: fine-tuned vs zero-shot
# ══════════════════════════════════════════════════════════════════════════════

def fig_per_sample_wer():
    try:
        results_dir = Path("./results")
        json_files  = sorted(results_dir.glob("final_eval_*.json"))
        if not json_files:
            print("  ✗ per_sample_wer_dist: no final_eval JSON found")
            return

        with open(json_files[-1]) as f:
            data = json.load(f)

        # Recompute per-sample WER from qualitative data
        # (full per-sample data not stored, use distribution stats instead)
        # We visualise using the stored percentile stats
        metrics = {m["label"]: m for m in data["metrics"]}
        ft_key  = [k for k in metrics if "fine-tuned" in k.lower()][0]
        zs_key  = [k for k in metrics if "zero-shot"  in k.lower()][0]
        ft = metrics[ft_key]
        zs = metrics[zs_key]

        fig, ax = plt.subplots(figsize=(9, 5))

        # Box-like visualization using percentiles
        labels  = ["Zero-shot Arabic", "LoRA Fine-tuned"]
        colors  = [COLORS["zeroshot"], COLORS["finetuned"]]
        models  = [zs, ft]

        for i, (label, color, m) in enumerate(zip(labels, colors, models)):
            # Draw distribution approximation using percentile boxes
            p25 = m["wer_p25"]
            med = m["wer_median"]
            p75 = m["wer_p75"]
            p90 = m["wer_p90"]
            wer = m["wer"]

            y = i * 0.6
            # IQR box
            ax.barh(y, p75 - p25, left=p25, height=0.3,
                    color=color, alpha=0.6, edgecolor=color)
            # Median line
            ax.plot([med, med], [y - 0.15, y + 0.15],
                    color="white", lw=2.5)
            ax.plot([med, med], [y - 0.15, y + 0.15],
                    color=color, lw=1.5, ls="--")
            # Whisker to P90
            ax.plot([p75, p90], [y, y], color=color, lw=1.5)
            ax.plot([p90, p90], [y - 0.08, y + 0.08], color=color, lw=1.5)
            # Mean marker
            ax.plot(wer, y, "D", color="black", markersize=7,
                    label=f"Mean WER: {wer:.1f}%" if i == 0 else f"Mean WER: {wer:.1f}%",
                    zorder=5)
            # Labels
            ax.text(p25 - 1, y, f"P25={p25:.0f}%",
                    ha="right", va="center", fontsize=8, color=color)
            ax.text(p75 + 1, y, f"P75={p75:.0f}%",
                    ha="left",  va="center", fontsize=8, color=color)
            ax.text(wer, y + 0.22, f"Mean={wer:.1f}%",
                    ha="center", fontsize=9, fontweight="bold", color=color)

        ax.set_yticks([0, 0.6])
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Per-sample WER (%)")
        ax.set_title("Per-Sample WER Distribution\n"
                     "Fine-tuned vs. Zero-shot (Full Test Set, 10,471 samples)\n"
                     "Box = IQR [P25, P75] · Line = Median · Diamond = Mean · Whisker = P90",
                     fontsize=11)
        ax.set_xlim(-5, 200)
        ax.axvline(100, color="gray", lw=0.8, ls=":", alpha=0.5,
                   label="WER = 100% threshold")

        perfect_ft = ft["perfect_pct"]
        perfect_zs = zs["perfect_pct"]
        ax.text(195, 0.6, f"Perfect: {perfect_ft:.1f}%",
                ha="right", fontsize=9, color=COLORS["finetuned"])
        ax.text(195, 0.0, f"Perfect: {perfect_zs:.1f}%",
                ha="right", fontsize=9, color=COLORS["zeroshot"])

        plt.tight_layout()
        plt.savefig(OUT_DIR / "per_sample_wer_dist.pdf")
        plt.savefig(OUT_DIR / "per_sample_wer_dist.png")
        plt.close()
        print("  ✓ per_sample_wer_dist")

    except Exception as e:
        print(f"  ✗ per_sample_wer_dist: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Upvotes distribution (data quality)
# ══════════════════════════════════════════════════════════════════════════════

def fig_upvotes():
    try:
        from datasets import load_from_disk
        ds  = load_from_disk("./data/processed")
        votes = ds["train"]["up_votes"]

        from collections import Counter
        counts = Counter(votes)
        keys   = sorted(counts.keys())
        vals   = [counts[k] for k in keys]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar([str(k) for k in keys], vals,
                      color=COLORS["train"], alpha=0.8, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{v:,}", ha="center", fontsize=9)
        ax.set_xlabel("Number of community upvotes")
        ax.set_ylabel("Number of samples")
        ax.set_title("Community Validation — Upvote Distribution\n"
                     "Training Split (5,000 subsampled from 28,408 retained samples)")
        ax.axhline(y=0, color="black", lw=0.5)
        total = sum(vals)
        ax.text(0.98, 0.95, f"Total: {total:,} samples\nMin upvotes: 2",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat",
                                      alpha=0.5))
        plt.tight_layout()
        plt.savefig(OUT_DIR / "upvotes_distribution.pdf")
        plt.savefig(OUT_DIR / "upvotes_distribution.png")
        plt.close()
        print("  ✓ upvotes_distribution")

    except Exception as e:
        print(f"  ✗ upvotes_distribution: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Memory budget comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_memory_budget():
    components = [
        "Whisper-small\nweights (fp32)",
        "Whisper-small\nweights (fp16)",
        "LoRA adapter\nweights",
        "AdamW states\n(LoRA only)",
        "Activations\nbatch=16",
        "Peak total\n(measured)",
        "GPU budget\n(constraint)",
    ]
    memory_gb = [1.86, 0.49, 0.03, 0.014, 5.5, 12.3, 16.0]
    colors_bar = [
        COLORS["neutral"], COLORS["train"], COLORS["finetuned"],
        COLORS["finetuned"], COLORS["val"],
        "#d62728", "#2ca02c"
    ]
    alphas = [0.4, 0.7, 0.7, 0.5, 0.6, 1.0, 0.4]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(components))
    bars = ax.barh(y, memory_gb, color=colors_bar, alpha=0.85,
                   edgecolor="white", height=0.6)
    for bar, v, a in zip(bars, memory_gb, alphas):
        bar.set_alpha(a)
        ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                f"{v:.2f} GB", va="center", fontsize=9.5,
                fontweight="bold" if v in [12.3, 16.0] else "normal")

    ax.axvline(16.0, color="#d62728", lw=2, ls="--",
               label="16 GB VRAM constraint", zorder=5)
    ax.axvline(12.3, color="#2ca02c", lw=1.5, ls=":",
               label="Peak measured (12.3 GB)", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(components)
    ax.set_xlabel("Memory (GB)")
    ax.set_title("GPU Memory Budget Analysis\n"
                 "LoRA Fine-Tuning of Whisper-small on NVIDIA A100")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 18.5)

    # Annotate safety margin
    ax.annotate("Safety margin\n3.7 GB",
                xy=(16.0, 5), xytext=(17.2, 4.5),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray"),
                color="gray")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "memory_budget.pdf")
    plt.savefig(OUT_DIR / "memory_budget.png")
    plt.close()
    print("  ✓ memory_budget")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig_wer_progression()
    fig_learning_curves()
    fig_audio_duration()
    fig_sentence_length()
    fig_per_sample_wer()
    fig_upvotes()
    fig_memory_budget()

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")
