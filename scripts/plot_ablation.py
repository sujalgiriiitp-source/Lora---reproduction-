"""Plot the LoRA rank ablation from experiments/runs.csv (no invented data).

Reads only the rows in runs.csv and writes two figures:
  experiments/figures/accuracy-vs-rank.png  (accuracy + F1 vs LoRA rank,
                                              full fine-tuning as dashed line)
  experiments/figures/params-vs-accuracy.png (accuracy vs trainable params)

Run:  .venv/bin/python scripts/plot_ablation.py
Re-run after new rows arrive; figures are regenerated from the file.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS_CSV = ROOT / "experiments" / "runs.csv"
FIG_DIR = ROOT / "experiments" / "figures"


def load_rows():
    with open(RUNS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    # Seed-42 SST-2 ablation only; extension rows (other seeds/datasets)
    # are plotted separately when they exist.
    base = [r for r in rows if r["dataset"] == "sst2" and r["seed"] == "42"]
    return base


def main() -> None:
    rows = load_rows()
    full = next(r for r in rows if r["run_id"] == "full")
    lora = sorted(
        (r for r in rows if r["mode"] == "lora" and r["run_id"].startswith("lora-r")),
        key=lambda r: int(r["lora_r"]),
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ranks = [int(r["lora_r"]) for r in lora]
    acc = [float(r["accuracy"]) for r in lora]
    f1 = [float(r["f1_macro"]) for r in lora]
    full_acc = float(full["accuracy"])
    full_f1 = float(full["f1_macro"])

    # --- Figure 1: accuracy / F1 vs rank ---------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ranks, acc, marker="o", label="LoRA accuracy")
    ax.plot(ranks, f1, marker="s", label="LoRA macro-F1")
    ax.axhline(full_acc, linestyle="--", label=f"Full fine-tuning acc ({full_acc:.4f})")
    ax.axhline(full_f1, linestyle=":", label=f"Full fine-tuning F1 ({full_f1:.4f})")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Validation score (SST-2, 872 examples)")
    ax.set_title("LoRA rank ablation — DistilBERT + SST-2 (seed 42, 3 epochs)")
    ax.set_xticks(ranks)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy-vs-rank.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: accuracy vs trainable params --------------------------
    pts = [(int(r["trainable_params"]), float(r["accuracy"]), r["run_id"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n_params, a, run_id in pts:
        ax.scatter([n_params], [a], label=run_id)
        ax.annotate(run_id, (n_params, a), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Validation accuracy (SST-2)")
    ax.set_title("Accuracy vs trainable parameters — DistilBERT + SST-2 (seed 42)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "params-vs-accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Wrote 2 figures to {FIG_DIR} from {len(rows)} seed-42 SST-2 rows.")


if __name__ == "__main__":
    main()
