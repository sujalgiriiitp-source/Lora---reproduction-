"""Plot the LoRA rank ablation + extensions from experiments/runs.csv.

Reads only the rows in runs.csv (no invented data) and writes four figures:
  accuracy-vs-rank.png       (seed-42 SST-2 ablation: accuracy + F1 vs rank)
  params-vs-accuracy.png     (all runs: accuracy vs trainable params)
  seed-stability.png         (full vs LoRA r=16, seed 42 vs seed 123)
  extensions.png             (all-linear LoRA + AG News vs the seed-42 baseline)

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
    return rows


def by_id(rows, run_id):
    return next(r for r in rows if r["run_id"] == run_id)



def main() -> None:
    rows = load_rows()
    # Seed-42 SST-2 ablation only (the original four runs).
    base = [r for r in rows if r["dataset"] == "sst2" and r["seed"] == "42"
            and r["run_id"] in ("full", "lora-r4", "lora-r8", "lora-r16")]
    full = by_id(base, "full")
    lora = sorted(
        (r for r in base if r["mode"] == "lora"),
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

    # --- Figure 2: accuracy vs trainable params (all runs) ---------------
    pts = [(int(r["trainable_params"]), float(r["accuracy"]), r["run_id"])
           for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n_params, a, run_id in pts:
        ax.scatter([n_params], [a], label=run_id)
        ax.annotate(run_id, (n_params, a), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Validation accuracy (SST-2 val / AG News test)")
    ax.set_title("Accuracy vs trainable parameters — all runs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "params-vs-accuracy.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: seed stability (full vs LoRA r=16, seed 42 vs 123) ----
    if any(r["run_id"] == "full-seed123" for r in rows):
        full123 = by_id(rows, "full-seed123")
        r16 = by_id(rows, "lora-r16")
        r16_123 = by_id(rows, "lora-r16-seed123")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = [0, 1]
        w = 0.35
        ax.bar([v - w / 2 for v in x],
               [float(full["accuracy"]), float(r16["accuracy"])],
               w, label="seed 42")
        ax.bar([v + w / 2 for v in x],
               [float(full123["accuracy"]), float(r16_123["accuracy"])],
               w, label="seed 123")
        for i, (a, b) in enumerate([(float(full["accuracy"]), float(full123["accuracy"])),
                                   (float(r16["accuracy"]), float(r16_123["accuracy"]))]):
            ax.text(i - w / 2, a + 0.002, f"{a:.4f}", ha="center", fontsize=9)
            ax.text(i + w / 2, b + 0.002, f"{b:.4f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Full fine-tuning", "LoRA r=16"])
        ax.set_ylabel("Validation accuracy (SST-2)")
        ax.set_title("Seed stability — full vs LoRA r=16 (SST-2)\n"
                     "gap 1.38pp at seed 42, 2.41pp at seed 123")
        ax.set_ylim(0.86, 0.93)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "seed-stability.png", dpi=150)
        plt.close(fig)

    # --- Figure 4: extensions (all-linear + AG News) ---------------------
    if any(r["run_id"] == "lora-r16-all" for r in rows):
        ext = [
            ("SST-2\nfull", by_id(rows, "full")),
            ("SST-2\nLoRA r16\nq/v only", by_id(rows, "lora-r16")),
            ("SST-2\nLoRA r16\nall-linear", by_id(rows, "lora-r16-all")),
            ("AG News\nfull", by_id(rows, "full-agnews")),
            ("AG News\nLoRA r16", by_id(rows, "lora-r16-agnews")),
        ]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        vals = [float(r["accuracy"]) for _, r in ext]
        bars = ax.bar([label for label, _ in ext], vals)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Accuracy (SST-2 val / AG News test)")
        ax.set_title("Extensions — all-linear LoRA closes the gap; "
                     "the rank result generalises to AG News")
        ax.set_ylim(0.86, 0.97)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "extensions.png", dpi=150)
        plt.close(fig)

    print(f"Wrote figures to {FIG_DIR} from {len(rows)} rows.")


if __name__ == "__main__":
    main()
