# Reproducing LoRA on DistilBERT: How Close Does Rank 16 Get to Full Fine-Tuning?

**Sujal Giri · November 2026 · [Code and full run log](https://github.com/sujalgiriiitp-source/Lora---reproduction-)**

## The question

LoRA (Hu et al., 2021) promises something that sounds too good: freeze a pre-trained model, train only small low-rank adapter matrices, and match full fine-tuning while updating ~1% of the parameters. The paper demonstrates it on large models. I wanted to check the claim at a scale I could actually run — and to find out where it breaks, not just where it works.

So this reproduction asks three concrete questions on DistilBERT + SST-2:

1. How close does LoRA come to full fine-tuning?
2. How does the answer change as the rank `r` grows from 4 to 16?
3. What does each rank actually cost in trainable parameters and training time?

## Setup

DistilBERT (`distilbert-base-uncased`, ~66M parameters) with a classification head, trained on SST-2 (67,349 training examples, 872 validation). All runs: 3 epochs, batch size 32, cosine schedule, warmup 0.1, weight decay 0.01, max length 128. Full fine-tuning at lr 2e-5; LoRA at lr 1e-4, dropout 0.1, `lora_alpha = 2r`, adapters on the attention query and value projections (`q_lin`, `v_lin`). One script (`src/train.py`) runs every configuration, so the runs differ only in mode and rank. Hardware: a free Colab T4 GPU.

Small scale is a deliberate choice: a run takes about 8–12 minutes on a free T4, so every claim here can be re-checked in an afternoon — including the seed correction below. Nine runs in total, all logged in `experiments/runs.csv`.

## Results

| Run | Trainable params | Accuracy | Macro-F1 | Train time |
| --- | --- | --- | --- | --- |
| Full fine-tuning | 66,955,010 | 0.9106 | 0.9104 | 689.6 s |
| LoRA r=4 | 665,858 (0.99%) | 0.8911 | 0.8910 | 466.0 s |
| LoRA r=8 | 739,586 (1.10%) | 0.8922 | 0.8921 | 504.2 s |
| LoRA r=16 | 887,042 (1.32%) | 0.8968 | 0.8966 | 480.6 s |

Accuracy rises with rank, but no rank closes the gap: r=16 sits 1.38 percentage points below full fine-tuning while training 1.32% of the parameters. The small steps between ranks (0.11pp, 0.46pp) are within noise on an 872-example validation set, so I don't claim them as wins — only the ordering. LoRA also trained faster in every case (466–504 s vs 689.6 s), though that mixes parameter count with optimiser overhead and isn't the main claim.

## The headline needed a correction

A single seed can flatter a result, so I repeated the comparison at seed 123. Full fine-tuning barely moved (0.9106 → 0.9083), but LoRA r=16 dropped to 0.8842 — the gap widened from 1.38pp to 2.41pp. The honest headline is therefore a range, not a point: **LoRA r=16 lands 1.4–2.4pp below full fine-tuning while training 1.32% of the parameters.** With two seeds this is a range, not an error bar, and I report it as one.

## Two extensions

**All-linear LoRA closes most of the gap.** Adapting every linear layer (`q_lin, k_lin, v_lin, out_lin, lin1, lin2`) at r=16 reaches 0.9071 — only 0.35pp below full fine-tuning — with 1,919,234 trainable parameters (2.79% of the model). The default attention-only adapters leave accuracy on the table; where you put the adapters matters as much as the rank.

**The pattern generalises to AG News.** On a 4-class task with 120,000 training examples, full fine-tuning scores 0.9467 and LoRA r=16 scores 0.9346 — a 1.21pp gap, the same order as SST-2.

| Extension | Accuracy | Gap to full |
| --- | --- | --- |
| LoRA r=16, seed 123 (SST-2) | 0.8842 | 2.41pp |
| LoRA r=16, all-linear (SST-2) | 0.9071 | 0.35pp |
| LoRA r=16 (AG News) | 0.9346 | 1.21pp |

## Limitations

One small model, one easy task: DistilBERT + SST-2 tests the method at small scale; it does not replicate the paper's GPT-3 / RoBERTa-large numbers. The validation set is small, so sub-1pp differences are noise. Learning rates are standard defaults, not tuned optima, and only two seeds were run. AG News was evaluated on its test split (the dataset ships no validation split), so cross-dataset accuracies aren't directly comparable — only the full-vs-LoRA gap within each dataset is.

## What I took away

LoRA delivers most of full fine-tuning's accuracy for ~1% of the trainable parameters, but "most" is doing real work in that sentence — on this setup the gap is 1.4–2.4pp, and it only shrinks to 0.35pp when adapters cover all linear layers. Rank helps, placement helps more, and a single seed can make any of it look better than it is. That is a useful thing to know before trusting a parameter-efficient method in a real system — and it is exactly what reproducing a paper, rather than citing it, is for.

## Reproduce it

```bash
git clone https://github.com/sujalgiriiitp-source/Lora---reproduction-
cd Lora---reproduction-
pip install -r requirements.txt
python src/train.py --mode full --epochs 3 --output runs/full
python src/train.py --mode lora --r 16 --epochs 3 --output runs/lora-r16
```

Every number in this write-up is copied from `experiments/runs.csv` (9 runs, 11 Nov 2026, Colab T4). Figures are in `experiments/figures/`.
