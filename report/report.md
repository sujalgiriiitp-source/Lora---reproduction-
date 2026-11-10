# Reproducing LoRA: Low-Rank Adaptation for Efficient Fine-Tuning — With a Rank Ablation Study

**Author:** Sujal Giri — B.Tech CSE (SIET, 2028) · B.S. Computer Science &
Data Analytics (IIT Patna, 2029)
**Date:** November 2026 (draft — results pending full runs)
**Code:** https://github.com/sujalgiriiitp-source/Lora---reproduction-
**Paper reproduced:** Hu et al., *LoRA: Low-Rank Adaptation of Large
Language Models*, ICLR 2022 (arXiv:2106.09685)

---

## 1. Question

Full fine-tuning updates every weight of a pre-trained model. LoRA freezes
those weights and trains only small low-rank matrices injected into the
attention layers. The paper claims this matches full fine-tuning with a
tiny fraction of trainable parameters.

This reproduction asks, on a small model and a single task:

1. How close does LoRA come to full fine-tuning on SST-2 sentiment
   classification with DistilBERT?
2. How does the answer change as the LoRA rank `r` grows (4 → 8 → 16)?
3. How many trainable parameters — and how much training time — does
   each rank actually save?

## 2. Method

**Model.** DistilBERT (`distilbert-base-uncased`, ~66M parameters) with a
sequence-classification head.

**Data.** SST-2 (GLUE), binary sentiment. Train 67,349 / validation 872.
Tokenised with the DistilBERT tokenizer, max length 128.

**Runs (all with seed 42, 3 epochs, cosine schedule, warmup 0.1, weight
decay 0.01).** Full fine-tuning, lr 2e-5; LoRA, lr 1e-4, dropout 0.1,
target modules `q_lin` and `v_lin`, `lora_alpha = 2r`.

| Run | Trainable parameters |
| --- | --- |
| Full fine-tuning | all (~66M) |
| LoRA r=4 | adapters only |
| LoRA r=8 | adapters only |
| LoRA r=16 | adapters only |

Exact counts are taken from the run log (`experiments/runs.csv`), not
estimated by hand.

**Metrics.** Accuracy and macro-F1 on the validation split, plus
trainable-parameter count and wall-clock training time. Implementation:
PyTorch + Hugging Face Transformers + PEFT; `src/train.py` is the single
entry point for all four runs so that only `mode` / `r` differ.

## 3. Results

> **TO BE FILLED FROM REAL RUN — no numbers claimed yet.**
> The full runs (3 epochs on SST-2, Colab T4 GPU) have not been completed.
> The table below is the exact shape the results will take; every cell
> will be copied from `experiments/runs.csv` once the runs exist. The
> CPU smoke test in this repo (200 examples, 1 epoch) proves only that
> the code runs end to end and is not reported here as a result.

| Run | Trainable params | Accuracy | F1 (macro) | Train time |
| --- | --- | --- | --- | --- |
| Full fine-tuning | — | — | — | — |
| LoRA r=4 | — | — | — | — |
| LoRA r=8 | — | — | — | — |
| LoRA r=16 | — | — | — | — |

**What I will conclude from these numbers (once real):** whether accuracy
rises with rank and where it plateaus; whether any LoRA rank matches full
fine-tuning within ~1 percentage point; and what fraction of parameters
each rank needed to get there. If LoRA underperforms, that is reported as
it is — a negative result is still a result.

## 4. Limitations

- **One small model, one easy task.** DistilBERT + SST-2 cannot show
  whether the same rank trade-off holds for larger models or harder
  tasks. The LoRA paper's headline numbers are on GPT-3 / RoBERTa-large;
  this study tests the *method* at small scale, it does not replicate
  their numbers.
- **Small validation set.** SST-2 validation is 872 examples, so
  differences below ~1–2 percentage points are within noise. Small gaps
  between ranks will be described as noise, not as wins.
- **No hyperparameter search.** Learning rates are the standard defaults
  for this setup (2e-5 full, 1e-4 LoRA), not tuned optima. A different
  learning rate could change the ranking.
- **Single seed.** All runs use seed 42; no variance across seeds is
  measured, so no error bars are claimed.

## 5. Next steps

1. Complete the four full runs on a Colab T4 GPU and fill in Section 3
   from `experiments/runs.csv`.
2. Repeat the best LoRA rank with a second seed to check stability.
3. Try one extension — e.g. LoRA on all linear layers, or rank varying
   per layer — and report whether it helps or not.
4. Extend to a second dataset (AG News) to test whether the rank
   ordering generalises beyond SST-2.

---

**Resume bullet (fill in only after the real runs):**

> "Reproduced LoRA (Hu et al., 2021) on DistilBERT + SST-2: matched full
> fine-tuning within [X]pp using [Y]% of trainable parameters (rank r=[R]);
> ran a rank ablation (r=4/8/16) and reported accuracy, F1, and training
> time for each setting."

The `[X]`, `[Y]`, and `[R]` placeholders must be replaced with the real
numbers from `experiments/runs.csv` — never with estimates.
