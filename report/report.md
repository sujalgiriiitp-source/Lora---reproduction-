# Reproducing LoRA: Low-Rank Adaptation for Efficient Fine-Tuning — With a Rank Ablation Study

**Author:** Sujal Giri — B.Tech CSE (SIET, 2028) · B.S. Computer Science &
Data Analytics (IIT Patna, 2029)
**Date:** November 2026 (results completed 11 Nov 2026, Colab T4 GPU)
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

> **Runs completed 11 Nov 2026 on a Colab T4 GPU.** Every cell below is
> copied from `experiments/runs.csv` (DistilBERT, SST-2, 3 epochs, batch
> 32, seed 42; validation split, 872 examples). The CPU smoke test in
> this repo was never run and is not reported here.

| Run | Trainable params | Accuracy | F1 (macro) | Train time |
| --- | --- | --- | --- | --- |
| Full fine-tuning | 66,955,010 (100%) | 0.9106 | 0.9104 | 689.6 s |
| LoRA r=4 | 665,858 (0.99%) | 0.8911 | 0.8910 | 466.0 s |
| LoRA r=8 | 739,586 (1.10%) | 0.8922 | 0.8921 | 504.2 s |
| LoRA r=16 | 887,042 (1.32%) | 0.8968 | 0.8966 | 480.6 s |

**What these numbers show:** accuracy rises monotonically with rank
(0.8911 → 0.8922 → 0.8968 → 0.9106), but no LoRA rank matches full
fine-tuning within 1 percentage point — the closest, r=16, is 1.38pp
below full while using 1.32% of the trainable parameters. The gaps
between adjacent ranks (0.11pp, 0.46pp) are within noise for an
872-example validation set, so they are not claimed as wins — only the
rank ordering is. LoRA also trained faster than full fine-tuning in
every case (466–504 s vs 689.6 s), though that comparison mixes
trainable-parameter count with optimiser overhead and is not the
study's main claim. The honest headline: LoRA gets most of the way
there for ~1% of the parameters, but on this setup it does not fully
close the gap.

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

1. ~~Complete the four full runs on a Colab T4 GPU and fill in Section 3
   from `experiments/runs.csv`.~~ **Done 11 Nov 2026** — all four runs
   finished, Section 3 filled from `experiments/runs.csv`.
2. Repeat the best LoRA rank with a second seed to check stability.
3. Try one extension — e.g. LoRA on all linear layers, or rank varying
   per layer — and report whether it helps or not.
4. Extend to a second dataset (AG News) to test whether the rank
   ordering generalises beyond SST-2.

---

**Resume bullet (filled from the real runs, 11 Nov 2026):**

> "Reproduced LoRA (Hu et al., 2021) on DistilBERT + SST-2: matched full
> fine-tuning within 1.38pp using 1.32% of trainable parameters (rank r=16);
> ran a rank ablation (r=4/8/16) and reported accuracy, F1, and training
> time for each setting."

All numbers from `experiments/runs.csv` — full fine-tuning 0.9106 accuracy
(66,955,010 trainable params); LoRA r=16 0.8968 accuracy (887,042 params).
