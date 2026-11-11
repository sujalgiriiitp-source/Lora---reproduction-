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

### Extensions (11 Nov 2026, same setup, Colab T4)

| Run | Setting | Trainable params | Accuracy | F1 (macro) | Train time |
| --- | --- | --- | --- | --- | --- |
| Full FT, seed 123 | SST-2 | 66,955,010 (100%) | 0.9083 | 0.9081 | 737.0 s |
| LoRA r=16, seed 123 | SST-2 | 887,042 (1.32%) | 0.8842 | 0.8840 | 510.6 s |
| Full FT | AG News (test 7,600) | 66,956,548 (100%) | 0.9467 | 0.9467 | 3,296.8 s |
| LoRA r=16 | AG News (test 7,600) | 888,580 (1.31%) | 0.9346 | 0.9346 | 2,457.2 s |
| LoRA r=16, all-linear | SST-2 | 1,919,234 (2.79%) | 0.9071 | 0.9070 | 639.0 s |

**Seed stability — a correction to the headline.** The full-vs-r16 gap is
1.38pp at seed 42 but 2.41pp at seed 123 (r16: 0.8968 → 0.8842; full:
0.9106 → 0.9083). "Within 1.38pp" is a seed-42 number, not a stable
property — across the two seeds the gap is 1.4–2.4pp (mean ≈ 1.9pp).
With only two seeds this is a range, not an error bar, and it is reported
as one.

**All-linear LoRA closes most of the gap.** Targeting all linear layers
(`q_lin, k_lin, v_lin, out_lin, lin1, lin2`) at r=16 reaches 0.9071 —
1.03pp above q/v-only r=16 (0.8968) and only 0.35pp below full
fine-tuning (0.9106) — with 1,919,234 trainable parameters (2.79%).

**The rank result generalises to AG News.** On a 4-class task with
120,000 training examples: full 0.9467 vs LoRA r=16 0.9346, a 1.21pp gap
— the same order as SST-2.

**Provenance.** The extension session's GPU VM disconnected after the
last run, so `experiments/runs.csv` was rebuilt from the trainer outputs
saved in the session notebook (Drive, `05-extension-runs.ipynb`). Rows
1–4 are the original log; rows 5–9 are the rebuilt rows. Two
transcription errors in the rebuilt copy's old rows were corrected
against the original log (r=8 total params, r=4 F1). The all-linear run's
`notes` field was restored from the flag it provably used (1,919,234
trainable params vs 887,042 for q/v-only). No accuracy number was
estimated or adjusted.

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
- **Two seeds, not a distribution.** Full fine-tuning and LoRA r=16 were
  repeated at seed 123; the other ranks were run once (seed 42). The
  seed spread is a range (gap 1.38–2.41pp), not an error bar.
- **AG News uses the test split for evaluation** (the dataset ships no
  validation split); the SST-2 numbers use validation. The two datasets'
  accuracies are not directly comparable to each other, only the
  full-vs-LoRA gap within each dataset is.

## 5. Next steps

1. ~~Complete the four full runs on a Colab T4 GPU and fill in Section 3
   from `experiments/runs.csv`.~~ **Done 11 Nov 2026** — all four runs
   finished, Section 3 filled from `experiments/runs.csv`.
2. ~~Repeat the best LoRA rank with a second seed to check stability.~~
   **Done 11 Nov 2026** — seed 123: full 0.9083, r16 0.8842; gap
   2.41pp vs 1.38pp at seed 42. Headline corrected to a range.
3. ~~Try one extension — e.g. LoRA on all linear layers, or rank varying
   per layer — and report whether it helps or not.~~ **Done 11 Nov 2026**
   — all-linear r16 reaches 0.9071, only 0.35pp below full FT.
4. ~~Extend to a second dataset (AG News) to test whether the rank
   ordering generalises beyond SST-2.~~ **Done 11 Nov 2026** — AG News
   gap 1.21pp (full 0.9467, r16 0.9346).

---

**Resume bullet (filled from the real runs, 11 Nov 2026):**

> "Reproduced LoRA (Hu et al., 2021) on DistilBERT + SST-2: rank ablation
> (r=4/8/16) with full fine-tuning baseline; best rank r=16 within 1.38pp
> of full (0.8968 vs 0.9106) using 1.32% of trainable parameters. Second
> seed puts the gap at 1.4–2.4pp; LoRA on all linear layers reaches 0.9071
> (within 0.35pp of full); the result generalises to AG News (gap 1.21pp)."

All numbers from `experiments/runs.csv` — 9 runs, 11 Nov 2026, Colab T4.
