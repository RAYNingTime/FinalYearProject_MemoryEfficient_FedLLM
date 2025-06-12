# Federated Learning with OPT-350M + LoRA using FedProx: 6000-Sample Run

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Analysis Date:** 9 June 2025
**Strategy:** FedProx (μ = 0.1) with LoRA adapters
**Configuration:** 2000 samples/client × 3 clients (6 000 total), LoRA (*r*= 16, α = 32, dropout = 0.5), 3 local epochs, LR = 1 × 10⁻⁵, 20 rounds scheduled (stopped at round 20)

---

## 1. Experiment Setup

| Component             | Configuration                            |
| --------------------- | ---------------------------------------- |
| **Model**             | `facebook/opt-350m`                      |
| **PEFT**              | LoRA (*r* = 16, *α* = 32, dropout = 0.5) |
| **Dataset**           | `codeparrot/codeparrot-clean-valid`      |
| **Total Samples**     | 6000                                     |
| **Client Split**      | 2000 samples/client × 3 clients          |
| **Epochs per Client** | 3                                        |
| **Learning Rate**     | 1 × 10⁻⁵                                 |
| **Batch Size**        | 2 (train), 4 (eval)                      |
| **Rounds Scheduled**  | 20                                       |
| **Rounds Completed**  | 20                                       |
| **Start Time**        | 2025-06-09 02:45:47                      |
| **Finish Time**       | 2025-06-09 10:38:34                      |
| **Total Runtime**     | 27 423.6 s (\~7.6 h)                     |

> **Memory profile:**
>
> * VRAM: fully utilized (8 GB) during client training; freed during server aggregation.
> * RAM: remained under 9 GB throughout (vs. 30–40 GB in full-model FedProx runs).

At the start of training, each process confirmed:

> **Trainable params:** 1 572 864 / 332 769 280 (0.47 %)

indicating that only the LoRA adapter matrices were being fine-tuned.

---

## 2. Loss Trajectories

### 2.1 Distributed (Train) Loss

| Round | Aggregated Train Loss |
| ----: | --------------------: |
|     1 |                2.6038 |
|     2 |                2.5122 |
|     3 |                2.4589 |
|     4 |                2.4234 |
|     5 |                2.3961 |
|     6 |                2.3736 |
|     7 |                2.3547 |
|     8 |                2.3383 |
|     9 |                2.3237 |
|    10 |                2.3108 |
|    11 |                2.2986 |
|    12 |                2.2879 |
|    13 |                2.2775 |
|    14 |                2.2683 |
|    15 |                2.2593 |
|    16 |                2.2510 |
|    17 |                2.2435 |
|    18 |                2.2364 |
|    19 |                2.2295 |
|    20 |                2.1748 |

### 2.2 Centralized (Eval) Loss

| Round | Eval Loss |
| ----: | --------: |
|     0 |    2.9871 |
|     1 |    2.5845 |
|     2 |    2.4933 |
|     3 |    2.4405 |
|     4 |    2.4054 |
|     5 |    2.3785 |
|     6 |    2.3562 |
|     7 |    2.3376 |
|     8 |    2.3213 |
|     9 |    2.3071 |
|    10 |    2.2944 |
|    11 |    2.2823 |
|    12 |    2.2718 |
|    13 |    2.2616 |
|    14 |    2.2524 |
|    15 |    2.2435 |
|    16 |    2.2353 |
|    17 |    2.2278 |
|    18 |    2.2207 |
|    19 |    2.2138 |
|    20 |    2.2073 |

---

## 3. Runtime & Throughput

| Metric                       | Value (per‐round avg.) |
| ---------------------------- | ---------------------: |
| **Eval runtime** (s)         |                \~734 s |
| **Eval samples/sec**         |                 \~4.10 |
| **Eval steps/sec**           |                 \~1.02 |
| **Total server runtime** (s) |               27 423.6 |

Minor fluctuations appeared (e.g., round 20 eval runtime spiked to \~891 s likely due to final aggregation overhead).

---

## 4. Comparative Analysis

### 4.1 Versus Previous 1000-Split LoRA Run

| Metric                 | 3000 total (1k/client) |                                 6000 total (2k/client) |
| ---------------------- | ---------------------: | -----------------------------------------------------: |
| **Δ Eval loss (0→10)** |   –0.604 (2.987→2.383) |                                   –0.693 (2.987→2.294) |
| **Convergence speed**  |               moderate |                                   faster early descent |
| **Stability**          |                 smooth | similarly smooth, with no overfitting even by round 20 |
| **Runtime**            |      \~10 486 s (10 r) |                                      \~27 424 s (20 r) |

Doubling per-client data accelerated convergence in the first 10 rounds (eval loss hit 2.294 by round 10 vs. 2.383 previously), and continued to decrease to 2.207 by round 20.

### 4.2 Versus Full-Model FedProx (No LoRA)

| Aspect            |  Full-Model FedProx |    LoRA-FedProx (6k) |
| ----------------- | ------------------: | -------------------: |
| **VRAM usage**    |        8 GB (spikes)|    consistently 8 GB |
| **RAM usage**     |   30–40 GB (spikes) |              < 11 GB |
| **Overfitting**   |       onset \~r6–r7 |       none up to r20 |
| **Comm. payload** |  full model weights | adapter weights only |

LoRA integration maintained memory efficiency, eliminated OOM risk, and prevented early overfitting.

---

## 5. Key Observations

1. **Data Volume Matters**
   Increasing local data from 1 000→2 000 samples/c delayed the eval‐loss plateau and drove steeper early descent.

2. **LoRA’s Efficiency**
   Only 0.47 % of parameters were trained, halving memory footprint and communication compared to full‐model updates.

3. **FedProx Regularization**
   μ = 0.1 kept client updates close to the global model, preventing drift and ensuring stable convergence.

4. **Scalability to 20 Rounds**
   No overfitting was observed even after 20 rounds, highlighting the combined effect of data volume, proximal term, and LoRA.

---

## 6. Conclusion

By combining LoRA with FedProx and increasing local data volume, a **memory-efficient** and **stable** federated fine-tuning pipeline for OPT-350M was demonstrated. The run with 6 000 samples and 20 rounds yielded smooth loss trajectories (final eval loss = 2.207) without overfitting or OOM errors. Future work will refine hyperparameters, expand to more clients, and compare alternative PEFT techniques to further validate and generalize these findings.

---

**Prepared by:**
Ivan Kosiakov
University of Nicosia
9 June 2025
