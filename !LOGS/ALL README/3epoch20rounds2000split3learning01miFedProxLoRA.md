# **Federated Learning with OPT-350M + LoRA using FedProx: 6000-Sample Run (μ = 0.1)**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Analysis Date:** 10 June 2025
**Strategy:** FedProx (μ = 0.1) with LoRA adapters
**Configuration:** 2000 samples/client × 3 clients (6 000 total), LoRA (r = 16, α = 32, dropout = 0.5), 3 local epochs, LR = 3 × 10⁻⁵, 20 rounds scheduled (completed at round 20)

---

## 1. Experiment Setup

| Component             | Configuration                        |
| --------------------- | ------------------------------------ |
| **Model**             | `facebook/opt-350m`                  |
| **FedProx μ**         | 0.1                                  |
| **PEFT**              | LoRA (r = 16, α = 32, dropout = 0.5) |
| **Dataset**           | `codeparrot/codeparrot-clean-valid`  |
| **Total Samples**     | 6000                                 |
| **Client Split**      | 2000 samples/client × 3 clients      |
| **Epochs per Client** | 3                                    |
| **Learning Rate**     | 3 × 10⁻⁵                             |
| **Batch Size**        | 2 (train), 4 (eval)                  |
| **Rounds Scheduled**  | 20                                   |
| **Rounds Completed**  | 20                                   |
| **Start Time**        | 2025-06-09 11:30:00 (approx)         |
| **Finish Time**       | 2025-06-09 19:05:16 (approx)         |
| **Total Runtime**     | 27 116.57 s (\~7.53 h)               |

> **Memory profile:**
> • VRAM: fully utilized (8 GB) during client training; freed during server aggregation.
> • RAM: stayed below 11 GB.

At the start of training, each process confirmed:

> **Trainable params:** 1 572 864 / 332 769 280 (0.47 %)

Only LoRA adapter matrices were fine-tuned.

---

## 2. Loss Trajectories

### 2.1 Distributed (Train) Loss

| Round | Aggregated Train Loss |
| ----: | --------------------: |
|     1 |                2.4491 |
|     2 |                2.3693 |
|     3 |                2.3214 |
|     4 |                2.2869 |
|     5 |                2.2595 |
|     6 |                2.2374 |
|     7 |                2.2187 |
|     8 |                2.2017 |
|     9 |                2.1873 |
|    10 |                2.1746 |
|    11 |                2.1627 |
|    12 |                2.1522 |
|    13 |                2.1437 |
|    14 |                2.1350 |
|    15 |                2.1272 |
|    16 |                2.1202 |
|    17 |                2.1143 |
|    18 |                2.1075 |
|    19 |                2.1024 |
|    20 |                2.0961 |

### 2.2 Centralized (Eval) Loss

| Round | Eval Loss |
| ----: | --------: |
|     0 |    2.9871 |
|     1 |    2.4409 |
|     2 |    2.3590 |
|     3 |    2.3101 |
|     4 |    2.2744 |
|     5 |    2.2461 |
|     6 |    2.2225 |
|     7 |    2.2030 |
|     8 |    2.1859 |
|     9 |    2.1708 |
|    10 |    2.1575 |
|    11 |    2.1458 |
|    12 |    2.1354 |
|    13 |    2.1259 |
|    14 |    2.1173 |
|    15 |    2.1090 |
|    16 |    2.1014 |
|    17 |    2.0945 |
|    18 |    2.0878 |
|    19 |    2.0819 |
|    20 |    2.0753 |

---

## 3. Runtime & Throughput

| Metric                       | Value (per-round avg.) |
| ---------------------------- | ---------------------: |
| **Eval runtime** (s)         |                \~677 s |
| **Eval samples/sec**         |                 \~4.42 |
| **Eval steps/sec**           |                 \~1.11 |
| **Total server runtime** (s) |              27 116.57 |

Evaluation time remained steady with minor variance; memory usage and throughput were consistent with previous LoRA runs.

---

## 4. Key Observations

1. **Stronger Regularization**
   A higher μ = 0.1 introduced stronger proximal regularization. While convergence was marginally slower early on, it resulted in steady improvement without oscillation across all rounds.

2. **Smooth Loss Descent**
   Train loss declined from 2.45 to 2.09 over 20 rounds. Centralized eval loss dropped from 2.987 to 2.075, showing consistent generalization gains.

3. **LoRA Adapter Efficiency**
   The small fine-tuning footprint (0.47 % of parameters) maintained computational efficiency, which is ideal for federated systems with resource constraints.

4. **No Instability or Divergence**
   Despite higher μ, no divergence or degradation was observed. FedProx kept client updates well-aligned with the global objective.

---

## 5. Conclusion

FedProx with μ = 0.1 and LoRA adapters proved to be a **robust** and **efficient** configuration for federated fine-tuning of OPT-350M. It delivered **stable convergence**, effective regularization, and a final centralized evaluation loss of **2.0753**. These results validate the use of proximal terms in settings with client variability and set a baseline for tuning μ in future experiments.

---

**Prepared by:**
Ivan Kosiakov
University of Nicosia
10 June 2025

---
