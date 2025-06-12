**Federated Learning with OPT-350M + LoRA using FedProx: 6000-Sample Run**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Analysis Date:** 10 June 2025
**Strategy:** FedProx (μ = 0.01) with LoRA adapters
**Configuration:** 2000 samples/client × 3 clients (6 000 total), LoRA (r = 16, α = 32, dropout = 0.5), 3 local epochs, LR = 1 × 10⁻⁵, 20 rounds scheduled (completed at round 20)

---

## 1. Experiment Setup

| Component             | Configuration                        |
| --------------------- | ------------------------------------ |
| **Model**             | `facebook/opt-350m`                  |
| **FedProx μ**         | 0.01                                 |
| **PEFT**              | LoRA (r = 16, α = 32, dropout = 0.5) |
| **Dataset**           | `codeparrot/codeparrot-clean-valid`  |
| **Total Samples**     | 6000                                 |
| **Client Split**      | 2000 samples/client × 3 clients      |
| **Epochs per Client** | 3                                    |
| **Learning Rate**     | 1 × 10⁻⁵                             |
| **Batch Size**        | 2 (train), 4 (eval)                  |
| **Rounds Scheduled**  | 20                                   |
| **Rounds Completed**  | 20                                   |
| **Start Time**        | 2025-06-09 16:29:53.148              |
| **Finish Time**       | 2025-06-10 00:24:51.901              |
| **Total Runtime**     | 27 542.26 s (\~7.65 h)               |

> **Memory profile:**
> • VRAM: fully utilized (8 GB) during client training; freed during server aggregation.
> • RAM: remained under 11 GB throughout.

At the start of training, each process confirmed:

> **Trainable params:** 1 572 864 / 332 769 280 (0.47 %)

Only LoRA adapter matrices were fine-tuned.

---

## 2. Loss Trajectories

### 2.1 Distributed (Train) Loss

| Round | Aggregated Train Loss |
| ----: | --------------------: |
|     1 |             2.5740583 |
|     2 |             2.4835164 |
|     3 |             2.4303647 |
|     4 |             2.3956054 |
|     5 |             2.3687410 |
|     6 |             2.3467308 |
|     7 |             2.3278407 |
|     8 |             2.3115593 |
|     9 |             2.2968490 |
|    10 |             2.2838386 |
|    11 |             2.2722223 |
|    12 |             2.2612797 |
|    13 |             2.2512322 |
|    14 |             2.2419480 |
|    15 |             2.2332526 |
|    16 |             2.2253166 |
|    17 |             2.2177324 |
|    18 |             2.2103818 |
|    19 |             2.2039829 |
|    20 |             2.1974477 |

### 2.2 Centralized (Eval) Loss

| Round | Eval Loss |
| ----: | --------: |
|     0 | 2.9870501 |
|     1 | 2.5826068 |
|     2 | 2.4919791 |
|     3 | 2.4392557 |
|     4 | 2.4050660 |
|     5 | 2.3787718 |
|     6 | 2.3571267 |
|     7 | 2.3386862 |
|     8 | 2.3224993 |
|     9 | 2.3080649 |
|    10 | 2.2952959 |
|    11 | 2.2837636 |
|    12 | 2.2731729 |
|    13 | 2.2633379 |
|    14 | 2.2541234 |
|    15 | 2.2456467 |
|    16 | 2.2377224 |
|    17 | 2.2298963 |
|    18 | 2.2227581 |
|    19 | 2.2162294 |
|    20 | 2.2097862 |

---

## 3. Runtime & Throughput

| Metric                       | Value (per-round avg.) |
| ---------------------------- | ---------------------: |
| **Eval runtime** (s)         |                \~736 s |
| **Eval samples/sec**         |                 \~4.07 |
| **Eval steps/sec**           |                 \~1.02 |
| **Total server runtime** (s) |              27 542.26 |

Minor fluctuations appeared (e.g., round‑20 eval runtime spiked to \~891 s, likely due to final aggregation overhead).

---

## 4. Key Observations

1. **Smaller Proximal Term**
   Reducing μ from 0.1 to 0.01 yielded slightly faster early descent (train loss round 1→5: –0.206 vs. –0.184 previously) with no instability.

2. **LoRA Efficiency**
   Training only 0.47 % of parameters maintained memory efficiency and low communication overhead.

3. **Stable Convergence**
   No overfitting observed up to round 20; eval loss decreased smoothly to 2.2098.

4. **Consistent Throughput**
   Eval throughput remained around 4 samples/s despite slight runtime spikes.

---

## 5. Conclusion

FedProx with μ = 0.01 and LoRA on OPT-350M enabled a **memory-efficient**, **stable**, and **fast-converging** federated fine-tuning pipeline. The run achieved a final eval loss of 2.2098 without overfitting or OOM issues. Future work will explore different μ values, additional clients, and alternative PEFT methods to further optimize convergence and resource usage.

---

**Prepared by:**
Ivan Kosiakov
University of Nicosia
10 June 2025
