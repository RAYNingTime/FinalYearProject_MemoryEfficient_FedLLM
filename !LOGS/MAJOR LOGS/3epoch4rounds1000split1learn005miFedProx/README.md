**Federated Learning with OPT-350M using FedProx (μ = 0.05): Experimental Report**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Experiment Date:** 8 June 2025
**Strategy:** FedProx (`proximal_mu = 0.05`)

---

## 1. Experiment Overview

This run replicates our previous federated fine-tuning of Meta’s **OPT-350M** on **CodeParrot-clean-valid**, adjusting only the proximal regularization coefficient from μ = 0.01 to μ = 0.05. Three clients (1 000 examples each) performed 3 local epochs at a learning rate of 1 × 10⁻⁵. While configured for 10 rounds, the experiment completed round 4 before a client-side CUDA error in round 5 prompted an early stop.

---

## 2. Setup

| Component             | Value                                               |
| --------------------- | --------------------------------------------------- |
| **Model**             | `facebook/opt-350m`                                 |
| **Dataset**           | `codeparrot/codeparrot-clean-valid`                 |
| **Total Samples**     | 3 000 (1 000 per client) + 1 000 server hold-out    |
| **Clients**           | 3 (each with 1 000 training examples)               |
| **Epochs per Client** | 3                                                   |
| **Learning Rate**     | 1 × 10⁻⁵                                            |
| **Batch Size**        | 2 (train), 4 (eval)                                 |
| **Federated Rounds**  | 10 (stopped after round 4)                          |
| **Strategy**          | FedProx (`proximal_mu = 0.05`)                      |
| **Aggregation Hook**  | Custom `MemoryEfficientFedProx` with `gc.collect()` |

---

## 3. Metric Summary

### 3.1 Centralized Evaluation Loss

Mean of the three clients’ `eval_loss` on the server hold-out set at round 0 (initial) and rounds 1–4:

| Round | Centralized Eval Loss |
| :---: | :-------------------: |
|   0   |        2.98705        |
|   1   |        2.07594        |
|   2   |        2.02588        |
|   3   |        2.02034        |
|   4   |        2.03139        |

> *Note:* Round 5 was not evaluated due to a client-side CUDA error.

### 3.2 Aggregated Train Loss

Weighted average of clients’ training losses reported at each round:

| Round | Aggregated Train Loss |
| :---: | :-------------------: |
|   1   |        2.07594        |
|   2   |        2.02588        |
|   3   |        2.02034        |
|   4   |        2.03139        |

---

## 4. Per-Round Client Eval Losses

#### Round 1

* Client 1: 1.99948
* Client 2: 1.94907
* Client 3: 2.11656

#### Round 2

* Client 1: 1.95653
* Client 2: 1.90193
* Client 3: 2.07187

#### Round 3

* Client 1: 1.95594
* Client 2: 1.89914
* Client 3: 2.07435

#### Round 4

* Client 1: 1.97177
* Client 2: 1.91138
* Client 3: 2.09101

---

## 5. Observations and Analysis

### 5.1 Initial Convergence (Rounds 0–3)

* **Rapid loss reduction:** Centralized eval loss dropped from **2.987 → 2.020** by round 3.
* **Strong alignment:** Aggregated train loss mirrored this trend (2.0759 → 2.0203), indicating effective regularization even with μ = 0.05.

### 5.2 Slight Plateau at Round 4

* **Minor uptick:** Centralized eval loss increased to **2.0314**, hinting at early signs of overfitting.
* **FedProx effect:** A higher μ slowed client drift but may also have constrained beneficial updates, leading to a plateau.

### 5.3 Client Heterogeneity

* **Per-client variance:** Within-round spreads of \~0.15 suggest non‑IID data splits still influence performance.
* **Stability trade-off:** μ = 0.05 reduced extreme divergences but could benefit from adaptive tuning per client.

### 5.4 Runtime Characteristics

* **Server round time:** Full `fit`+`evaluate` cycles averaged **∼700 s** per round.
* **Client eval throughput:** Varied between **9–65 samples/s** (200 eval examples, 4‑batch).

---

## 6. Key Takeaways

| Insight                       | Explanation                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Fast initial descent**      | μ = 0.05 still achieves rapid loss reduction through round 3.                                         |
| **Early plateau**             | A slight central eval loss increase by round 4 indicates over-regularization may limit further gains. |
| **Client data heterogeneity** | Non‑IID splits lead to \~0.15 loss variance; adaptive μ or weighting could help.                      |
| **Mu trade-offs**             | Higher proximal μ improves stability but risks slowing convergence beyond early rounds.               |

---

## 7. Recommendations

1. **Adaptive μ Scheduling**

   * Start at μ = 0.05 for rounds 1–3, then decay to μ = 0.01 in later rounds to balance stability and convergence.
2. **Early Stopping Criterion**

   * Halt when centralized eval loss increases for two consecutive rounds (e.g., at round 4).
3. **Reduce Local Epochs**

   * Try 1–2 epochs per round to mitigate overfitting on small client splits.
4. **PEFT Integration**

   * Incorporate LoRA or QLoRA to reduce VRAM usage and potentially enhance generalization.
5. **Client Weighting**

   * Explore heterogeneity-aware aggregation (FedOpt, FedNova) or sample‑size weighting to account for client variability.

---

## 8. Conclusion

This μ = 0.05 run confirms that increased proximal regularization enforces stable descent through round 3 but may induce early plateau around round 4. Future work should focus on adaptive μ schedules, early stopping, and parameter-efficient techniques (LoRA/QLoRA) to extend effective training rounds and improve generalization.

---

**Prepared by:** Ivan Kosiakov
**Date:** 8 June 2025
**Affiliation:** University of Nicosia
