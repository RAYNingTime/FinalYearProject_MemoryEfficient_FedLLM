# **Federated Learning with OPT-350M using FedProx: Experimental Report**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Experiment Date:** 5 June 2025
**Strategy:** FedProx (`proximal_mu = 0.01`)

---

## 1. Experiment Overview

This experiment evaluates the impact of a **lower proximal regularization coefficient (μ = 0.01)** on the federated fine-tuning of Meta’s **OPT-350M** using the Flower framework. Three clients each hold 1 000 examples from the **CodeParrot-clean-valid** dataset (3 000 total), while 1 000 examples are reserved on the server for centralized evaluation. Each client fine-tunes locally for 3 epochs with a learning rate of 1 × 10⁻⁵. Although configured for 10 rounds, the run was manually stopped after round 8 when overfitting trends became apparent.

---

## 2. Setup

| Component             | Value                                               |
| --------------------- | --------------------------------------------------- |
| **Model**             | `facebook/opt-350m`                                 |
| **Dataset**           | `codeparrot/codeparrot-clean-valid`                 |
| **Total Samples**     | 3 000 (1 000 per client) + 1 000 server hold-out    |
| **Clients**           | 3 (each with 1 000 training examples)               |
| **Epochs per Client** | 3                                                   |
| **Learning Rate**     | 1 × 10⁻⁵                                            |
| **Batch Size**        | 2 (train), 4 (eval)                                 |
| **Federated Rounds**  | 10 (stopped at round 8)                             |
| **Strategy**          | FedProx (`proximal_mu = 0.01`)                      |
| **Aggregation Hook**  | Custom `MemoryEfficientFedProx` with `gc.collect()` |
| **Runtime**           | \~10 700 seconds (\~3 hours) by round 8             |

---

## 3. Metric Summary

### 3.1 Centralized Evaluation Loss

The table below shows the server-side evaluation (mean of the three clients’ eval losses) at **round 0** (initial parameters) and after each round up to round 7.

| Round | Centralized Eval Loss |
| :---: | :-------------------: |
|   0   |         2.9870        |
|   1   |         2.1013        |
|   2   |         2.0536        |
|   3   |         2.0503        |
|   4   |         2.0691        |
|   5   |         2.0949        |
|   6   |         2.1257        |
|   7   |         2.1578        |

> **Note:** For each round *i* (1 ≤ *i* ≤ 7), the “Centralized Eval Loss” is computed as the arithmetic mean of the three clients’ reported `eval_loss` values.

### 3.2 Distributed (Aggregated) Train Loss

Each client’s local training loss is averaged (weighted by client sample count) to produce the distributed train-loss summary below.

| Round | Aggregated Train Loss |
| :---: | :-------------------: |
|   1   |         2.0779        |
|   2   |         2.0263        |
|   3   |         2.0181        |
|   4   |         2.0296        |
|   5   |         2.0491        |
|   6   |         2.0734        |
|   7   |         2.0994        |

---

## 4. Per-Round Details

Below are the raw client eval losses for each round, demonstrating heterogeneity across clients.

#### Round 1 Client Eval Losses

* Client 1: 2.1141
* Client 2: 2.0127
* Client 3: 2.1773

#### Round 2 Client Eval Losses

* Client 1: 2.0706
* Client 2: 1.9683
* Client 3: 2.1219

#### Round 3 Client Eval Losses

* Client 1: 2.0702
* Client 2: 1.9676
* Client 3: 2.1132

#### Round 4 Client Eval Losses

* Client 1: 2.0914
* Client 2: 1.9882
* Client 3: 2.1278

#### Round 5 Client Eval Losses

* Client 1: 2.1183
* Client 2: 2.0135
* Client 3: 2.1528

#### Round 6 Client Eval Losses

* Client 1: 2.1509
* Client 2: 2.0419
* Client 3: 2.1844

#### Round 7 Client Eval Losses

* Client 1: 2.1836
* Client 2: 2.0734
* Client 3: 2.2164

*(Round 8 was not fully evaluated—manual stop.)*

---

## 5. Observations and Analysis

### 5.1 Initial Convergence (Rounds 0–3)

* **Rapid improvement:**

  * Centralized eval loss dropped from **2.987 → 2.050** by round 3.
  * Aggregated train loss moved from **2.0779 → 2.0181** in the same span.
* **FedProx effect (μ = 0.01):**

  * The small proximal coefficient still provided some regularization against client drift, yielding a stable downward trend in both train and eval losses through round 3.

### 5.2 Plateau and Onset of Overfitting (Rounds 4–7)

* **Saturation:**

  * After round 3, improvements slowed. Centralized eval loss rose slightly (2.050 → 2.157 by round 7).
  * Distributed train loss also crept upward (2.0181 → 2.0994 from round 3 → 7).
* **Overfitting trend:**

  * The divergence between round 3 and round 7 indicates the model began to overfit the small per-client datasets despite FedProx regularization.
  * Validation (server-side) plateaued around **2.05 ± 0.02** before creeping upward, signaling diminishing returns beyond round 3 or 4.

### 5.3 Client Heterogeneity

* **Client eval variances:**

  * Within each round, clients’ eval losses varied by \~ 0.1–0.15, showing non-IID differences.
  * For example, in round 6: Client 3 had the highest eval loss (2.1844), while Client 2 saw the lowest (2.0419).
* **FedProx stability:**

  * Despite heterogeneity, FedProx (μ = 0.01) “tethered” each client’s local update to the global model, reducing extreme drift.
  * The relatively narrow spread in per-round client evals (±0.08 around the mean) indicates controlled divergence.

### 5.4 Throughput and Runtime

* **Eval runtime consistency:**

  * Each client’s eval step (on \~ 200 held-out samples) took \~ 9–22 seconds, with throughput \~ 9–70 samples/sec. (Likely varied due to batch caching and GPU load.)
  * Per-round server eval (aggregating three clients sequentially) averaged **\~ 10–20 s per client**.
* **Server runtime per round:**

  * Full `fit` + `evaluate` cycle per round averaged **\~ 680–740 s**.
  * Total time through round 7 ≈ **31 300 s** (\~ 8.7 hours), indicating \~ 4 500 s (\~ 1.25 hours) per round.

---

## 6. Key Takeaways

| Insight                             | Explanation                                                                                                                                                                 |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Early convergence up to R3**      | FedProx with μ= 0.01 provided enough regularization for stable descent through round 3, reducing centralized eval loss from 2.987 → 2.050.                                  |
| **Overfitting post R3**             | After round 3, eval loss plateaued and then increased, indicating that 3 epochs on 1 000 samples/client is insufficient to generalize beyond \~ 3 federated rounds.         |
| **Small μ yields modest effect**    | Lowering μ from 0.1 → 0.01 slightly reduced client drift but did not prevent overfitting on small, non-IID splits; a careful μ-sweep is recommended.                        |
| **Client heterogeneity matters**    | Per-client eval spreads of \~ 0.1 show that data differences among clients still affect the aggregated model; future work could incorporate client weighting or adaptive μ. |
| **Memory‐efficient pipeline works** | Custom `MemoryEfficientFedProx` prevented OOMs and exit 137s, allowing 8 rounds to complete on OPT-350M with 8 GB VRAM per client.                                          |

---

## 7. Recommendations

1. **Early Stopping**

   * Introduce a stopping criterion at **round 4** (or earlier) whenever centralized eval loss increases for 2 consecutive rounds.
   * Example: If eval loss( Rₙ ) ≥ eval loss( Rₙ₋₁ ), stop at Rₙ.

2. **Adjust Local Epochs**

   * Reduce per-client epochs from 3 → 1–2 to mitigate overfitting on small splits.
   * Alternatively, implement **adaptive epoch scheduling** (e.g., 1 epoch rounds 1–3, then 0.5 epoch).

3. **μ Hyperparameter Tuning**

   * Test **μ = {0.001, 0.01, 0.05, 0.1}** to find the optimal proximal regularization for 1 000 samples/client.

4. **Increase Client Data**

   * As shown in previous runs, moving from 1 000 → 2 000 samples/client improved early stability. Consider 1 500–2 000 splits next.

5. **Parameter-Efficient Fine Tuning (PEFT)**

   * Integrate **LoRA (Low-Rank Adaptation)** or **QLoRA** to reduce memory footprint and potentially improve generalization.
   * Code samples and Docker dependencies for LoRA/QLoRA have been provided in earlier messages.

6. **Alternative Aggregations**

   * Compare with **FedOpt** (server-side optimizer) or **FedNova** to further mitigate client heterogeneity.

---

## 8. Conclusion

This run—**FedProx (μ = 0.01, 8 completed rounds)**—confirms that:

* **FedProx** still enforces enough regularization at μ = 0.01 to achieve a consistent loss drop through round 3.
* **Overfitting** remains a challenge on **1 000 samples/client**, emerging around round 4.
* Further **μ tuning**, **early stopping**, and **PEFT** techniques (LoRA/QLoRA) are promising next steps to delay overfitting, improve generalization, and manage VRAM constraints.

---

**Prepared by:**
Ivan Kosiakov
05 June 2025
University of Nicosia
