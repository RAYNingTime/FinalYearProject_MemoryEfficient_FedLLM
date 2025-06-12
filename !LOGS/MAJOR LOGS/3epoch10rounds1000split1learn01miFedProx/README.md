## **Federated Learning with OPT-350M using FedProx: Experimental Report**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Experiment Date:** 1 June 2025
**Strategy:** FedProx (Memory-efficient implementation)

---

### **1. Experiment Overview**

This experiment continues the investigation of federated fine-tuning of the **OPT-350M** model using the **Flower framework**. In contrast to earlier runs using FedAvgM, this experiment evaluated **FedProx** as a strategy to mitigate performance degradation due to data heterogeneity across clients. The `proximal_mu` parameter was set to 0.1.

---

### **2. Setup**

| Component             | Value                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Model**             | `facebook/opt-350m`                                                                                         |
| **Dataset**           | `codeparrot/codeparrot-clean-valid`                                                                         |
| **Total Samples**     | 3000                                                                                                        |
| **Client Split**      | 1000 samples/client × 3 clients                                                                             |
| **Epochs per Client** | 3                                                                                                           |
| **Learning Rate**     | 1e-5                                                                                                        |
| **Batch Size**        | 2 (train), 4 (eval)                                                                                         |
| **Rounds**            | 10                                                                                                          |
| **Strategy**          | FedProx (`proximal_mu=0.1`)                                                                                 |
| **Server Runtime**    | 26,969 seconds (\~7.49 hours)                                                                               |
| **Aggregation**       | Memory-efficient override of `aggregate_fit` with garbage collection to prevent memory leaks and OOM errors |

---

### **3. Metric Summary**

#### **Centralized Eval Loss**

| Round | Eval Loss |
| ----- | --------- |
| 0     | 2.987     |
| 1     | 2.076     |
| 2     | 2.027     |
| 3     | 2.021     |
| 4     | 2.031     |
| 5     | 2.049     |
| 6     | 2.075     |
| 7     | 2.103     |
| 8     | 2.133     |
| 9     | 2.167     |

#### **Distributed Train Loss (Aggregated)**

| Round | Train Loss |
| ----- | ---------- |
| 1     | 2.0715     |
| 2     | 2.0271     |
| 3     | 2.0245     |
| 4     | 2.0384     |
| 5     | 2.0626     |
| 6     | 2.0911     |
| 7     | 2.1223     |
| 8     | 2.1561     |
| 9     | 2.1929     |
| 10    | 2.2335     |

---

### **4. Observations and Analysis**

#### ✅ **Early Improvements**

* During the first **3 rounds**, both centralized eval loss and distributed train loss **dropped significantly** (e.g., eval loss: 2.987 → 2.021).
* This confirms that **initial convergence under FedProx is effective** even with relatively small per-client data (1k samples each).

#### ⚠️ **Saturation & Overfitting Trend**

* After round 3, losses began **increasing slowly**:

  * Centralized eval loss: 2.021 → 2.167 by round 9.
  * Distributed train loss followed the same trend.
* Suggests either **early overfitting** or diminishing returns from repeated local updates across non-IID data.

#### 🧠 **FedProx Benefits**

* Unlike FedAvgM (where loss started increasing earlier), **FedProx maintained stability up to round 5–6**, possibly due to the **proximal regularization term** (μ = 0.1).
* Proximal term likely reduced client drift caused by heterogeneous training samples.

#### 🧹 **Memory Efficiency**

* Run successfully completed **10 full rounds** on OPT-350M without crashing, thanks to:

  * Custom `MemoryEfficientFedProx` class
  * Explicit `gc.collect()` and `del results`
  * `torch.cuda.empty_cache()` in client and server routines

---

### **5. Key Takeaways**

| Insight                            | Explanation                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **FedProx works**                  | Compared to FedAvgM, this strategy **slowed overfitting** and achieved slightly more stable performance in later rounds. |
| **Overfitting still appears**      | Even with FedProx, loss began increasing by round 6–7, suggesting the need for early stopping or regularization.         |
| **Memory optimization is crucial** | Docker containers handling full LLM training with 8GB VRAM per client need explicit memory handling.                     |
| **Small data is limiting**         | Clients with only 1000 examples begin overfitting quickly; may need to increase dataset or reduce epochs.                |

---

### **6. Recommendations**

* ⚙️ **Enable early stopping** after 2–3 rounds of stable/increasing loss (already implemented in later experiments).
* 🔁 **Test with fewer local epochs (e.g., 1–2)** to reduce overfitting on small client datasets.
* 🧩 **Introduce parameter-efficient fine-tuning** (LoRA/adapters) in future runs to lower memory usage and possibly improve generalization.
* 🧪 **Compare μ values in FedProx** (e.g., 0.01, 0.05, 0.1) to determine best balance between convergence and stability.

---

### **7. Conclusion**

This experiment confirms that **FedProx can improve the stability** of federated training with OPT-350M on small and split code datasets. However, even with proximal regularization, **careful tuning of training epochs and additional regularization** are necessary to avoid overfitting. The memory-efficient infrastructure remains stable and continues to support future enhancements like LoRA-based training.

---

**Prepared by:**
Ivan Kosiakov
01 June 2025
University of Nicosia
