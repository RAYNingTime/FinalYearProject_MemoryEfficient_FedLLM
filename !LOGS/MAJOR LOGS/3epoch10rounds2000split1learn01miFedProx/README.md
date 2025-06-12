## **Federated Learning with OPT-350M using FedProx: Experimental Report**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Experiment Date:** 2 June 2025
**Strategy:** FedProx (`proximal_mu=0.1`)

---

### **1. Experiment Overview**

This experiment explores the impact of increased client-side data volume on federated fine-tuning of Meta’s OPT-350M model using the Flower framework with the **FedProx strategy**. Each client received 2000 training samples, a significant increase from the earlier 1000-sample split. This was done to analyze whether overfitting onset could be delayed and convergence quality improved.

---

### **2. Configuration Summary**

| Component             | Value                               |
| --------------------- | ----------------------------------- |
| **Model**             | `facebook/opt-350m`                 |
| **Dataset**           | `codeparrot/codeparrot-clean-valid` |
| **Total Samples**     | 6000                                |
| **Client Split**      | 2000 per client × 3 clients         |
| **Server Test Set**   | 1000 (held out)                     |
| **Epochs per Client** | 3                                   |
| **Learning Rate**     | 1e-5                                |
| **Batch Size**        | 2 (train), 4 (eval)                 |
| **Rounds**            | 10                                  |
| **Strategy**          | FedProx (`proximal_mu=0.1`)         |
| **Runtime**           | 45,686 seconds (\~12.7 hours)       |

---

### **3. Results Summary**

#### ✅ **Centralized Evaluation Loss (Test on Shared Dataset)**

| Round | Eval Loss |
| ----- | --------- |
| 0     | 2.987     |
| 1     | 1.973     |
| 2     | 1.913     |
| 3     | 1.898     |
| 4     | 1.898     |
| 5     | 1.906     |
| 6     | 1.921     |
| 7     | 1.939     |
| 8     | 1.963     |
| 9     | 1.990     |
| 10    | 2.018     |

#### 📉 **Distributed Training Loss (Aggregated)**

| Round | Train Loss |
| ----- | ---------- |
| 1     | 2.001      |
| 2     | 1.958      |
| 3     | 1.959      |
| 4     | 1.973      |
| 5     | 1.995      |
| 6     | 2.024      |
| 7     | 2.056      |
| 8     | 2.091      |
| 9     | 2.131      |
| 10    | 2.170      |

---

### **4. Observations and Analysis**

#### ✅ **Improved Generalization**

* Initial evaluation loss dropped **from 2.987 to 1.898 by round 4**, showing strong generalization improvements over previous smaller-data runs.
* Increasing data per client (2× previous amount) contributed significantly to performance gains in early rounds.

#### ⚠️ **Overfitting Still Appears After Round 5**

* Both evaluation and training loss curves **started rising after round 5**, suggesting that despite the data increase, overfitting still occurred.
* Compared to the 1000-sample split experiment, **the increase in loss was slower and less steep**, which indicates better stability.

#### 🧠 **FedProx Contribution**

* The proximal term helped **slow model drift**, particularly in early rounds.
* Stable convergence suggests FedProx with `μ=0.1` is effective in multi-client non-IID settings.

#### ⏱️ **Runtime and Throughput**

| Metric                    | Average Value (approx.)       |
| ------------------------- | ----------------------------- |
| Eval runtime per round    | \~680–730s                    |
| Samples per second (eval) | \~4.3–4.4                     |
| Steps per second (eval)   | \~0.52–0.55                   |
| Server total runtime      | 45,686 seconds (\~12.7 hours) |

---

### **5. Lessons Learned**

| Area                 | Takeaway                                                                                                  |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| **Dataset size**     | Increasing data from 1000 → 2000 samples/client helped reduce early overfitting and improved convergence  |
| **FedProx**          | Continued to provide regularization against client drift and worked well in 3-client setup                |
| **Model saturation** | Beyond round 5, adding rounds yielded diminishing returns and loss slowly increased again                 |
| **Runtime scaling**  | Runtime nearly doubled compared to 1000-sample runs, suggesting a trade-off between data quality and time |

---

### **6. Next Steps & Ideas**

* 📉 **Try LoRA or adapter tuning** to enable deeper experiments with larger datasets and longer rounds.
* 🧪 **Experiment with FedOpt** or lower `proximal_mu` to fine-tune stability.
* 🛑 **Test early stopping with round 5 as threshold**, or adjust epochs to 2 per round.
* 🧠 **Include gradient clipping or weight decay** for better regularization.
* 📊 **Generate graphs** of eval/train loss per round to visualize overfitting point clearly.

---

### **7. Conclusion**

This run demonstrated that scaling up client-side data improves early training quality and convergence stability. While overfitting trends still emerged, the delay in onset and better early-round metrics validate the strategy. The FedProx regularization continues to offer a valuable foundation for federated fine-tuning of LLMs, though future runs will benefit from **adaptive training schedules and regularization methods**.

---

**Prepared by:**
Ivan Kosiakov
2 June 2025
University of Nicosia

