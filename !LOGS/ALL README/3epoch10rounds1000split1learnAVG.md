# Federated Learning with OPT-350M using Flower Framework

**Author:** Ivan Kosiakov
**Program:** BSc Computer Science, University of Nicosia
**Project:** Final Year Thesis – Federated Fine-Tuning of Large Language Models
**Supervisor:** \[To be filled]
**Report Date:** 29 May 2025

---

## 1. Introduction

This project explores the application of Federated Learning (FL) for training large language models (LLMs) using decentralized data. The central focus is on using Meta’s OPT-350M model within a multi-client FL architecture powered by the Flower framework. The goal is to evaluate the effectiveness of collaborative fine-tuning without centralizing data, leveraging the Hugging Face Transformers ecosystem for local training.

---

## 2. Objectives

* Implement Federated Learning using Flower with Hugging Face models
* Fine-tune OPT-350M on split datasets using `codeparrot/codeparrot-clean-valid`
* Compare federated strategies (FedAvg vs. FedAvgM)
* Measure convergence, resource efficiency, and model quality
* Investigate overfitting, regularization, and data augmentation

---

## 3. Experimental Setup

| Component          | Configuration                              |
| ------------------ | ------------------------------------------ |
| **Model**          | `facebook/opt-350m`                        |
| **Dataset**        | `codeparrot/codeparrot-clean-valid`        |
| **Split**          | 3000 samples total, 1000 per client        |
| **Epochs**         | 3                                          |
| **Learning Rate**  | 1e-5                                       |
| **Batch Size**     | 2 (per device)                             |
| **Strategy**       | FedAvg (not momentum-based)                |
| **Rounds**         | 10                                         |
| **Frameworks**     | Flower, PyTorch, Hugging Face Transformers |
| **Hardware**       | RTX 4060, 64GB RAM, Ryzen 7000             |
| **Start Time**     | 29 May 2025, 12:32 PM                      |
| **Total Duration** | \~9 hours 11 minutes (\~33088 seconds)     |

---

## 4. Training Results

### Centralized Eval Loss (Hugging Face Evaluation)

| Round | Eval Loss |
| ----- | --------- |
| 0     | 7.144     |
| 1     | 2.593     |
| 2     | 2.395     |
| 3     | 2.323     |
| 4     | 2.296     |
| 5     | 2.293     |
| 6     | 2.301     |
| 7     | 2.319     |
| 8     | 2.342     |
| 9     | 2.371     |
| 10    | 2.401     |

### Distributed Train Loss

| Round | Train Loss |
| ----- | ---------- |
| 1     | 2.658      |
| 2     | 2.477      |
| 3     | 2.411      |
| 4     | 2.390      |
| 5     | 2.392      |
| 6     | 2.408      |
| 7     | 2.433      |
| 8     | 2.460      |
| 9     | 2.497      |
| 10    | 2.534      |

---

## 5. Key Observations

* The model demonstrated a rapid initial loss drop from **7.14 to \~2.29** in the first 5 rounds.
* A gentle upward trend in both train and eval loss was noted after round 5, indicating slight overfitting or learning saturation.
* The average eval throughput was \~4.5 samples/sec with 0.56 steps/sec.
* Switching from FedAvgM to **FedAvg** produced comparable convergence performance in fewer rounds.

---

## 6. Insights and Challenges

* **Memory optimization:** Achieved through 16-bit weights and efficient garbage collection.
* **Training time per round:** \~55 minutes, acceptable for large-scale language model training.
* **Convergence:** Effective convergence in the first few rounds but performance plateaued by round 8–10.
* **Generalization:** Limited dataset size may constrain generalization.

---

## 7. What Worked Well

* Hugging Face Trainer integration simplified training and evaluation.
* Flower’s flexibility allowed easy switch between FedAvg and FedAvgM.
* Docker and retry logic enabled stable, long-running distributed training jobs.
* Accurate per-round logging supported clear visual and numeric analysis.

---

## 8. Future Plans

* **Incorporate Data Augmentation:** To increase model robustness on limited data.
* **Compare FL strategies:** Experiment with FedProx and FedOpt.
* **Early Stopping:** Introduce logic to detect and prevent overfitting.
* **Larger datasets:** Use 6K–10K splits across 3+ clients.
* **Visualization:** Graphs for eval/train loss, memory, and time per round.

---

## 9. Conclusion

This run demonstrated the feasibility of **Federated Learning with Large Language Models**, validating that even large models like OPT-350M can be trained collaboratively on limited hardware. Using **standard FedAvg**, the model converged quickly and stably on realistic code datasets. This forms a strong foundation for exploring augmentation, optimization, and privacy-preserving techniques in upcoming iterations.

---

**Prepared by:**
Ivan Kosiakov
29 May 2025

