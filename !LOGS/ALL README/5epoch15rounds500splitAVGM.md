## **Federated Learning with OPT-350M using Flower Framework: Initial Training Report**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Supervisor:** \[Professor's Name]
**Submission Date:** \[To be filled]

---

### 1. Introduction

This report presents the results of an early training experiment in the context of federated learning (FL) with large language models (LLMs). The setup involved using Meta's OPT-350M model, fine-tuned in a federated environment using the Flower framework. The goal was to assess how model performance changes across communication rounds when clients locally train on subsets of a code-based dataset (`codeparrot/codeparrot-clean-valid`).

---

### 2. Objectives

* Validate the end-to-end federated learning pipeline with 3 clients
* Evaluate model convergence behavior under moderately aggressive hyperparameters
* Track evaluation metrics and runtime statistics across multiple rounds
* Understand limitations related to overfitting and generalization in FL settings

---

### 3. Current Setup

* **Model:** Meta's OPT-350M (Hugging Face Transformers)
* **Framework:** Flower (v1.11+), Docker Compose for deployment
* **Dataset:** `codeparrot/codeparrot-clean-valid`
* **Evaluation:** Server-side centralized evaluation using Hugging Face `Trainer`
* **Hardware:** 64GB RAM, RTX 4060 GPU

#### Experiment Details

* **Epochs per client:** 5
* **Total samples:** 1500 (split 3×500 across clients)
* **Rounds:** 15
* **Learning Rate:** 5e-5
* **Strategy:** FedAvgM
* **Duration:** \~8.9 hours (32,049 seconds)

---

### 4. Training Results Summary (15 Rounds)

| Metric                      | Value  |
| --------------------------- | ------ |
| Initial Eval Loss (Round 0) | 7.14   |
| Best Eval Loss (Round 1)    | 2.64   |
| Final Eval Loss (Round 15)  | 3.70   |
| Final Train Loss (Round 15) | 3.78   |
| Evaluation Runtime (avg)    | \~338s |
| Samples/sec (eval)          | \~4.4  |
| Steps/sec (eval)            | \~0.55 |

#### Observations:

* Initial rounds showed rapid convergence (7.14 → 2.64) due to aggressive training.
* After round 5, both training and evaluation loss began **steadily increasing**.
* Overfitting likely started beyond round 6, as validation loss diverged.
* Training remained stable, with consistent runtime and throughput.

---

### 5. Insights and Lessons Learned

This initial training confirmed that **Federated Learning is viable for fine-tuning LLMs** across distributed clients. It highlighted the need for carefully tuned hyperparameters and monitoring tools to mitigate overfitting.

**Federated Learning helped achieve:**

* **Privacy preservation**: All training was local, with only model weights shared.
* **Scalability**: Clients were simulated using Docker containers for clean isolation.
* **Practical orchestration**: Flower allowed seamless coordination of rounds and evaluation.

Through this run, I better understood:

* The trade-off between **aggressive learning rates** and **generalization**
* How to monitor for **overfitting across federated clients**
* The importance of server-side centralized evaluation for comparability

---

### 6. Planned Improvements

* Switch to **lower learning rate (1e-5)** and **fewer epochs** for better generalization
* Implement **early stopping** or **round-capped decay** to prevent overfitting
* Use **larger training dataset (3000–6000)** per client in future experiments
* Introduce **dropout or weight decay** regularization
* Add evaluation on unseen (non-training) data partitions

---

### 7. Deliverables

* Federated training pipeline with Hugging Face + Flower + Docker
* Full logs and training metadata from all rounds
* Evaluation and training metrics visualized in Excel/CSV
* Final Year Project report and presentation-ready summary

---

**Prepared by:**
Ivan Kosiakov
26 May 2025