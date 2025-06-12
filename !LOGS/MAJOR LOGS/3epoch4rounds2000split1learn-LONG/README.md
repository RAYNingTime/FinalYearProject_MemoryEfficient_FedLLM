### **Federated Learning with OPT-350M using Flower Framework: Interim Evaluation Report**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Supervisor:** \[Professor's Name]
**Submission Date:** \[To be filled]

---

### 1. Introduction

This report presents the results of a short yet informative Federated Learning (FL) experiment using Meta’s OPT-350M language model. The purpose of this run was to examine early convergence trends, evaluate training behavior in a realistic decentralized setup, and help inform future hyperparameter tuning.

---

### 2. Objectives

* Measure early-stage convergence across FL rounds
* Evaluate stability of centralized validation metrics
* Benchmark resource usage and runtime expectations
* Establish a baseline for longer experiments

---

### 3. Experiment Configuration

| Parameter              | Value                               |
| ---------------------- | ----------------------------------- |
| **Model**              | `facebook/opt-350m`                 |
| **Dataset**            | `codeparrot/codeparrot-clean-valid` |
| **Data Split**         | 3000 samples (1000 per client)      |
| **Epochs per Client**  | 3                                   |
| **Learning Rate**      | `1e-5`                              |
| **Federated Strategy** | `FedAvgM`                           |
| **Number of Rounds**   | 4                                   |
| **Platform**           | Flower + Hugging Face Transformers  |
| **Hardware**           | Local: 64GB RAM, RTX 4060 GPU       |

---

### 4. Training Timeline

| Round | Start Time | Eval Loss | Cumulative Time (hh\:mm\:ss) | Samples/sec | Steps/sec |
| ----- | ---------- | --------- | ---------------------------- | ----------- | --------- |
| 0     | —          | 7.129     | —                            | 4.49        | 0.561     |
| 1     | 02:06      | 2.304     | 02:06:00                     | 4.506       | 0.563     |
| 2     | 04:30      | 2.145     | 04:30:00                     | 4.504       | 0.563     |
| 3     | 06:54      | 2.086     | 06:54:00                     | 4.482       | 0.560     |
| 4     | 09:14      | 2.059     | 09:06:00 (Finished \~11:12)  | 4.492       | 0.561     |

* **Total Wall Time:** \~9 hours 6 minutes (from round 1 start to round 4 completion)

---

### 5. Observations

* **Significant convergence in first round**: From 7.13 to 2.30, indicating strong initial learning.
* **Gradual improvement after round 1**, with diminishing returns per round — reaching 2.059 by round 4.
* **Stable performance and throughput** across rounds (\~4.5 samples/sec).
* Loss values indicate model continues to improve but may benefit from either more data or fine-tuning strategies to sustain gains.

---

### 6. Value of Federated Learning

This test validated **Federated Learning as a viable technique** for training LLMs on decentralized hardware. Key takeaways:

* No central data pooling was needed — client privacy preserved.
* The global model improved in only a few rounds with local updates.
* Coordinating clients and the server in a realistic setup provided **valuable experience in orchestration, memory management, and fault tolerance**.
* The `FedAvgM` strategy handled large model updates efficiently even with constrained system memory.

---

### 7. Future Plans

* Conduct longer experiments (≥10 rounds) for deeper insights into overfitting/generalization
* Compare `FedAvgM` with other strategies like **FedProx**, **FedOpt**
* Add **weight decay or dropout** to mitigate overfitting
* Introduce **validation loss plotting and automatic early stopping**
* Scale up to 4+ clients or larger datasets to simulate more diverse environments

---

### 8. Deliverables

* Reproducible Docker-based setup
* Centralized logs and metrics for plotting
* Structured experiment reports (including this one)
* Final written report with technical and research findings

---

**Prepared by:**
Ivan Kosiakov
29 May 2025
