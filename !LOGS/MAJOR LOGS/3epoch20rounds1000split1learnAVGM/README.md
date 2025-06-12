**Federated Learning with OPT-350M using Flower Framework: Mid-Project Report**

**Student:** Ivan Kosiakov
**Degree Program:** BSc Computer Science, University of Nicosia
**Project Title:** Federated Learning Experiments with Large Language Models
**Supervisor:** \[Professor's Name]
**Submission Date:** \[To be filled]

---

### 1. Introduction

The goal of this final year project is to explore the integration of Federated Learning (FL) techniques with Large Language Models (LLMs), specifically Meta's OPT-350M. The project investigates the feasibility, memory usage, and performance trade-offs of fine-tuning LLMs across decentralized clients using the Flower FL framework and Hugging Face ecosystem.

---

### 2. Objectives

* Deploy a multi-client federated training setup using Docker and Flower
* Fine-tune OPT-350M model on client-specific datasets (CodeParrot-clean-valid)
* Aggregate updates centrally and evaluate a global model on a shared test set
* Monitor resource utilization (memory, time per round, communication overhead)
* Explore parameter-efficient fine-tuning (planned)

---

### 3. Current Setup

* **Model:** Meta's OPT-350M via Hugging Face Transformers
* **Framework:** Flower (v1.11+), with Docker Compose for orchestration
* **Data:** `codeparrot/codeparrot-clean-valid` (code-based language modeling dataset)
* **Evaluation:** Server-side global validation using Hugging Face's `Trainer`
* **Infrastructure:** Local machine with 64GB RAM, RTX 4060 GPU

#### Architecture Overview

* Flower SuperLink server handles training rounds
* Clients implemented as Flower NumPyClients using Hugging Face `Trainer`
* Client containers simulate private data and perform local training
* Server aggregates model weights, runs evaluation, and logs metrics

---

### 4. Achievements So Far

* Successfully deployed federated learning using Docker containers
* Clients load and train OPT-350M on their data subsets
* Global model aggregates and is evaluated on shared test set
* Implemented memory-efficient strategy to avoid OOM (exit code 137)
* Collected per-round metrics (loss, evaluation scores)
* Integrated logging for reproducibility and debugging
* Performed a complete 20-round training experiment using real-world federated setup

---

### 5. Training Experiment Summary (20 Rounds)

* **Dataset:** `codeparrot/codeparrot-clean-valid`
* **Split:** 3000 samples total, 1000 per client
* **Epochs per client:** 3
* **Learning rate:** 1e-5
* **Federated Strategy:** FedAvgM
* **Model:** `facebook/opt-350m`
* **Number of rounds:** 20

#### Observations:

* Initial eval loss dropped significantly: from 7.14 (round 0) to \~2.3 by round 4.
* Post round 10, eval loss slowly increased, indicating early signs of overfitting or model saturation.
* Distributed and centralized metrics closely aligned in trend.
* Runtime per round remained stable (\~660s), with good throughput (\~4.5 samples/sec).

---

### 6. Insights and Lessons Learned

This training run confirmed that **Federated Learning is a practical and effective way to train large language models** across distributed clients with limited resources. By distributing the workload and avoiding centralized data collection, it was possible to:

* Maintain user privacy while achieving solid convergence
* Monitor how model behavior varies across rounds with decentralized updates
* Detect overfitting trends and improve metric analysis methodology
* Learn how to orchestrate real-world LLM training at scale using Flower

The **FedAvgM strategy** proved memory-efficient and enabled 3 clients to successfully train OPT-350M without crashes. Logging tools and Hugging Face integration were essential for performance tracking.

---

### 7. Planned Improvements

* **Early stopping mechanism** to prevent overfitting after a loss increase
* **Increased dataset size** to improve model generalization
* **Regularization strategies** like weight decay or dropout
* **Experiment with FedProx and FedOpt** strategies to address data heterogeneity
* **Plotting and result visualization:** Graphs of eval loss, train loss, and runtime

---

### 8. Deliverables

* Complete Dockerized codebase with README and execution guide
* Full project report with motivation, architecture, experiments, results, and discussion
* Logs and metric charts for reproducibility
* Optional: presentation slides if required

---

**Prepared by:**
Ivan Kosiakov
28 May 2025
