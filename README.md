# Memory-Efficient Federated Fine-Tuning of Large Language Models: A Dockerized Flower Framework with FedProx and LoRA

**Author:** Ivan Kosiakov  
**University:** University of Nicosia  
**Degree:** BSc Computer Science  
**Final Year Project (2024–2025)**

---

## 🧠 Project Overview

This project explores the application of **Federated Learning (FL)** to fine-tune a **Large Language Model (LLM)**—Meta’s `facebook/opt-350m`—across decentralized clients using the **Flower** framework and **Hugging Face Transformers**. The primary aim is to evaluate performance, convergence behavior, and training stability in scenarios where client data is private and non-IID.

---

## ⚙️ Architecture

- **Model:** [`facebook/opt-350m`](https://huggingface.co/facebook/opt-350m)
- **Frameworks:** Flower (FL), Hugging Face Transformers
- **Dataset:** [`codeparrot/codeparrot-clean-valid`](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid)
- **Setup:** Dockerized server + 3 client containers
- **Evaluation:** Centralized validation on a held-out server dataset

Each client trains on a distinct data partition and returns updated weights. The server aggregates updates using either **FedAvgM** or **FedProx** and evaluates the global model centrally.

---

## 📁 Repository Structure

```bash
.
├── client.py               # Federated client with Hugging Face Trainer
├── server.py               # Flower server with aggregation + evaluation
├── Dockerfile              # Docker image for client/server
├── docker-compose.yml      # Orchestration for full setup
├── !LOGS/                  # Logs and template codes
└── README.md               # This file
````


