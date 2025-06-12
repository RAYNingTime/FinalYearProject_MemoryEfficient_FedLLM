import flwr as fl
import gc
from flwr.server.strategy import FedProx
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict, set_peft_model_state_dict


# ======= USER-CONFIGURABLE PARAMETERS =======
MODEL_NAME = "facebook/opt-350m"                     # Base LLM
DATASET_NAME = "codeparrot/codeparrot-clean-valid"   # Dataset used for training/eval
TOTAL_EXAMPLES = 6000                                # Total dataset size
DATA_SPLIT = 2000                                    # Split per client (for manual client setups)
NUM_ROUNDS = 20                                      # Federated training rounds
PROXIMAL_MU = 0.1                                    # FedProx proximal term coefficient
LORA_R = 16                                          # LoRA rank
LORA_ALPHA = 32                                      # LoRA alpha
LORA_DROPOUT = 0.05                                  # LoRA dropout
EVAL_BATCH_SIZE = 4                                  # Eval batch size per device
MAX_LENGTH = 128                                     # Max sequence length for tokenization


# ======= LOAD MODEL AND TOKENIZER =======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# ======= APPLY LoRA CONFIGURATION =======
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Display percentage of tunable parameters

# ======= LOAD AND TOKENIZE DATASET =======
dataset = load_dataset(DATASET_NAME, split="train").select(range(TOTAL_EXAMPLES))

def tokenize(batch):
    return tokenizer(batch["content"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

# ======= EVALUATION SETUP =======
eval_args = TrainingArguments(
    output_dir="./eval_logs",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    report_to="none",
    logging_strategy="no",
    save_strategy="no"
)

trainer = Trainer(model=model, args=eval_args, eval_dataset=dataset, tokenizer=tokenizer)

# ======= EVALUATION FUNCTION =======
def get_evaluate_fn(model, eval_dataset, tokenizer):
    def evaluate_fn(server_round, parameters, config):
        # Reconstruct adapter weights from parameter list
        fake_state = get_peft_model_state_dict(model, model.state_dict())
        keys = list(fake_state.keys())

        # Reconstruct adapter state
        adapter_state = dict(zip(keys, [torch.tensor(p) for p in parameters]))
        set_peft_model_state_dict(model, adapter_state)

        # Evaluate
        trainer.model = model  # ensure trainer uses adapter
        metrics = trainer.evaluate()
        return metrics.get("eval_loss", 0.0), metrics
    return evaluate_fn

# ======= WEIGHTED AVERAGE FUNCTION =======
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {}
    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value * num_examples
    for key in aggregated_metrics:
        aggregated_metrics[key] /= total_examples
    return aggregated_metrics


# ======= MEMORY-EFFICIENT FEDPROX STRATEGY =======
class MemoryEfficientFedProx(FedProx):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        del results, failures
        gc.collect()
        return aggregated_result

# ======= FEDPROX STRATEGY CONFIGURATION =======
strategy = MemoryEfficientFedProx(
    proximal_mu=PROXIMAL_MU,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=3,
    min_available_clients=3,
    on_fit_config_fn=lambda rnd: {"epoch": rnd},
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(model, dataset, tokenizer),
)

# ======= START SERVER =======
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),  # or whatever you need
    strategy=strategy,
)
