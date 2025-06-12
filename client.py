import flwr as fl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict, set_peft_model_state_dict


# ======= USER-CONFIGURABLE PARAMETERS =======
MODEL_NAME = "facebook/opt-350m"                    # Base LLM
DATASET_NAME = "codeparrot/codeparrot-clean-valid"  # Dataset used for training/eval
TOTAL_EXAMPLES = 9000                               # Total dataset size
SPLIT_SIZE = 3000                                   # Split per client (for manual client setups)
EPOCHS = 3                                          # Number of training epochs
LR = 1e-5                                           # Learning rate
TRAIN_BATCH_SIZE = 2                                # Training batch size per device
MAX_LENGTH = 128                                    # Max sequence length for tokenization
LORA_R = 16                                         # LoRA rank
LORA_ALPHA = 32                                     # LoRA alpha
LORA_DROPOUT = 0.05                                 # LoRA dropout


# ======= LOAD MODEL AND TOKENIZER =======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# ======= APPLY LoRA CONFIGURATION =======
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],  # key attention layers
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ======= LOAD AND TOKENIZE DATASET =======
dataset = load_dataset(DATASET_NAME)
dataset = dataset["train"].shuffle(seed=42).select(range(TOTAL_EXAMPLES))
client_id = int(os.getenv("CLIENT_ID", 0))
client_data = dataset.select(range(client_id * SPLIT_SIZE, (client_id + 1) * SPLIT_SIZE))

def tokenize(batch):
    return tokenizer(batch["content"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

tokenized_data = client_data.map(tokenize, batched=True)
tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
tokenized_data = tokenized_data.map(lambda x: {"labels": x["input_ids"]})
split_dataset = tokenized_data.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


train_loader = DataLoader(tokenized_data, batch_size=2)

# ======= TRAINING ARGUMENTS =======
args = TrainingArguments(
    output_dir="./opt_checkpoints",
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    learning_rate=LR,
    report_to="none",
    save_strategy="no",
    disable_tqdm=True,
)

# ======= TRAINER SETUP =======
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# ======= FLOWER CLIENT CLASS =======
class FlowerClient(fl.client.NumPyClient):
    # Get initial parameters from the model
    def get_parameters(self, config):
        adapter_state = get_peft_model_state_dict(model, model.state_dict())
        return [p.detach().cpu().numpy() for p in adapter_state.values()]

    # Set parameters to the model
    def set_parameters(self, parameters):
        keys = list(get_peft_model_state_dict(model).keys())
        adapter_state = {k: torch.tensor(p) for k, p in zip(keys, parameters)}
        set_peft_model_state_dict(model, adapter_state)

    # Fit the model with given parameters and return metrics
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer.train()
        metrics = trainer.evaluate()
        result = self.get_parameters(config), len(train_dataset), {"train_loss": metrics.get("eval_loss", 0.0)}

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return result

    # Evaluate the model with given parameters and return loss and metrics
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss", 0.0)
        
        return loss, len(tokenized_data), metrics


# ======= START FLOWER CLIENT =======
server_ip = os.getenv("SERVER_IP", "server:8080")
fl.client.start_numpy_client(server_address=server_ip, client=FlowerClient())