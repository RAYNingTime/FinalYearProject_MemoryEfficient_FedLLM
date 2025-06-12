import flwr as fl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

# ======= USER-CONFIGURABLE PARAMETERS =======
MODEL_NAME = "facebook/opt-350m"                   # Base LLM
DATASET_NAME = "codeparrot/codeparrot-clean-valid" # Dataset used for training/eval
TOTAL_EXAMPLES = 3000                              # Total dataset size
SPLIT_SIZE = 1000                                  # Split per client (for manual client setups)
EPOCHS = 3                                          # Number of training epochs
LR = 1e-5                                           # Learning rate
TRAIN_BATCH_SIZE = 2                                # Training batch size per device
MAX_LENGTH = 128                                    # Max sequence length for tokenization


# ======= LOAD MODEL AND TOKENIZER =======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ======= LOAD AND TOKENIZE DATASET =======
dataset = load_dataset("codeparrot/codeparrot-clean-valid")
dataset = dataset["train"].shuffle(seed=42).select(range(TOTAL_EXAMPLES))  # Keep it small for test
client_id = int(os.getenv("CLIENT_ID", 0))  # 0 or 1
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
    # This method is called to get the model parameters
    def get_parameters(self, config):
        with torch.no_grad():
            parameters = []
            for name, param in model.state_dict().items():
                param = param.detach().to("cpu").half().contiguous().numpy()
                parameters.append(param)
                del param
            torch.cuda.empty_cache()
            return parameters

    # Set parameters to the model
    def set_parameters(self, parameters):
        state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict, strict=True)

    # Fit the model with given parameters and return metrics
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer.train()
        metrics = trainer.evaluate()
        result = self.get_parameters(config), len(train_dataset), {"train_loss": metrics.get("eval_loss", 0.0)}

        # Clean up
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