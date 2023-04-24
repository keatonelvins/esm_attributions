import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import EsmTokenizer, EsmForSequenceClassification
from transformers.integrations import TensorBoardCallback


# Read data
data = pd.read_excel("MOESM7_ESM.xlsx")

# Define pretrained tokenizer and model
model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=1)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["sequence"])
y = list(data["red_brightness"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    mse = mean_squared_error(y_true=labels, y_pred=pred)
    mae = mean_absolute_error(y_true=labels, y_pred=pred)
    r2 = r2_score(y_true=labels, y_pred=pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=25,
    per_device_eval_batch_size=25,
    num_train_epochs=1,
    seed=0,
    logging_dir='logs',
    logging_steps=10,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), TensorBoardCallback()],
)

# Train pre-trained model
trainer.train()
trainer.save_model("output/final_model")

# Call tensorboard --logdir='./logs' to view training progress
