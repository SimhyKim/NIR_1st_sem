import torch
import numpy as np
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# 1. Configuration

MODEL_CHECKPOINT = "microsoft/codebert-base"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# 2. Load dataset

dataset = load_dataset("bstee615/bigvul")

train_dataset = dataset["train"]
val_dataset   = dataset["validation"]
test_dataset  = dataset["test"]

# 3. Tokenization

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(
        examples["func_before"],
        truncation=True,
        max_length=MAX_LEN
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("vul", "labels")
val_dataset   = val_dataset.rename_column("vul", "labels")
test_dataset  = test_dataset.rename_column("vul", "labels")

cols_to_remove = ["func_before", "project", "commit_id", "func_after"]
train_dataset = train_dataset.remove_columns(
    [c for c in cols_to_remove if c in train_dataset.column_names]
)
val_dataset = val_dataset.remove_columns(
    [c for c in cols_to_remove if c in val_dataset.column_names]
)
test_dataset = test_dataset.remove_columns(
    [c for c in cols_to_remove if c in test_dataset.column_names]
)

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

# 4. Class imbalance handling
labels = train_dataset["labels"]
class_counts = np.bincount(labels)
total = class_counts.sum()

class_weights = torch.tensor(
    total / (2 * class_counts),
    dtype=torch.float
)

print("Class weights [safe, vulnerable]:", class_weights.tolist())

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(model.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# 5. Model

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2,
    use_safetensors=True
)

# 6. Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# 7. Training setup
training_args = TrainingArguments(
    output_dir="./results_bigvul_codebert",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",       
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    report_to=[]
)


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)


# 8. Training

print("Starting training...")
trainer.train()

print("Validation evaluation:")
trainer.evaluate()

print("Test evaluation:")
trainer.evaluate(test_dataset)
