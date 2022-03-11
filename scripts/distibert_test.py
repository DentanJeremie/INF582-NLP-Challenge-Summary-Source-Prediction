from transformers import BartTokenizer, BartForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

dataset = pd.read_json('../raw_data/train_set.json')
print(dataset)

dataset = dataset[['summary', 'label']]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.apply(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return roc_auc_score(y_score=predictions, y_true=labels)


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

model = BartForSequenceClassification.from_pretrained("facebook/bart-large")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits


