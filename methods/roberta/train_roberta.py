from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DistilBertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import EarlyStoppingCallback
import csv

torch.cuda.empty_cache()


dataset = pd.read_json('../../raw_data/train_set.json')

dataset = dataset[['summary', 'label']]

model_type = 'distilroberta-base'
#model_type ='distilbert-base-cased'

tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast = True)

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

model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels = 2)



# ----- 1. Preprocess data -----#
# Preprocess data
X = list(dataset["summary"])
y = list(dataset["label"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir = True,
    evaluation_strategy="steps",
    eval_steps=250,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    seed=0,
    load_best_model_at_end=True,
    save_steps= 500, 
    gradient_accumulation_steps=1
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

# Train pre-trained model
trainer.train()


# ----- 3. Predict -----#
# Load test data
test_data = pd.read_json("../../raw_data/test_set.json")
X_test = list(test_data["summary"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

with open("distibert/submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(y_pred):
        csv_out.writerow([i, row])