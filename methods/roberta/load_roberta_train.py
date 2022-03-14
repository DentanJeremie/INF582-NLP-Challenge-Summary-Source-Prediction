from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DistilBertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


torch.cuda.empty_cache()

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
    
    
# ----- 3. Predict -----#
# Load train data
train_data = pd.read_json("raw_data/train_set.json")

X_train = list(train_data["summary"])

docs = list(train_data["document"])

X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)

Y_train = list(train_data['label'])
# Create torch dataset
train_dataset = Dataset(X_train_tokenized)

# Load trained model
model_path = "methods/roberta/output/checkpoint-2500"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define train trainer
train_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = train_trainer.predict(train_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

def compute_metrics(y_pred, true_labels):

    accuracy = accuracy_score(y_true = true_labels, y_pred = y_pred)
    recall = recall_score(y_true = true_labels, y_pred = y_pred)
    precision = precision_score(y_true = true_labels, y_pred = y_pred)
    f1 = f1_score(y_true = true_labels, y_pred = y_pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

print(compute_metrics(y_pred, Y_train))

with open("methods/roberta/analysis/false_positive.txt", "w") as file:
    for i,a in enumerate(X_train):
        if y_pred[i]==1 and Y_train[i] == 0:
            file.write("\n######################################################################################################################\n")
            file.write(str(raw_pred[i]))
            file.write("\n-----------------------------------------------\n")
            file.write(a)
            file.write("\n-----------------------------------------------\n")
            file.write(docs[i])
            file.write("\n\n\n")
    file.close()
    
with open("methods/roberta/analysis/false_negative.txt", "w") as file:
    for i,a in enumerate(X_train):
        if y_pred[i]==0 and Y_train[i] == 1:
            file.write("\n######################################################################################################################\n")
            file.write(str(raw_pred[i]))
            file.write("\n-----------------------------------------------\n")
            file.write(a)
            file.write("\n-----------------------------------------------\n")
            file.write(docs[i])
            file.write("\n\n\n")
    file.close()