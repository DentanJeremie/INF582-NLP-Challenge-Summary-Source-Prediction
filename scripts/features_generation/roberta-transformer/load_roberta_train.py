from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DistilBertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.special import softmax


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
# ----- A. On train set -----#
# Load train data
train_data = pd.read_json("raw_data/train_set.json")

X_train = list(train_data["summary"])

docs = list(train_data["document"])

X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)

Y_train = list(train_data['label'])
# Create torch dataset
train_dataset = Dataset(X_train_tokenized)

# Load trained model
model_path = "methods/roberta/output/checkpoint-4000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define train trainer
train_trainer = Trainer(model)

# Make prediction on train set
raw_pred, _, _ = train_trainer.predict(train_dataset)

# Preprocess raw predictions
y_pred = softmax(raw_pred, axis=1)

with open("processed_data/roberta_train.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(y_pred):
        csv_out.writerow([i, row[1]])

# ----- A. On test set -----#
# Load test data
test_data = pd.read_json("processed_data/test_set.json")
X_test = list(test_data["summary"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = "methods/roberta/output/checkpoint-4000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess probability predictions
y_pred = softmax(raw_pred, axis=1)

with open("processed_data/roberta_test.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(y_pred):
        csv_out.writerow([i, row[1]])

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