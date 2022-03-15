from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DistilBertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import csv

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
# Load test data
test_data = pd.read_json("raw_data/test_set.json")
X_test = list(test_data["summary"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = "methods/roberta/output/checkpoint-2000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

with open("methods/roberta/distibert/submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(y_pred):
        csv_out.writerow([i, row])