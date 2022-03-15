import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import csv
import sys
from sklearn.metrics import accuracy_score

if len(sys.argv) != 2:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<algorithm>")
    exit()

algorithm = sys.argv[1]
print("WARNING: algorithm argument should be \'xgboost\' for XG-Boost or \'rf\' for Random Forest. Otherwise logistic regression is applied.")

# Read The data
training_set = pd.read_json('processed_data/train_set.json')
test_set = pd.read_json('processed_data/test_set.json')

roberta_train = pd.read_csv("processed_data/roberta_train.csv")[["label"]]
roberta_test = pd.read_csv("processed_data/roberta_test.csv")[["label"]]
roberta_train.rename(columns={"label": "roberta"},inplace=True)
roberta_test.rename(columns={"label": "roberta"},inplace=True)

gltr_train = pd.read_csv("processed_data/gltr_train.csv")
gltr_test = pd.read_csv("processed_data/gltr_test.csv")

keywords_train = pd.read_csv("processed_data/keywords_train.csv")
keywords_test = pd.read_csv("processed_data/keywords_test.csv")

embedding_train = pd.read_csv("processed_data/embedding_train.csv")
embedding_test = pd.read_csv("processed_data/embedding_test.csv")

ngrams_train = pd.read_csv("processed_data/ngrams_train.csv")
ngrams_test = pd.read_csv("processed_data/ngrams_test.csv")

rouge_train = pd.read_csv("processed_data/rouge_train.csv")
rouge_test = pd.read_csv("processed_data/rouge_test.csv")

# Combining
X = pd.concat([roberta_train, gltr_train, keywords_train, embedding_train, ngrams_train, rouge_train], axis = 1)
Y = training_set.label

if algorithm == "xgboost":
    classifier = XGBClassifier(objective='binary:logistic')
elif algorithm == "rf":
    classifier  = RandomForestClassifier()
else:
    classifier = LogisticRegression(max_iter=10000)

clf = classifier.fit(X, Y)
# Compute accuracy on train set
Y_pred = classifier.predict(X)
print(accuracy_score(Y,Y_pred))

# Write predictions on test set to a file
X_test = pd.concat([roberta_test, gltr_test, keywords_test, embedding_test, ngrams_test, rouge_test], axis = 1)

predictions = classifier.predict(X_test)

with open("output/submission"+algorithm+".csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])