import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np

# Read The data
training_set = pd.read_json('../raw_data/train_set.json')
test_set = pd.read_json('../raw_data/test_set.json')
document_set = pd.read_json('../raw_data/documents.json')


summs = training_set['summary'].to_numpy()

with open('recap.txt', "w") as f:
    for a in summs:
        f.write(a)
        f.write("\n\n\n")
    f.close

quit()
labels_train = training_set['label'].to_numpy()
labels_test = training_set["label"].to_numpy()

print('> Label repartition in the dataset')
fq_1_train = np.sum(labels_train) / len(labels_train)
fq_0_train = 1 - fq_1_train
print('     > For the training set:', fq_1_train,  fq_0_train)

fq_1_test = np.sum(labels_test) / len(labels_test)
fq_0_test = 1 - fq_1_test
print('     > For the testing set:', fq_1_test,  fq_0_test)

#print(training_set['summary'].iloc[0])

#print(training_set['summary'].iloc[2])

for _, row in training_set.iterrows():
    if row['label'] == 0:
        print("doc", row['document'])
        print("sum", row['summary'])
        print("\n---------------------------")
        