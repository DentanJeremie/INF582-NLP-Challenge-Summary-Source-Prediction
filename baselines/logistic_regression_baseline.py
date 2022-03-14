import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv


# Read The data
training_set = pd.read_json('./train_set.json')
test_set = pd.read_json('./test_set.json')
documents_set = pd.read_json('./documents.json')
print(documents_set.head())

print(training_set.loc[0].summary)
print(training_set.loc[0].document)

# Use logistic regression to predict the class
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(training_set['summary'])
clf = LogisticRegression()
clf.fit(X, training_set['label'])
X_test = vectorizer.transform(test_set['summary'])
predictions = clf.predict(X_test)


# Write predictions to a file
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])