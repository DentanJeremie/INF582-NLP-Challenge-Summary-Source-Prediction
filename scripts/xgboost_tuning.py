from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


xgbc = XGBClassifier(objective='binary:logistic')

# Read The data
training_set = pd.read_json('processed_data/train_set.json')

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

X = pd.concat([roberta_train, gltr_train, keywords_train, embedding_train, ngrams_train, rouge_train], axis = 1)
Y = training_set.label

# Define the grid
params = { 'max_depth': [2,3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [50, 100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}

clf = GridSearchCV(estimator=xgbc, 
                   param_grid=params,
                   scoring='accuracy', 
                   verbose=1)

# Get the best parameters
clf.fit(X,Y)
print("Best parameters:", clf.best_params_)
print("Best score: ", clf.best_score_)