import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from tqdm.auto import tqdm


import xgboost as xgb


# Read The data
training_set = pd.read_json('../raw_data/train_set.json')
test_set = pd.read_json('../raw_data/test_set.json')

doc2v_train = pd.read_json('../processed_data/d2v_training_20.json')
doc2v_test = pd.read_json('../processed_data/d2v_training_5.json')



params = {}
params['learning_rate']=0.05
params['boosting_type']='gbdt' 
params['objective']='binary'
params['metric']='binary_logloss'
params['max_depth']= 20
params['num_leaves'] = 100


"""# Use logistic regression to predict the class
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(training_set['summary'])
num_round = 100

Y = training_set.label

X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=0)

train_data = lgb.Dataset(X_train, label = Y_train)"""

tqdm.pandas(desc = 'Processing difference between rows')
doc2v_train['difference'] = doc2v_train.progress_apply(lambda x: np.array(x['summary_embedding']) - np.array(x['document_embedding']), axis=1)
X = doc2v_train.difference.to_numpy()
X = np.array([np.array(a) for a in X])
Y = training_set.label

print(X.shape)

X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=0)

data_dmatrix = xgb.DMatrix(data = X_train, label = y)

train_data = lgb.Dataset(X_train, label = Y_train)


clf = lgb.train(params, train_data, 100, verbose_eval= 100)

y_pred = clf.predict(X_test)
y_pred = y_pred.round(0)
y_pred = y_pred.astype(int)

print(roc_auc_score(y_pred, Y_test))

#X_test = vectorizer.transform(test_set['summary'])
doc2v_test['difference'] = doc2v_test.progress_apply(lambda x: np.array(x['summary_embedding']) - np.array(x['document_embedding']), axis=1)
X_test = doc2v_test.difference
predictions = clf.predict(X_test)
predictions = predictions.round(0)
predictions = predictions.astype(int)

# Write predictions to a file
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])