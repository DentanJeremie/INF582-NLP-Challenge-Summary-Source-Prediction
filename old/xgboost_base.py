import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas(desc = 'Processing difference between rows')
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Read The data
training_set = pd.read_json('../raw_data/train_set.json')
test_set = pd.read_json('../raw_data/test_set.json')

doc2v_train = pd.read_json('../processed_data/d2v_training_20.json')
doc2v_test = pd.read_json('../processed_data/d2v_testing_20.json')



params = {}
params['learning_rate']=0.05
params['boosting_type']='gbdt' 
params['objective']='binary'
params['metric']='binary_logloss'
params['max_depth']= 20
params['num_leaves'] = 100

'''# Use logistic regression to predict the class
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(training_set['summary'])
num_round = 100

Y = training_set.label

X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)'''


doc2v_train['difference'] = doc2v_train.progress_apply(lambda x: np.array(x['summary_embedding']) - np.array(x['document_embedding']), axis=1)
X = doc2v_train.difference.to_numpy()
X = np.array([np.array(a) for a in X])
Y = training_set.label


Standardscaler = StandardScaler()
Standardscaler.fit_transform(X)


pca = PCA(n_components=10)
pca.fit_transform(X)

print("> PCA results",pca.explained_variance_ratio_)


X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)


#data_dmatrix = xgb.DMatrix(data = X_train, label = y)

kwargs = {
            'max_depth': 20,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'verbosity': 1 # 2 for progress bar
        }

clf = xgb.XGBClassifier(**kwargs)

clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)

print(roc_auc_score(y_pred, Y_test))

X_test = vectorizer.transform(test_set['summary'])
'''doc2v_test['difference'] = doc2v_test.progress_apply(lambda x: np.array(x['summary_embedding']) - np.array(x['document_embedding']), axis=1)
X_test = doc2v_test.difference
'''
predictions = clf.predict(X_test)

# Write predictions to a file
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])