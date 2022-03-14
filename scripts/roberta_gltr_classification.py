# Compute GLTR statistics on summaries
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from gltr import LM, BERTLM


# Read The data
training_set = pd.read_json('data/train_set.json')
roberta_train = pd.read_csv("data/roberta_train.csv")[["label"]]
roberta_test = pd.read_csv("data/roberta_test.csv")[["label"]]
roberta_train.rename(columns={"label": "roberta"},inplace=True)
test_set = pd.read_json('data/test_set.json')
roberta_test.rename(columns={"label": "roberta"},inplace=True)
column_names = ["maxk","avgk","avgp","sigmak","sigmap","top10","top100","top1000","sup1000","histp20","histp40","histp60","histp80","histp100","histp_bound0","histp_bound1","histp_bound2","histp_bound3","histp_bound4","histp_bound5","histent20","histent40","histent60","histent80","histent100","histent_bound0","histent_bound1","histent_bound2","histent_bound3","histent_bound4","histent_bound5"]
columns_train = np.zeros((len(training_set),len(column_names)))
columns_test = np.zeros((len(test_set),len(column_names)))

lm = LM()
for i in range(len(training_set.index)):
    gltr_results = lm.check_probabilities(training_set.loc[i]["summary"], topk=10)
    gltr_k = np.array([x[0] for x in gltr_results["real_topk"]])
    gltr_p = np.array([x[1] for x in gltr_results["real_topk"]])
    gltr_ent = np.zeros(len(gltr_results["pred_topk"]))
    for j,x in enumerate(gltr_results["pred_topk"]):
      for elt in x:
        gltr_ent[j] = gltr_ent[j] - elt[1]*np.log(elt[1])
    columns_train[i] = np.concatenate((np.array([np.max(gltr_k),np.mean(gltr_k),np.mean(gltr_p),np.std(gltr_k),np.std(gltr_p),np.sum(gltr_k<10),np.sum(np.logical_and(gltr_k>=10,gltr_k<100)),np.sum(np.logical_and(gltr_k>=100, gltr_k<1000)),np.sum(gltr_k>=1000)]),np.histogram(gltr_p,bins=5)[0],np.histogram(gltr_p,bins=5)[1],np.histogram(gltr_ent,bins=5)[0],np.histogram(gltr_ent,bins=5)[1]))

X = pd.concat([roberta_train,pd.DataFrame(columns_train,columns=column_names)],axis=1)
X.to_csv("x.csv")
print(X.head())

for i in range(len(test_set.index)):
    gltr_results = lm.check_probabilities(training_set.loc[i]["summary"], topk=10)
    gltr_k = np.array([x[0] for x in gltr_results["real_topk"]])
    gltr_p = np.array([x[1] for x in gltr_results["real_topk"]])
    gltr_ent = np.zeros(len(gltr_results["pred_topk"]))
    for j,x in enumerate(gltr_results["pred_topk"]):
      for elt in x:
        gltr_ent[j] = gltr_ent[j] - elt[1]*np.log(elt[1])
    columns_test[i] = np.concatenate((np.array([np.max(gltr_k),np.mean(gltr_k),np.mean(gltr_p),np.std(gltr_k),np.std(gltr_p),np.sum(gltr_k<10),np.sum(np.logical_and(gltr_k>=10,gltr_k<100)),np.sum(np.logical_and(gltr_k>=100, gltr_k<1000)),np.sum(gltr_k>=1000)]),np.histogram(gltr_p,bins=5)[0],np.histogram(gltr_p,bins=5)[1],np.histogram(gltr_ent,bins=5)[0],np.histogram(gltr_ent,bins=5)[1]))
X_test = pd.concat([roberta_test,pd.DataFrame(columns_test,columns=column_names)],axis=1)

Y = training_set.label
# X = pd.read_csv("x.csv",index_col=0) (activate to save time)
X_train, X_val , Y_train, Y_val = train_test_split(X, Y, test_size=0.05, random_state=0)

xgbc = XGBClassifier(objective='binary:logistic', colsample_bytree= 0.7, learning_rate= 0.01, max_depth= 3, n_estimators= 100)

clf = xgbc.fit(X_train,Y_train)

y_pred_val = xgbc.predict(X_val)
y_pred_val = y_pred_val.round(0).astype(int)

print(accuracy_score(Y_val, y_pred_val))
print(accuracy_score(Y_val, np.round(X_val[["roberta"]].to_numpy(),0)))

# Write predictions to a file
predictions = xgbc.predict(X_test)
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])