# Compute GLTR statistics on summaries
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from tqdm import tqdm
from gltr import LM, BERTLM


# Read The data
training_set = pd.read_json('processed_data/train_set.json')
test_set = pd.read_json('processed_data/test_set.json')

# Column names
column_names = ["maxk","avgk","avgp","sigmak","sigmap","top10","top100","top1000","sup1000","histp20","histp40","histp60","histp80","histp100","histp_bound0","histp_bound5","histent20","histent40","histent60","histent80","histent100","histent_bound0","histent_bound5"]
columns_train = np.zeros((len(training_set),len(column_names)))
columns_test = np.zeros((len(test_set),len(column_names)))

# Adding features from GLTK
lm = LM()

for i in tqdm(range(len(training_set.index))):
    gltr_results = lm.check_probabilities(training_set.loc[i]["summary"], topk=10)
    gltr_k = np.array([x[0] for x in gltr_results["real_topk"]])
    gltr_p = np.array([x[1] for x in gltr_results["real_topk"]])
    gltr_ent = np.zeros(len(gltr_results["pred_topk"]))
    for j,x in enumerate(gltr_results["pred_topk"]):
        for elt in x:
            gltr_ent[j] = gltr_ent[j] - elt[1]*np.log(elt[1])
    columns_train[i] = np.concatenate((np.array([np.max(gltr_k),np.mean(gltr_k),np.mean(gltr_p),np.std(gltr_k),np.std(gltr_p),np.sum(gltr_k<10),np.sum(np.logical_and(gltr_k>=10,gltr_k<100)),np.sum(np.logical_and(gltr_k>=100, gltr_k<1000)),np.sum(gltr_k>=1000)]),np.histogram(gltr_p,bins=5)[0],np.histogram(gltr_p,bins=5)[1][0],np.histogram(gltr_p,bins=5)[1][1],np.histogram(gltr_ent,bins=5)[0],np.histogram(gltr_ent,bins=5)[1][0],np.histogram(gltr_ent,bins=5)[1][1]))

X_train = pd.DataFrame(columns_train,columns=column_names)
X_train.to_csv("processed_data/gltr_train.csv",index=False)
print("Train set :\n",X_train.head())


# The same for the test set
for i in tqdm(range(len(test_set.index))):
    gltr_results = lm.check_probabilities(training_set.loc[i]["summary"], topk=10)
    gltr_k = np.array([x[0] for x in gltr_results["real_topk"]])
    gltr_p = np.array([x[1] for x in gltr_results["real_topk"]])
    gltr_ent = np.zeros(len(gltr_results["pred_topk"]))
    for j,x in enumerate(gltr_results["pred_topk"]):
      for elt in x:
        gltr_ent[j] = gltr_ent[j] - elt[1]*np.log(elt[1])
    columns_test[i] = np.concatenate((np.array([np.max(gltr_k),np.mean(gltr_k),np.mean(gltr_p),np.std(gltr_k),np.std(gltr_p),np.sum(gltr_k<10),np.sum(np.logical_and(gltr_k>=10,gltr_k<100)),np.sum(np.logical_and(gltr_k>=100, gltr_k<1000)),np.sum(gltr_k>=1000)]),np.histogram(gltr_p,bins=5)[0],np.histogram(gltr_p,bins=5)[1][0],np.histogram(gltr_p,bins=5)[1][1],np.histogram(gltr_ent,bins=5)[0],np.histogram(gltr_ent,bins=5)[1][0],np.histogram(gltr_ent,bins=5)[1][1]))
X_test = pd.DataFrame(columns_test,columns=column_names)
X_test.to_csv("processed_data/gltr_test.csv",index=False)
print("Test set :\n",X_test.head())


