import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
import math
import operator
from functools import reduce
from rouge import Rouge
import sys
sys.setrecursionlimit(2000)

nltk.download('punkt')

# Data
training_set = pd.read_json('processed_data/train_set.json')
test_set = pd.read_json('processed_data/test_set.json')

# Output
#column_names = ["rouge_1_r", "rouge_1_p", "rouge_1_f", "rouge_2_r", "rouge_2_p", "rouge_2_f", "rouge_3_r", "rouge_3_p", "rouge_3_f", "rouge_4_r", "rouge_4_p", "rouge_4_f", "rouge_5_r", "rouge_5_p", "rouge_5_f", "rouge_l_r", "rouge_l_p", "rouge_l_f"]
column_names = ["rouge_1_r", "rouge_1_p", "rouge_1_f", "rouge_2_r", "rouge_2_p", "rouge_2_f", "rouge_l_r", "rouge_l_p", "rouge_l_f"]
columns_train = np.zeros((len(training_set),len(column_names)))
columns_test = np.zeros((len(test_set),len(column_names)))

# Word tokenizer - alphanumeric only
tokenizer = RegexpTokenizer(r'\w+')

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

rouge = Rouge(metrics=Rouge.AVAILABLE_METRICS)

for step, (to_process, columns) in enumerate([(training_set,columns_train), (test_set,columns_test)]):
    for i in tqdm(range(len(to_process.index))):
        # Data
        text_doc = to_process.loc[i]["document"]
        text_sum = to_process.loc[i]["summary"]
        
        dict_scores = rouge.get_scores(text_sum, text_doc)[0]
                
        rouge_1_r = dict_scores["rouge-1"]['r']
        rouge_1_p = dict_scores["rouge-1"]['p'] 
        rouge_1_f = dict_scores["rouge-1"]['f'] 
        rouge_2_r = dict_scores["rouge-2"]['r'] 
        rouge_2_p = dict_scores["rouge-2"]['p'] 
        rouge_2_f = dict_scores["rouge-2"]['f'] 
        rouge_3_r = dict_scores["rouge-3"]['r'] 
        rouge_3_p = dict_scores["rouge-3"]['p'] 
        rouge_3_f = dict_scores["rouge-3"]['f'] 
        rouge_4_r = dict_scores["rouge-4"]['r'] 
        rouge_4_p = dict_scores["rouge-4"]['p'] 
        rouge_4_f = dict_scores["rouge-4"]['f'] 
        rouge_5_r = dict_scores["rouge-5"]['r'] 
        rouge_5_p = dict_scores["rouge-5"]['p'] 
        rouge_5_f = dict_scores["rouge-5"]['f'] 
        rouge_l_r = dict_scores["rouge-l"]['r'] 
        rouge_l_p = dict_scores["rouge-l"]['p'] 
        rouge_l_f = dict_scores["rouge-l"]['f']
    
        #columns[i] = np.array([rouge_1_r, rouge_1_p, rouge_1_f, rouge_2_r, rouge_2_p, rouge_2_f, rouge_3_r, rouge_3_p, rouge_3_f, rouge_4_r, rouge_4_p, rouge_4_f, rouge_5_r, rouge_5_p, rouge_5_f, rouge_l_r, rouge_l_p, rouge_l_f])
        columns[i] = np.array([rouge_1_r, rouge_1_p, rouge_1_f, rouge_2_r, rouge_2_p, rouge_2_f, rouge_l_r, rouge_l_p, rouge_l_f])

    if step==0:
        pd.DataFrame(columns,columns=column_names).to_csv("processed_data/rouge_train.csv",index=False)
    else:
        pd.DataFrame(columns,columns=column_names).to_csv("processed_data/rouge_test.csv",index=False)