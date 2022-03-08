from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel
import torch
import pandas as pd
from tqdm.auto import tqdm


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

training_set = pd.read_json('../preprocessed_data/train_set.json')
test_set = pd.read_json('../preprocessed_data/test_set.json')

def embed(row):
    input_sum = tokenizer(row.summary, return_tensors="pt")
    embed_sum = model(**input_sum).pooler_output.ravel().detach().numpy()
    return embed_sum

RECOM_TRAINING = True

if RECOM_TRAINING:
    tqdm.pandas(desc = 'Processing training rows')

    res = training_set.progress_apply(lambda row : embed(row), axis = 1)

    training_set.insert(len(training_set.columns), "sum_rob_emb", res)

    training_set.to_json("../processed_data/roberta_training.json")

RECOM_TESTING = True

if RECOM_TESTING:
    tqdm.pandas(desc = 'Processing testing rows')

    res = test_set.progress_apply(lambda row : embed(row), axis = 1)

    test_set.insert(len(test_set.columns), "sum_rob_emb", res)

    test_set.to_json("../processed_data/roberta_testing.json")