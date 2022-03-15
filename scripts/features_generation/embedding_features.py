import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Data
training_set = pd.read_json('processed_data/train_set.json')
test_set = pd.read_json('processed_data/test_set.json')

# Output
column_names = ["bert-emb-diff","roberta-emb-diff"]
columns_train = np.zeros((len(training_set),len(column_names)))
columns_test = np.zeros((len(test_set),len(column_names)))

# BERT MODEL
model_bert = SentenceTransformer('distilbert-base-nli-mean-tokens')
model_roberta = SentenceTransformer('roberta-base-nli-mean-tokens')

for step, (to_process, columns) in enumerate([(training_set,columns_train), (test_set,columns_test)]):
    for i in tqdm(range(len(to_process.index))):
        # Candidates for keywords
        text_doc = to_process.loc[i]["document"]
        text_sum = to_process.loc[i]["summary"]

        # Embedding with BERT
        embedding_bert_doc = model_bert.encode([text_doc])
        embedding_bert_sum = model_bert.encode([text_sum])

        # Embedding with ROBERTA
        embedding_roberta_doc = model_roberta.encode([text_doc])
        embedding_roberta_sum = model_roberta.encode([text_sum])

        # Cosine similarity
        bert_emb_diff = cosine_similarity(embedding_bert_doc, embedding_bert_sum)[0]
        roberta_emb_diff = cosine_similarity(embedding_roberta_doc, embedding_roberta_sum)[0]
        
        # Score
        score = np.concatenate((bert_emb_diff, roberta_emb_diff))
        columns[i] = score
    
    if step==0:
        pd.DataFrame(columns,columns=column_names).to_csv("processed_data/embedding_train.csv",index=False)
    else:
        pd.DataFrame(columns,columns=column_names).to_csv("processed_data/embedding_test.csv",index=False)