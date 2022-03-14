# See for more details : https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data
training_set = pd.read_json('processed_data/train_set.json')
test_set = pd.read_json('processed_data/test_set.json')

# Config
n_gram_range = (1, 1)
stop_words = "english"
top_n = 10

# Output
column_names = ["docKw_in_sum", "docKw_in_sumKw", "sumKw_in_doc", "sumKw_in_docKw"]
columns_train = np.zeros((len(training_set),len(column_names)))
columns_test = np.zeros((len(test_set),len(column_names)))

# BERT MODEL
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

for step, (to_process, columns) in enumerate([(training_set,columns_train), (test_set,columns_test)]):
    for i in tqdm(range(len(to_process.index))):
        # Candidates for keywords
        text_doc = to_process.loc[i]["document"]
        count_doc = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text_doc])
        candidates_doc = count_doc.get_feature_names_out()

        text_sum = to_process.loc[i]["summary"]
        count_sum = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text_sum])
        candidates_sum = count_sum.get_feature_names_out()

        # Embedding with BERT
        embedding_doc = model.encode([text_doc])
        candidate_embeddings_doc = model.encode(candidates_doc)

        embedding_sum = model.encode([text_sum])
        candidate_embeddings_sum = model.encode(candidates_sum)

        # Cosine similarity
        distances_doc = cosine_similarity(embedding_doc, candidate_embeddings_doc)[0]
        keywords_doc = [candidates_doc[index] for index in distances_doc.argsort()[-top_n:]]

        distances_sum = cosine_similarity(embedding_sum, candidate_embeddings_sum)[0]
        keywords_sum = [candidates_sum[index] for index in distances_sum.argsort()[-top_n:]]
        
        # Score
        docKw_in_sum = 0
        docKw_in_sumKw = 0
        sumKw_in_doc = 0
        sumKw_in_docKw = 0
        for word in keywords_doc:
            if word in text_sum:
                docKw_in_sum += 1
            if word in keywords_sum:
                docKw_in_sumKw += 1
        for word in keywords_sum:
            if word in text_doc:
                sumKw_in_doc += 1
            if word in keywords_doc :
                sumKw_in_docKw += 1

        
        columns[i] = np.array([docKw_in_sum, docKw_in_sumKw, sumKw_in_doc, sumKw_in_docKw])
    
    if step==0:
        pd.DataFrame(columns,columns=column_names).to_csv("processed_data/keywords_train.csv",index=False)
    else:
        pd.DataFrame(columns,columns=column_names).to_csv("processed_data/keywords_test.csv",index=False)