import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
import re
import nltk
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
nltk.download('punkt')


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, epochs=5)) for doc in sents])
    return targets, regressors


cores = multiprocessing.cpu_count()

training_set = pd.read_json('../processed_data/train_set.json')
test_set = pd.read_json('../processed_data/test_set.json')

train_tagged_summary = training_set.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['summary']), tags=[r.label]), axis=1)
train_tagged_document = training_set.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['document']), tags=[r.label]), axis=1)

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)

print("\n> Building model vocab")

model_dbow.build_vocab([x for x in tqdm(train_tagged_summary.values)] + [x for x in tqdm(train_tagged_document.values)])

print("\n> Training bag of words model")
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged_summary.values, desc = 'epoch {}'.format(epoch))]), total_examples=len(train_tagged_summary.values), epochs=30)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)

print("\n> Building model vocab")

model_dmm.build_vocab([x for x in tqdm(train_tagged_summary.values)] + [x for x in tqdm(train_tagged_document.values)])

print("\n> Training Distributed memory model")
for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged_summary.values, desc = 'epoch {}'.format(epoch))]), total_examples=len(train_tagged_summary.values), epochs=30)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

test_tagged_summary = test_set.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['summary']), tags=[1]), axis=1)
test_tagged_document = test_set.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['document']), tags=[1]), axis=1)

print('\n> Generating final vectors')
y_train, X_train = get_vectors(new_model, train_tagged_summary)
y_test, X_test = get_vectors(new_model, test_tagged_summary)

y_train, X_train_document = get_vectors(new_model, train_tagged_document)
y_test, X_test_document = get_vectors(new_model, test_tagged_document)

print('> Saving result')
training_result = pd.DataFrame()
training_result.insert(0, "summary_embedding", X_train)
training_result.insert(0, "document_embedding", X_train_document)
training_result.insert(2, "labels", training_set['label'])

testing_result = pd.DataFrame()
testing_result.insert(0, "summary_embedding", X_test)
testing_result.insert(0, "document_embedding", X_test_document)

training_result.to_json("../processed_data/d2v_training_5.json")
testing_result.to_json("../processed_data/d2v_testing_5.json")