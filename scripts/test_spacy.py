import pandas as pd
import csv
import numpy as np

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

import lightgbm as lgb

from pprint import pprint

nlp = en_core_web_sm.load()

# Read The data
training_set = pd.read_json('../raw_data/train_set.json')
test_set = pd.read_json('../raw_data/test_set.json')
document_set = pd.read_json('../raw_data/documents.json')


doc = nlp(training_set['document'][0])

pprint([(X.text, X.label_) for X in doc.ents])

doc = nlp(training_set['summary'][0])
pprint([(X.text, X.label_) for X in doc.ents])