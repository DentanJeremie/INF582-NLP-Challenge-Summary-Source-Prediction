from os import truncate
from transformers import TrainingArguments, Trainer, AutoTokenizer, BartForConditionalGeneration
from transformers import pipeline
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(desc = 'Processing summarisation')

tok = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6", use_fast = True)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer=tok)

dataset = pd.read_json('../raw_data/documents.json')

def compute_summary(x):
    return summarizer(str(x[0]), max_length=512, min_length=30, do_sample=False, truncation = True, padding = True)[0]['summary_text']

dataset = dataset.progress_apply(lambda row : compute_summary(row), axis = 1)    

dataset.to_json('../processed_data/documents.json')
