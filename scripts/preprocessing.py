import re
import numpy as np
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd
from tqdm.auto import tqdm

SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

def clean(text, stem_words=True):    
    def pad_str(s):
        return ' '+s+' '
    
    if pd.isnull(text):
        return ''

#    stops = set(stopwords.words("english"))
    # Clean the text, with the option to stem words.
    
    # Empty question
    
    if type(text) != str or text=='':
        return ''

    # Clean the text
    text = re.sub("\u2019", "'", text)
    text = re.sub("\u2018", "'", text)
    text = re.sub("\u00a0", " ", text)
    # text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    # text = re.sub("\'ve", " have ", text)
    # text = re.sub("can't", "can not", text)
    # text = re.sub("n't", " not ", text)
    # text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    # text = re.sub("\'re", " are ", text)
    # text = re.sub("\'d", " would ", text)
    # text = re.sub("\'ll", " will ", text)
    # text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    # text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    # text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    # text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    # text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"
    text = re.sub('[0-9]+\.[0-9]+', pad_str(str(np.random.randint(100))), text)
    
    # adding padding to numbers (separate them from text) and special characters
    def pad_pattern(pattern):
        matched_string = pattern.group(0)
        return pad_str(matched_string)
    text = re.sub('[0-9]+', pad_pattern, text)
    text = re.sub('[\!\$\%\&\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text)
        
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) # replace non-ascii word with special word
    
    # indian dollar
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    # text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    # text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    # text = re.sub(r" india ", " India ", text)
    # text = re.sub(r" switzerland ", " Switzerland ", text)
    # text = re.sub(r" china ", " China ", text)
    # text = re.sub(r" chinese ", " Chinese ", text) 
    # text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    # text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    # text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    # text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    # text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    # text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    # text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    # text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    # text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    # text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    # text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    # text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    # text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    # text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    # text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    # text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    # text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    # text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    # text = re.sub(r" III ", " 3 ", text)
    #  text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    # text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    # text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    
    words = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in words]
       # Return a list of words
    return text




# Loading
training_set = pd.read_json('raw_data/train_set.json')
test_set = pd.read_json('raw_data/test_set.json')

    
    
tqdm.pandas(desc = 'Preprocessing doc train')
training_set['document'] = training_set['document'].progress_apply(clean)
tqdm.pandas(desc = 'Preprocessing sum train')
training_set['summary'] = training_set['summary'].progress_apply(clean)
tqdm.pandas(desc = 'Preprocessing doc test ')
test_set['document'] = test_set['document'].progress_apply(clean)
tqdm.pandas(desc = 'Preprocessing sum test ')
test_set['summary'] = test_set['summary'].progress_apply(clean)


# Saving
training_set.to_json('processed_data/train_set.json')
test_set.to_json('processed_data/test_set.json')


