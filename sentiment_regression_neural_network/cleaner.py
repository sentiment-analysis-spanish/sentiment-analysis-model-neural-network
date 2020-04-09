#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

from keras.datasets import imdb
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
import pandas as pd
import re
import numpy as np
import pickle

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^\w\s]')
STOPWORDS = set(stopwords.words('spanish'))


class Cleaner:
    def __init__(self):
        self.tokenizer = None
    


    def clean_text(self, text):
        text = text.lower() # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
        #    text = re.sub(r'\W+', '', text)
        STOPWORDS.add("\n")
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
        text = text.strip()
        return text
        

    def clean_news(self, df):
        print("cleaning the text data")
        df = df.reset_index(drop=True)
        df.dropna(subset=['rate'], inplace=True)
        df['sentiment'] = df.apply (lambda row: self.categorize(row), axis=1)
        df = self.select_rows(df)
        df['content'] = df['content'].apply(self.clean_text)
        df['content'] = df['content'].str.replace('\d+', '')
        return df

    def select_rows(self, df):
        df_negative = df.loc[df['sentiment'] < 0.5]
        df_positive = df.loc[df['sentiment'] > 0.5]

        count = len(df_negative.index)
        print(count)
        df_positive_selected = df_positive.sample(count)
        frames = [df_negative, df_positive_selected]
        df_concat = pd.concat(frames)
        print(df_concat)
        return df_concat

    def categorize (self, row):
        if row['rate'] > 0.5 :
            return 1
        if row['rate'] < 0.5 :
            return 0
        
    def load_tokenizer(self, sentences):
        print("loading toikenizer")
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(sentences)

        # saving tokenizer
        with open('../data/neural_network_config/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def create_tokenizer_and_clean(self):
        filename = "../data/json_bundle_reviews/large-bundle.json"
        output = "../data/json_bundle_reviews/large-bundle-clean.json"
        
        df=  pd.read_json(filename)
        print(df)

        df = self.clean_news(df)
        df.to_json(output,force_ascii=False)
        
        sentences = df['content'].values
        self.load_tokenizer(sentences)


if __name__== "__main__":
    Cleaner().create_tokenizer_and_clean()