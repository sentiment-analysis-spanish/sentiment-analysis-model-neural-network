
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re

nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^\w\s]')
STOPWORDS = set(stopwords.words('spanish'))

def clean_text(text):
  text = text.lower() # lowercase text
  text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
  text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#    text = re.sub(r'\W+', '', text)
  text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
  return text

def categorize (row):
  if row['rate'] > 0.5 :
      return 1
  if row['rate'] < 0.5 :
      return 0

def clean_news(df):
  print("cleaning the text data")
  df = df.reset_index(drop=True)
  df.dropna(subset=['rate'], inplace=True)
  df['sentiment'] = df.apply (lambda row: categorize(row), axis=1)
  df = select_rows(df)
  df['content'] = df['content'].apply(clean_text)
  df['content'] = df['content'].str.replace('\d+', '')
  return df

def select_rows(df):
  df_negative = df.loc[df['sentiment'] < 0.5]
  df_positive = df.loc[df['sentiment'] > 0.5]

  count = len(df_negative.index)
  print(count)
  df_positive_selected = df_positive.sample(count)
  frames = [df_negative, df_positive_selected]
  df_concat = pd.concat(frames)
  print(df_concat)
  return df_concat



def main():
  filename = "../data/json_bundle_reviews/large-bundle.json"
  output = "../data/json_bundle_reviews/large-bundle-clean.json"
  df = pd.read_json(filename)
  df = clean_news(df)
  df.to_json(output)


if __name__ == "__main__":
    main()

#tar -cJf reviews-174k-bundle.tar.xz large-bundle.json