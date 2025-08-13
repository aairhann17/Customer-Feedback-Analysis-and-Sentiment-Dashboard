import pandas as pd
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)              # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)             # Remove punctuation and numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['cleaned_feedback'] = df['feedback_text'].apply(clean_text)
    df.dropna(subset=['cleaned_feedback'], inplace=True)
    return df
