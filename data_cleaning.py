import pandas as pd
import re
from nltk.corpus import stopwords

# Load a set of common English stopwords to remove from text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans and preprocesses raw feedback text.
    Steps:
    1. Convert text to lowercase.
    2. Remove URLs.
    3. Remove punctuation and numbers.
    4. Tokenize text into words.
    5. Remove stopwords.
    6. Join tokens back into a single string.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)              # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)             # Remove punctuation and numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_clean_data(filepath):
    """
    Loads feedback data from CSV, cleans it, and removes empty rows.
    Expects a column named 'feedback_text' in the dataset.
    """
    df = pd.read_csv(filepath)
    df['cleaned_feedback'] = df['feedback_text'].apply(clean_text)
    df.dropna(subset=['cleaned_feedback'], inplace=True)  # Remove rows with missing cleaned text
    return df
