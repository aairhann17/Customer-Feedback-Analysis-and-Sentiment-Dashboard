from textblob import TextBlob
from gensim import corpora, models

def analyze_sentiment(text):
    """
    Uses TextBlob to calculate sentiment polarity.
    Returns a float between -1 (negative) and 1 (positive).
    """
    return TextBlob(text).sentiment.polarity

def topic_modeling(texts, num_topics=5):
    """
    Performs topic modeling using Latent Dirichlet Allocation (LDA).
    Steps:
    1. Tokenize text into words.
    2. Create a dictionary mapping words to IDs.
    3. Create a bag-of-words corpus from the dictionary.
    4. Train an LDA model to find 'num_topics' topics.
    Returns the trained LDA model and a list of topics.
    """
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics()
    return lda_model, topics
