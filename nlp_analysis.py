from textblob import TextBlob
from gensim import corpora, models

def analyze_sentiment(text):
    # Returns polarity (-1 to 1)
    return TextBlob(text).sentiment.polarity

def topic_modeling(texts, num_topics=5):
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics()
    return lda_model, topics
