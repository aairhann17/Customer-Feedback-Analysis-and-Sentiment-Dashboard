from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_sentiment_classifier(df):
    df['sentiment_label'] = df['cleaned_feedback'].apply(lambda x: 'positive' if analyze_sentiment(x) > 0 else 'negative')
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_feedback'], df['sentiment_label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train)
    preds = clf.predict(X_test_tfidf)
    
    print(classification_report(y_test, preds))
    return clf, vectorizer
