from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.nlp_analysis import analyze_sentiment

def train_sentiment_classifier(df):
    """
    Trains a Logistic Regression model to classify sentiment (positive/negative).
    Steps:
    1. Create sentiment labels from polarity scores.
    2. Split the dataset into training and test sets.
    3. Vectorize text using TF-IDF.
    4. Train a Logistic Regression classifier.
    5. Evaluate the model and print performance metrics.
    Returns the trained classifier and the TF-IDF vectorizer.
    """
    # Label as positive if polarity > 0, else negative
    df['sentiment_label'] = df['cleaned_feedback'].apply(
        lambda x: 'positive' if analyze_sentiment(x) > 0 else 'negative'
    )
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_feedback'], df['sentiment_label'], test_size=0.2, random_state=42
    )
    
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression model
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train)
    
    # Evaluate performance
    preds = clf.predict(X_test_tfidf)
    print(classification_report(y_test, preds))
    
    return clf, vectorizer

