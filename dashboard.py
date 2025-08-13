import streamlit as st
from src.data_cleaning import load_and_clean_data
from src.nlp_analysis import analyze_sentiment, topic_modeling
from src.predictive_model import train_sentiment_classifier

# Set dashboard title
st.title("📊 Customer Feedback Analysis Dashboard")

# File uploader for user to upload feedback data
uploaded_file = st.file_uploader("Upload a CSV file containing feedback", type=["csv"])

if uploaded_file:
    # Step 1: Load and clean the uploaded data
    df = load_and_clean_data(uploaded_file)
    
    st.write("### 📄 Sample Feedback Data")
    st.write(df.head())  # Show first few rows of cleaned data

    # Step 2: Sentiment Analysis
    df['sentiment_score'] = df['cleaned_feedback'].apply(analyze_sentiment)
    st.write("### Sentiment Distribution")
    st.bar_chart(df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative').value_counts())

    # Step 3: Topic Modeling
    lda_model, topics = topic_modeling(df['cleaned_feedback'], num_topics=5)
    st.write("### 📝 Discovered Topics")
    for idx, topic in topics:
        st.write(f"**Topic {idx}:** {topic}")

    # Step 4: Train and evaluate ML model
    clf, vectorizer = train_sentiment_classifier(df)
    st.success("✅ Sentiment classifier trained successfully!")
