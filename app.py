import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# --- Load stopwords ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Load and preprocess data ---

news_df = pd.read_csv('training.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['title'].fillna('') + ' ' + news_df['text'].fillna('')
# --- Encode labels: 'real' -> 1, 'fake' -> 0 ---
news_df['label'] = news_df['label'].map({'real': 1, 'fake': 0})

# --- Stemming ---
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

news_df['content'] = news_df['content'].apply(stemming)

# --- Train/Test split ---
from sklearn.model_selection import train_test_split
X = news_df['content'].values
y = news_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# --- TF-IDF + Logistic Regression ---
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article text below to check if it's **real** or **fake**.")

user_input = st.text_area("Paste your news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and vectorize
        processed_input = stemming(user_input)
        input_vectorized = tfidf_vectorizer.transform([processed_input])
        prediction = model.predict(input_vectorized)[0]
        print("Prediction:", prediction)
        
        # Display result
        if prediction == 1:
            st.success("✅ This news looks REAL.")
        else:
            st.error("🚨 This news seems FAKE!")

st.write("---")
st.caption("Model: Logistic Regression | Features: TF-IDF (max 5000) | Dataset: training.csv")
