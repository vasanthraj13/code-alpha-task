import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------
# FIX: Ensure NLTK resources are downloaded
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# App Title
# -----------------------------
st.title("💬 FAQ Chatbot")

# -----------------------------
# Sample FAQ Dataset
# -----------------------------
faq_data = {
    "Question": [
        "What is CodeAlpha?",
        "How do I apply for internship?",
        "Do you provide certificates?",
        "Is there job placement support?"
    ],
    "Answer": [
        "CodeAlpha is a software development company focused on AI and emerging technologies.",
        "You can apply for internships on the CodeAlpha website by filling out the form.",
        "Yes, we provide QR-verified completion certificates.",
        "Yes, job placement support is available based on performance."
    ]
}

df = pd.DataFrame(faq_data)

# -----------------------------
# Preprocessing Function
# -----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered)

# Preprocess questions
questions = [preprocess(q) for q in df['Question']]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(questions)

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please type a question!")
    else:
        user_processed = preprocess(user_input)
        user_vec = vectorizer.transform([user_processed])
        question_vecs = vectorizer.transform(questions)

        # Cosine similarity to find best match
        similarity = cosine_similarity(user_vec, question_vecs)
        best_idx = similarity.argmax()

        st.write("💡 Answer:")
        st.success(df['Answer'][best_idx])
