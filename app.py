import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("email_dataset.csv")   # dataset file

# Preprocessing function
def clean_email(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return " ".join(tokens)

df["cleaned"] = df["email"].apply(clean_email)

# -------------------------------
# TRAIN MODEL
# -------------------------------
X = df["cleaned"]
y = df["label"]

tfidf = TfidfVectorizer()
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

# -------------------------------
# REPLY GENERATOR
# -------------------------------
def generate_reply(sentiment):
    if sentiment == "positive":
        return (
            "Dear Sender,\n\n"
            "Thank you for your kind message. I'm glad to hear it.\n"
            "Let me know if you need any more assistance.\n\nRegards,\nYour Assistant"
        )
    elif sentiment == "neutral":
        return (
            "Dear Sender,\n\n"
            "Thank you for your email. I will get back to you with the details shortly.\n\nRegards,\nYour Assistant"
        )
    elif sentiment == "negative":
        return (
            "Dear Sender,\n\n"
            "I sincerely apologize for the inconvenience caused.\n"
            "I will look into this issue immediately and send you an update soon.\n\nRegards,\nYour Assistant"
        )
    elif sentiment == "urgent":
        return (
            "Dear Sender,\n\n"
            "I understand that this matter is urgent.\n"
            "I am prioritizing your request and will respond shortly.\n\nRegards,\nYour Assistant"
        )

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📧 AI Email Auto-Responder")
st.write("Paste an email below and get an automatic reply!")

email_input = st.text_area("Enter the email you received:")

if st.button("Generate Reply"):
    cleaned = clean_email(email_input)
    vector = tfidf.transform([cleaned])
    sentiment = model.predict(vector)[0]

    st.subheader("Detected Sentiment:")
    st.success(sentiment.upper())

    st.subheader("AI Generated Reply:")
    reply = generate_reply(sentiment)
    st.code(reply)

st.write(f"Model Accuracy: {acc * 100:.2f}%")
