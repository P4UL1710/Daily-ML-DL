import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('stopwords')

print("=" * 60)
print("FAKE NEWS DETECTION SYSTEM")
print("=" * 60)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

data = pd.concat([fake_df, true_df], axis=0)

print(f"\nDataset Shape: {data.shape}")

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

data["content"] = (
    data["title"].astype(str)
    + " "
    + data["subject"].astype(str)
    + " "
    + data["text"].astype(str)
)

data = data[["content", "label"]]

# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------

port_stem = PorterStemmer()

def stemming(content):

    content = re.sub('[^a-zA-Z]', ' ', content)

    content = content.lower()

    content = content.split()

    content = [
        port_stem.stem(word)
        for word in content
        if word not in stopwords.words('english')
    ]

    content = " ".join(content)

    return content

print("\nPreprocessing text...")

data["content"] = data["content"].apply(stemming)

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# TF-IDF Vectorization
# --------------------------------------------------

tfidf = TfidfVectorizer()

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

# --------------------------------------------------
# Model Training
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# Prediction System
# --------------------------------------------------

while True:

    print("\n" + "=" * 60)

    news = input("Enter News Article (or type 'exit'): ")

    if news.lower() == "exit":
        break

    processed_news = stemming(news)

    news_vector = tfidf.transform([processed_news])

    prediction = model.predict(news_vector)

    confidence = max(
        model.predict_proba(news_vector)[0]
    ) * 100

    print("\nPrediction Result:")

    if prediction[0] == 1:
        print(
            f"The news is likely REAL "
            f"({confidence:.2f}% confidence)"
        )
    else:
        print(
            f"The news is likely FAKE "
            f"({confidence:.2f}% confidence)"
        )

print("\nProgram Ended.")