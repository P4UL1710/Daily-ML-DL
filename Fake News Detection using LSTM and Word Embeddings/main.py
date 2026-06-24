import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# Download stopwords
nltk.download('stopwords')

# Loading Dataset
data = pd.read_csv("data.csv")

# Data Cleaning
data.drop(columns="date", axis=1, inplace=True)

# Remove Missing Values
data = data.dropna()

# Features and Target
X = data.drop('label', axis=1)
Y = data['label']

# Vocabulary Size
voc_size = 10000

# Copy Dataset
messages = X.copy()
messages.reset_index(inplace=True)

# Text Preprocessing
ps = PorterStemmer()

corpus = []

for i in range(len(messages)):
    review = re.sub(
        '[^a-zA-Z]',
        ' ',
        str(messages['title'][i])
    )

    review = review.lower()
    review = review.split()

    review = [
        ps.stem(word)
        for word in review
        if word not in stopwords.words('english')
    ]

    review = ' '.join(review)

    corpus.append(review)

# One Hot Representation
one_hot_repr = [
    one_hot(words, voc_size)
    for words in corpus
]

# Padding
sent_length = 20

embedded_docs = pad_sequences(
    one_hot_repr,
    padding='pre',
    maxlen=sent_length
)

# Final Dataset
X_final = np.array(embedded_docs)
Y_final = np.array(Y)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_final,
    Y_final,
    test_size=0.2,
    random_state=42
)

# Model Building
model = Sequential([
    Input(shape=(sent_length,)),
    Embedding(voc_size, 40),

    LSTM(
        100,
        dropout=0.3,
        recurrent_dropout=0.3
    ),

    Dense(
        1,
        activation='sigmoid'
    )
])

# Compile Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model Summary
model.summary()

# Training
model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=10,
    batch_size=62
)

# Prediction
y_pred = model.predict(X_test)

y_pred = np.where(
    y_pred > 0.5,
    1,
    0
)

# Evaluation
print("\nAccuracy Score:")
print(accuracy_score(Y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# ======================================================
# Predictive System
# ======================================================

sample_news = "Scientists discover a revolutionary treatment for cancer"

review = re.sub('[^a-zA-Z]', ' ', sample_news)
review = review.lower()
review = review.split()

review = [
    ps.stem(word)
    for word in review
    if word not in stopwords.words('english')
]

review = ' '.join(review)

onehot_repr = one_hot(
    review,
    voc_size
)

embedded_doc = pad_sequences(
    [onehot_repr],
    padding='pre',
    maxlen=sent_length
)

prediction = model.predict(embedded_doc)

print("\n" + "=" * 50)

if prediction[0][0] > 0.5:
    print("Prediction: Fake News")
else:
    print("Prediction: Real News")

print("=" * 50)