# Fake Mail Classifier

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv("mail_data.csv")

# Checking for missing values
data = data.fillna("")

# Converting labels into numerical values
data.replace({'Category': {'spam': 0, 'ham': 1}}, inplace=True)

# Separating features and labels
X = data['Message']
Y = data['Category']

# Train-Test Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, Y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]

# Feature Extraction
vectorizer = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Training the Model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Training Accuracy
train_prediction = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_prediction)

print(f"Training Accuracy: {train_accuracy:.4f}")

# Testing Accuracy
test_prediction = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_prediction)

print(f"Testing Accuracy: {test_accuracy:.4f}")

# Predictive System
while True:
    message = input("\nEnter Email Message (or type 'exit' to quit): ")

    if message.lower() == "exit":
        print("Exiting...")
        break

    message_features = vectorizer.transform([message])

    prediction = model.predict(message_features)

    if prediction[0] == 0:
        print("📩 Spam Email")
    else:
        print("✅ Ham (Legitimate) Email")