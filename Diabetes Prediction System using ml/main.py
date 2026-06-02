import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("diabetes.csv")

# Basic Information
print("Dataset Shape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())
print("\nClass Distribution:\n", data["Outcome"].value_counts())

# Separating Features and Target
X = data.drop(columns="Outcome", axis=1)
Y = data["Outcome"]

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

X = pd.DataFrame(X_std)

# Train-Test Split
split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(X, Y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]

# Model Training
classifier = svm.SVC(kernel="linear")

classifier.fit(X_train, Y_train)

# Training Accuracy
train_pred = classifier.predict(X_train)
train_score = accuracy_score(Y_train, train_pred)

print(f"\nTraining Accuracy: {train_score:.4f}")

# Testing Accuracy
test_pred = classifier.predict(X_test)
test_score = accuracy_score(Y_test, test_pred)

print(f"Testing Accuracy: {test_score:.4f}")

# Predictive System
input_data = (5, 116, 74, 0, 0, 25.6, 0.201, 30)

input_data_as_np_array = np.asarray(input_data)

input_data_reshaped = input_data_as_np_array.reshape(1, -1)

input_data_scaled = scaler.transform(input_data_reshaped)

prediction = classifier.predict(input_data_scaled)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("The person is Non-Diabetic")
else:
    print("The person is Diabetic")