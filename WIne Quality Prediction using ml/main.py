import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("winequality-red.csv")

# Basic Information
print("Dataset Shape:", data.shape)
print("\nMissing Values:\n")
print(data.isnull().sum())

print("\nQuality Distribution:\n")
print(data["quality"].value_counts())

# Correlation Matrix
correlation = data.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(
    correlation,
    cbar=True,
    annot=True,
    fmt=".1f",
    annot_kws={"size": 8},
    cmap="Blues"
)
plt.title("Correlation Heatmap")
plt.show()

# Data Preprocessing

X = data.drop(columns="quality", axis=1)

# Convert quality into binary classification
# Good Wine = 1 (quality >= 7)
# Bad Wine = 0 (quality < 7)

Y = data["quality"].apply(lambda value: 1 if value >= 7 else 0)

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
model = RandomForestClassifier(random_state=42)

model.fit(X_train, Y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)

print("\nTraining Accuracy:", train_accuracy)

# Testing Accuracy
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_pred)

print("Testing Accuracy:", test_accuracy)

# Predictive System
input_data = (
    7.9,
    0.60,
    0.06,
    1.6,
    0.069,
    15.0,
    59.0,
    0.9964,
    3.30,
    0.46,
    9.4
)

input_data_np = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data_np)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Wine Quality is Good")
else:
    print("Wine Quality is Bad")