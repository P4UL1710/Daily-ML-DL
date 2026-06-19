import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv("data.csv")

# Data Cleaning
data.drop(columns="Unnamed: 32", axis=1, inplace=True)

# Converting categorical values to numerical
data.replace({"diagnosis": {"B": 0, "M": 1}}, inplace=True)

# Splitting features and target
X = data.drop(columns=["id", "diagnosis"], axis=1)
Y = data["diagnosis"]

# Train-Test Split using Stratified Sampling
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
model = LogisticRegression(max_iter=10000)

model.fit(X_train, Y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_score = accuracy_score(Y_train, train_pred)

print("Training Accuracy:", train_score)

# Testing Accuracy
test_pred = model.predict(X_test)
test_score = accuracy_score(Y_test, test_pred)

print("Testing Accuracy:", test_score)

# Building a Predictive System
input_data = [
    20.29, 14.34, 135.1, 1297, 0.1003, 0.1328, 0.198,
    0.1043, 0.1809, 0.05883, 0.7572, 0.7813, 5.438,
    94.44, 0.01149, 0.02461, 0.05688, 0.01885,
    0.01756, 0.005115, 22.54, 16.67, 152.2, 1575,
    0.1374, 0.205, 0.4, 0.1625, 0.2364, 0.07678
]

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The tumor is Benign (Not Malignant).")
else:
    print("The tumor is Malignant.")