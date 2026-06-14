import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("parkinsons.csv")

# Basic preprocessing
data.drop(columns='name', axis=1, inplace=True)

# Features and Target
X = data.drop(columns='status', axis=1)
Y = data['status']

# Train-Test Split
split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.1,
    random_state=42
)

for train_index, test_index in split.split(X, Y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = svm.SVC()

model.fit(X_train, Y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_score = accuracy_score(Y_train, train_pred)

print("Training Accuracy:", train_score)

# Testing Accuracy
test_pred = model.predict(X_test)
test_score = accuracy_score(Y_test, test_pred)

print("Testing Accuracy:", test_score)

# Predictive System
input_data = [
    116.01400, 141.78100, 110.65500,
    0.01284, 0.00011, 0.00655, 0.00908,
    0.01966, 0.06425, 0.58400, 0.03490,
    0.04825, 0.04465, 0.10470, 0.01767,
    19.64900, 0.417356, 0.823484,
    -3.747787, 0.234513, 2.332180,
    0.410335
]

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

input_data_scaled = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_scaled)

if prediction[0] == 0:
    print("The person is Healthy.")
else:
    print("The person has Parkinson's Disease.")