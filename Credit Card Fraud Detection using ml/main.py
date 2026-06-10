import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Dataset
data = pd.read_csv("creditcard.csv")

# Dataset Information
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data['Class'].value_counts())

# 0 -> Legit Transaction
# 1 -> Fraudulent Transaction

# Separate Legit and Fraud Transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

print("Legit Transactions Shape:", legit.shape)
print("Fraud Transactions Shape:", fraud.shape)

# Statistical Measures
print("\nLegit Transaction Amount Statistics:")
print(legit.Amount.describe())

print("\nFraud Transaction Amount Statistics:")
print(fraud.Amount.describe())

# Compare Mean Values
print("\nClass-wise Mean Values:")
print(data.groupby('Class').mean())

# Under Sampling
legit_sample = legit.sample(n=492, random_state=42)

# Create Balanced Dataset
new_data = pd.concat([legit_sample, fraud], axis=0)

print("\nBalanced Dataset Class Distribution:")
print(new_data['Class'].value_counts())

# Features and Target
X = new_data.drop(columns='Class', axis=1)
Y = new_data['Class']

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

print("Training Data Shape:", X_train.shape)
print("Training Labels Shape:", Y_train.shape)

# Model Training
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_score = accuracy_score(Y_train, train_pred)

print("\nTraining Accuracy:", train_score)

# Testing Accuracy
test_pred = model.predict(X_test)
test_score = accuracy_score(Y_test, test_pred)

print("Testing Accuracy:", test_score)

# Predictive System
input_data = [
    1, -0.966271711572087, -0.185226008082898, 1.79299333957872,
    -0.863291275036453, -0.0103088796030823, 1.24720316752486,
    0.23760893977178, 0.377435874652262, -1.38702406270197,
    -0.0549519224713749, -0.226487263835401, 0.178228225877303,
    0.507756869957169, -0.28792374549456, -0.631418117709045,
    -1.0596472454325, -0.684092786345479, 1.96577500349538,
    -1.2326219700892, -0.208037781160366, -0.108300452035545,
    0.00527359678253453, -0.190320518742841, -1.17557533186321,
    0.647376034602038, -0.221928844458407, 0.0627228487293033,
    0.0614576285006353, 123.5
]

input_data = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data)

if prediction[0] == 0:
    print("\nModel Prediction: Legit Transaction")
else:
    print("\nModel Prediction: Fraudulent Transaction")