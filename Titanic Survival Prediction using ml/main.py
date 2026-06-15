import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("train.csv")

# Basic Information
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())

# Check categorical values
print(data['Embarked'].value_counts())
print(data['Sex'].value_counts())

# Encoding
data.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
data.replace({'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Handling Missing Values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].median(), inplace=True)

print(data.isnull().sum())

# Data Visualization
sns.set()

plt.figure(figsize=(5, 4))
sns.countplot(x='Survived', data=data)
plt.title("Survival Distribution")
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(x='Sex', data=data)
plt.title("Gender Distribution")
plt.show()

# Feature and Target Separation
X = data.drop(
    columns=['PassengerId', 'Name', 'Survived', 'Ticket'],
    axis=1
)

Y = data['Survived']

# Train-Test Split
from sklearn.model_selection import StratifiedShuffleSplit

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
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# Evaluation
from sklearn.metrics import accuracy_score

train_pred = model.predict(X_train)
train_score = accuracy_score(Y_train, train_pred)

print("Training Accuracy:", train_score)

test_pred = model.predict(X_test)
test_score = accuracy_score(Y_test, test_pred)

print("Testing Accuracy:", test_score)

# Predictive System
input_data = [3, 0, 2, 3, 1, 21.075, 0]

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("Model predicts: Passenger did NOT survive.")
else:
    print("Model predicts: Passenger survived.")