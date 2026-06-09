import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("data.csv")

# Features and Target
X = data.drop(columns=["target"], axis=1)
Y = data["target"]

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

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)

# Testing Accuracy
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# ------------------------------------
# Heart Disease Prediction
# ------------------------------------

input_data = [57, 1, 0, 140, 192, 0, 1, 148, 0, 0.4, 1, 0, 1]

input_data_np = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data_np)

if prediction[0] == 0:
    print("\nResult: The person does NOT have heart disease.")
else:
    print("\nResult: The person HAS heart disease.")