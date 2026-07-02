import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================================
# Load Dataset
# ==========================================================

data = pd.read_csv("creditcard.csv")

# ==========================================================
# Exploratory Data Analysis
# ==========================================================

print(data.head())
print(data.info())
print(data.isnull().sum())

print("\nClass Distribution")
print(data["Class"].value_counts())

# ==========================================================
# Handling Imbalanced Dataset
# ==========================================================

legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

print("\nLegitimate Transactions :", legit.shape)
print("Fraudulent Transactions :", fraud.shape)

# Random Undersampling
legit_sample = legit.sample(
    n=len(fraud),
    random_state=42
)

new_data = pd.concat(
    [legit_sample, fraud],
    axis=0
)

print("\nBalanced Dataset")
print(new_data["Class"].value_counts())

# ==========================================================
# Feature & Target Split
# ==========================================================

X = new_data.drop(
    columns=["Class", "Time"],
    axis=1
)

Y = new_data["Class"]

# ==========================================================
# Train Test Split
# ==========================================================

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

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# Build Deep Learning Model
# ==========================================================

model = Sequential([

    tf.keras.Input(shape=(29,)),

    Dense(32, activation="relu"),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")

])

# ==========================================================
# Compile Model
# ==========================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ==========================================================
# Train Model
# ==========================================================

history = model.fit(

    X_train,
    Y_train,

    validation_split=0.1,

    epochs=10,

    batch_size=32

)

# ==========================================================
# Accuracy Graph
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(
    ["Training", "Validation"],
    loc="lower right"
)

plt.show()

# ==========================================================
# Loss Graph
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(
    ["Training", "Validation"],
    loc="upper right"
)

plt.show()

# ==========================================================
# Model Evaluation
# ==========================================================

loss, accuracy = model.evaluate(
    X_test,
    Y_test
)

print(f"\nTest Accuracy : {accuracy:.4f}")

# ==========================================================
# Predictions
# ==========================================================

Y_pred = model.predict(X_test)

Y_pred = (Y_pred > 0.5).astype(int)

print("\nAccuracy Score")
print(accuracy_score(Y_test, Y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report")
print(classification_report(Y_test, Y_pred))

# ==========================================================
# Predictive System
# ==========================================================

input_data = [
    -0.966271711572087,
    -0.185226008082898,
    1.79299333957872,
    -0.863291275036453,
    -0.0103088796030823,
    1.24720316752486,
    0.23760893977178,
    0.377435874652262,
    -1.38702406270197,
    -0.0549519224713749,
    -0.226487263835401,
    0.178228225877303,
    0.507756869957169,
    -0.28792374549456,
    -0.631418117709045,
    -1.0596472454325,
    -0.684092786345479,
    1.96577500349538,
    -1.2326219700892,
    -0.208037781160366,
    -0.108300452035545,
    0.00527359678253453,
    -0.190320518742841,
    -1.17557533186321,
    0.647376034602038,
    -0.221928844458407,
    0.0627228487293033,
    0.0614576285006353,
    123.50
]

input_data = np.asarray(input_data)

input_data = input_data.reshape(1, -1)

input_data = scaler.transform(input_data)

prediction = model.predict(input_data)

print("\n" + "=" * 50)

if prediction[0][0] > 0.5:
    print("Prediction : Fraudulent Transaction")
else:
    print("Prediction : Legitimate Transaction")

print("=" * 50)

# ==========================================================
# Save Model
# ==========================================================

model.save("credit_card_fraud_detection_dl.keras")

print("\nModel saved successfully.")