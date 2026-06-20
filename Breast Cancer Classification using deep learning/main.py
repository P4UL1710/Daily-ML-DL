import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Loading the dataset
data = pd.read_csv("data.csv")

# Data Cleaning
data.drop(columns="Unnamed: 32", axis=1, inplace=True)

# Converting categorical values to numerical
data.replace({"diagnosis": {"B": 0, "M": 1}}, inplace=True)

# Splitting features and target
X = data.drop(columns=["diagnosis", "id"], axis=1)
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

# Feature Scaling
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Building Neural Network
tf.random.set_seed(3)

model = Sequential([
    Flatten(input_shape=(30,)),
    Dense(20, activation="relu"),
    Dense(2, activation="sigmoid")
])

# Compiling Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Training Model
history = model.fit(
    X_train_std,
    Y_train,
    validation_split=0.1,
    epochs=10
)

# Accuracy Graph
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.legend(
    ["Training Data", "Validation Data"],
    loc="lower right"
)

plt.show()

# Loss Graph
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")

plt.legend(
    ["Training Data", "Validation Data"],
    loc="lower right"
)

plt.show()

# Model Evaluation
loss, accuracy = model.evaluate(X_test_std, Y_test)

print("Test Accuracy:", accuracy)

# Predictions
Y_pred = model.predict(X_test_std)

y_pred_labels = [np.argmax(i) for i in Y_pred]

# Predictive System
input_data = [
    20.29, 14.34, 135.1, 1297, 0.1003, 0.1328,
    0.198, 0.1043, 0.1809, 0.05883, 0.7572,
    0.7813, 5.438, 94.44, 0.01149, 0.02461,
    0.05688, 0.01885, 0.01756, 0.005115,
    22.54, 16.67, 152.2, 1575, 0.1374,
    0.205, 0.4, 0.1625, 0.2364, 0.07678
]

input_data_as_np_array = np.asarray(input_data)

input_data_reshaped = input_data_as_np_array.reshape(1, -1)

input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)

prediction_label = np.argmax(prediction)

if prediction_label == 0:
    print("The tumor is Benign (Non-Malignant).")
else:
    print("The tumor is Malignant.")