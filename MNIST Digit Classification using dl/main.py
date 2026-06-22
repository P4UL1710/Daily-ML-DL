import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import confusion_matrix

# Loading Dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Data Scaling
X_train = X_train / 255.0
X_test = X_test / 255.0

# Building Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='sigmoid')
])

# Compiling Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training Model
model.fit(
    X_train,
    Y_train,
    epochs=10
)

# Model Evaluation
loss, accuracy = model.evaluate(
    X_test,
    Y_test
)

print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
Y_pred = model.predict(X_test)

Y_pred_labels = [np.argmax(i) for i in Y_pred]

# Confusion Matrix
conf_mat = confusion_matrix(
    Y_test,
    Y_pred_labels
)

plt.figure(figsize=(15, 7))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('MNIST Confusion Matrix')
plt.show()

# Predictive System
index = np.random.randint(0, len(X_test))

input_image = X_test[index]

plt.imshow(input_image, cmap='gray')
plt.axis('off')
plt.show()

prediction = model.predict(
    input_image.reshape(1, 28, 28)
)

predicted_digit = np.argmax(prediction)

print("=" * 50)
print(f"The model predicts the digit is: {predicted_digit}")
print(f"Actual digit is: {Y_test[index]}")
print("=" * 50)