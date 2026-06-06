# ==============================
# Car Price Prediction System
# ==============================

# Importing Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ==============================
# Loading Dataset
# ==============================

data = pd.read_csv("car data.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

# ==============================
# Categorical Data Analysis
# ==============================

print("\nFuel Type Counts:")
print(data["Fuel_Type"].value_counts())

print("\nSeller Type Counts:")
print(data["Seller_Type"].value_counts())

print("\nTransmission Counts:")
print(data["Transmission"].value_counts())

# ==============================
# Data Preprocessing
# ==============================

data.replace(
    {
        "Fuel_Type": {
            "Petrol": 0,
            "Diesel": 1,
            "CNG": 2
        },
        "Seller_Type": {
            "Dealer": 0,
            "Individual": 1
        },
        "Transmission": {
            "Manual": 0,
            "Automatic": 1
        }
    },
    inplace=True
)

print("\nEncoded Dataset:")
print(data.head())

# ==============================
# Feature Selection
# ==============================

X = data.drop(columns=["Car_Name", "Selling_Price"], axis=1)
Y = data["Selling_Price"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", Y.shape)

# ==============================
# Train-Test Split
# ==============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.1,
    random_state=42
)

# ==============================
# Model Training
# ==============================

model = LinearRegression()

model.fit(X_train, Y_train)

# ==============================
# Training Evaluation
# ==============================

train_pred = model.predict(X_train)

train_r2 = metrics.r2_score(Y_train, train_pred)

print("\nTraining R² Score:", train_r2)

# ==============================
# Testing Evaluation
# ==============================

test_pred = model.predict(X_test)

test_r2 = metrics.r2_score(Y_test, test_pred)

print("Testing R² Score:", test_r2)

# ==============================
# Visualization
# ==============================

plt.figure(figsize=(6, 5))
plt.scatter(Y_train, train_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Training Data)")
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(Y_test, test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Testing Data)")
plt.show()

# ==============================
# Predicting New Car Price
# ==============================

input_data = [2007, 9.54, 70000, 0, 1, 0, 0]

input_data_np = np.asarray(input_data)
input_data_reshaped = input_data_np.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

print(
    f"\nPredicted Selling Price: ₹{prediction[0] * 100000:.2f}"
)