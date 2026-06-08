import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = pd.read_csv("data.csv")

# Basic Information
print("Dataset Shape:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())

# Correlation Analysis
correlation = data.drop('Date', axis=1).corr()

plt.figure(figsize=(8, 8))
sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    annot=True,
    fmt='.1f',
    annot_kws={'size': 8},
    cmap="Blues"
)
plt.title("Correlation Heatmap")
plt.show()

# Correlation with Gold Price
print("\nCorrelation with GLD:")
print(correlation["GLD"])

# Distribution Plot
sns.displot(data["GLD"], color="green")
plt.show()

# Features and Target
X = data.drop(columns=["Date", "GLD"], axis=1)
Y = data["GLD"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", Y.shape)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Training Evaluation
train_pred = model.predict(X_train)
train_score = metrics.r2_score(Y_train, train_pred)

print(f"\nTraining R² Score: {train_score:.4f}")

# Testing Evaluation
test_pred = model.predict(X_test)
test_score = metrics.r2_score(Y_test, test_pred)

print(f"Testing R² Score: {test_score:.4f}")

# Actual vs Predicted Plot
Y_test = list(Y_test)

plt.figure(figsize=(10, 5))
plt.plot(Y_test, color="blue", label="Actual Value")
plt.plot(test_pred, color="green", label="Predicted Value")

plt.title("Actual Gold Price vs Predicted Gold Price")
plt.xlabel("Number of Samples")
plt.ylabel("Gold Price")
plt.legend()
plt.show()