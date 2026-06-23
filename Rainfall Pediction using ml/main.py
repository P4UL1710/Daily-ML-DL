import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import pickle

# Loading Dataset
data = pd.read_csv("Rainfall.csv")

# Data Cleaning
data.columns = data.columns.str.strip()

data = data.drop(columns=["day"])

# Missing Value Handling
data["winddirection"] = data["winddirection"].fillna(
    data["winddirection"].mode()[0]
)

data["windspeed"] = data["windspeed"].fillna(
    data["windspeed"].median()
)

# Encoding Target Variable
data["rainfall"] = data["rainfall"].map({
    "yes": 1,
    "no": 0
})

# Exploratory Data Analysis

sns.set(style="whitegrid")

plt.figure(figsize=(15, 10))

for i, column in enumerate(
    [
        "pressure",
        "maxtemp",
        "temparature",
        "mintemp",
        "dewpoint",
        "humidity",
        "cloud",
        "sunshine",
        "windspeed"
    ],
    1
):
    plt.subplot(3, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")

plt.tight_layout()
plt.show()

# Rainfall Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="rainfall", data=data)
plt.title("Rainfall Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# Boxplots
plt.figure(figsize=(15, 10))

for i, column in enumerate(
    [
        "pressure",
        "maxtemp",
        "temparature",
        "mintemp",
        "dewpoint",
        "humidity",
        "cloud",
        "sunshine",
        "windspeed"
    ],
    1
):
    plt.subplot(3, 3, i)
    sns.boxplot(x=data[column])
    plt.title(f"Boxplot of {column}")

plt.tight_layout()
plt.show()

# Dropping Highly Correlated Features
data = data.drop(
    columns=[
        "maxtemp",
        "temparature",
        "mintemp"
    ]
)

# Handling Imbalanced Data
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_downsampled = pd.concat(
    [df_majority_downsampled, df_minority]
)

df_downsampled = (
    df_downsampled
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

# Feature & Target Split
X = df_downsampled.drop(columns=["rainfall"])
Y = df_downsampled["rainfall"]

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

# Random Forest Model
model = RandomForestClassifier(
    random_state=42
)

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Hyperparameter Tuning
grid_search_rf = GridSearchCV(
    estimator=model,
    param_grid=param_grid_rf,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search_rf.fit(X_train, Y_train)

best_rf_model = grid_search_rf.best_estimator_

print(
    "Best Parameters:",
    grid_search_rf.best_params_
)

# Cross Validation
cv_score = cross_val_score(
    best_rf_model,
    X_train,
    Y_train,
    cv=5
)

print("Cross Validation Scores:", cv_score)
print(
    "Mean CV Score:",
    np.mean(cv_score)
)

# Model Evaluation
y_pred = best_rf_model.predict(X_test)

print(
    "Test Accuracy:",
    accuracy_score(Y_test, y_pred)
)

print(
    "Confusion Matrix:\n",
    confusion_matrix(Y_test, y_pred)
)

print(
    "Classification Report:\n",
    classification_report(Y_test, y_pred)
)

# Predictive System
input_data = (
    1015.9,
    19.9,
    95,
    81,
    0.0,
    40.0,
    13.7
)

input_df = pd.DataFrame(
    [input_data],
    columns=[
        "pressure",
        "dewpoint",
        "humidity",
        "cloud",
        "sunshine",
        "winddirection",
        "windspeed"
    ]
)

prediction = best_rf_model.predict(input_df)

print(
    "\nPrediction Result:",
    "Rainfall"
    if prediction[0] == 1
    else "No Rainfall"
)

# Save Model
model_data = {
    "model": best_rf_model,
    "feature_names": X.columns.tolist()
}

with open(
    "rainfall_prediction_model.pkl",
    "wb"
) as file:
    pickle.dump(model_data, file)

print("\nModel saved successfully.")