import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


def main():
    # Load datasets
    data_cal = pd.read_csv("calories.csv")
    data_ez = pd.read_csv("exercise.csv")

    # Merge datasets
    data = pd.concat([data_ez, data_cal["Calories"]], axis=1)

    # Basic checks
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDataset info:")
    print(data.info())

    print("\nGender counts:")
    print(data["Gender"].value_counts())

    # Visualization
    sns.distplot(data["Age"])
    plt.title("Age Distribution")
    plt.show()

    # Encode Gender column
    data.replace({"Gender": {"female": 0, "male": 1}}, inplace=True)

    # Remove User_ID column
    data = data.drop(columns="User_ID", axis=1)

    # Correlation heatmap
    corr = data.corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr,
        cbar=True,
        square=True,
        fmt=".1f",
        annot=True,
        annot_kws={"size": 8},
        cmap="Blues",
    )
    plt.title("Correlation Heatmap")
    plt.show()

    # Split features and target
    X = data.drop(columns="Calories", axis=1)
    Y = data["Calories"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Train model
    model = XGBRegressor()
    model.fit(X_train, Y_train)

    # Training evaluation
    train_pred = model.predict(X_train)
    train_score = metrics.r2_score(Y_train, train_pred)
    print(f"\nTraining R² Score: {train_score:.4f}")

    # Testing evaluation
    test_pred = model.predict(X_test)
    test_score = metrics.r2_score(Y_test, test_pred)
    print(f"Testing R² Score: {test_score:.4f}")

    # Prediction example
    input_data = [0, 27, 154.0, 58.0, 10.0, 81.0, 39.8]

    input_array = np.asarray(input_data).reshape(1, -1)

    prediction = model.predict(input_array)

    print(f"\nModel predicted calories burnt: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()