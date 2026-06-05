# main.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return [name, rmse, mae, r2]


def main():

    print("Loading dataset...")
    data = pd.read_csv("housing.csv")

    # Create income category
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    X = data.drop(["median_house_value", "income_cat"], axis=1)
    y = data["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(include="object").columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            random_state=42
        )
    }

    results = []

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train_prepared, y_train)

        result = evaluate_model(
            name,
            model,
            X_test_prepared,
            y_test
        )

        results.append(result)

        if result[1] < best_rmse:
            best_rmse = result[1]
            best_model = model

    comparison_df = pd.DataFrame(
        results,
        columns=["Model", "RMSE", "MAE", "R2 Score"]
    )

    comparison_df = comparison_df.sort_values("RMSE")

    print("\nModel Comparison")
    print(comparison_df)

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])

    print("\nSaving best model...")

    joblib.dump(
        final_pipeline,
        "house_price_pipeline.pkl"
    )

    print("Saved as house_price_pipeline.pkl")


if __name__ == "__main__":
    main()