import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# =========================
# Load Data
# =========================

data = pd.read_csv("Train.csv")

# =========================
# Data Cleaning
# =========================

data['Item_Weight'] = data['Item_Weight'].fillna(
    data['Item_Weight'].mean()
)

mode_data = data.pivot_table(
    values='Outlet_Size',
    columns='Outlet_Type',
    aggfunc=lambda x: x.mode()[0]
)

missing_values = data['Outlet_Size'].isnull()

data.loc[missing_values, 'Outlet_Size'] = (
    data.loc[missing_values, 'Outlet_Type']
    .apply(lambda x: mode_data[x][0])
)

# =========================
# Features & Target
# =========================

X = data.drop("Item_Outlet_Sales", axis=1)
Y = data["Item_Outlet_Sales"]

# =========================
# Train Test Split
# =========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# =========================
# Preprocessing
# =========================

num_cols = X_train.select_dtypes(
    include=['int64', 'float64']
).columns

cat_cols = X_train.select_dtypes(
    include=['object']
).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# =========================
# Model
# =========================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        random_state=42
    ))
])

# =========================
# Training
# =========================

model.fit(X_train, Y_train)

# =========================
# Evaluation
# =========================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_r2 = r2_score(Y_train, train_pred)
test_r2 = r2_score(Y_test, test_pred)

print(f"Train R2 Score : {train_r2:.4f}")
print(f"Test R2 Score  : {test_r2:.4f}")

# =========================
# Predictive System
# =========================

input_data = [
    'FDL48',
    19.35,
    'Regular',
    0.082601537,
    'Baking Goods',
    50.1034,
    'OUT018',
    2009,
    'Medium',
    'Tier 3',
    'Supermarket Type2'
]

input_df = pd.DataFrame(
    [input_data],
    columns=X.columns
)

prediction = model.predict(input_df)

print("\nPredicted Sales:")
print(round(prediction[0], 2))