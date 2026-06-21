import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from xgboost import XGBRegressor

# Loading Dataset
data = pd.read_csv("supplement_impact_data.csv")

# Encoding Gender
data.replace({
    'Gender': {
        'Non-Binary': 0,
        'Male': 1,
        'Female': 2
    }
}, inplace=True)

# Encoding Supplement
data.replace({
    'Supplement': {
        'Both': 0,
        'Mass Gainer': 1,
        'Creatine Monohydrate': 2
    }
}, inplace=True)

# Converting Strength Gain from Percentage String to Integer
data["Strength_Gain"] = (
    data["Strength_Gain"]
    .str.replace("%", "", regex=False)
    .astype(int)
)

# Encoding Primary Benefit
le = LabelEncoder()
data["Primary_Benefit"] = le.fit_transform(data["Primary_Benefit"])

# Splitting Features and Target
X = data.drop(columns=["ID", "Strength_Gain"], axis=1)
Y = data["Strength_Gain"]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = XGBRegressor(
    random_state=42
)

model.fit(X_train, Y_train)

# Training Score
train_pred = model.predict(X_train)

train_score = metrics.r2_score(
    Y_train,
    train_pred
)

print("Training R² Score:", train_score)

# Testing Score
test_pred = model.predict(X_test)

test_score = metrics.r2_score(
    Y_test,
    test_pred
)

print("Testing R² Score:", test_score)

# Predictive System
input_data = [64, 1, 2, 5, 79.1, 81.2, 3]

input_data_as_np_array = np.asarray(input_data)

input_data_reshaped = input_data_as_np_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

print(
    f"The model has predicted the Strength Gain is: {prediction[0]:.2f}%"
)