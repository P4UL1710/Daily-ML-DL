import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
data = pd.read_csv("insurance.csv")

# Encoding categorical columns
data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
data.replace({'smoker': {'no': 0, 'yes': 1}}, inplace=True)
data.replace({
    'region': {
        'southeast': 0,
        'southwest': 1,
        'northeast': 2,
        'northwest': 3
    }
}, inplace=True)

# Split Features and Target
X = data.drop(columns='charges', axis=1)
Y = data['charges']

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, Y_train)

# ---------------------------
# Predictive System
# ---------------------------

input_data = [37, "female", 27.74, 3, "no", "northwest"]

input_df = pd.DataFrame(
    [input_data],
    columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
)

# Encoding input data
input_df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
input_df.replace({'smoker': {'no': 0, 'yes': 1}}, inplace=True)
input_df.replace({
    'region': {
        'southeast': 0,
        'southwest': 1,
        'northeast': 2,
        'northwest': 3
    }
}, inplace=True)

# Prediction
prediction = model.predict(input_df)

print("\n================================")
print(f"Predicted Insurance Charge: ₹{prediction[0]:,.2f}")
print("================================")