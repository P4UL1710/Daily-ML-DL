import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.metrics import accuracy_score

# Importing Data
data = pd.read_csv("Data.csv")

# Data Preprocessing
data = data.dropna()

# Data Encoding
data.replace({'Gender': {'Male': 0, 'Female': 1}}, inplace=True)
data.replace({'Married': {'Yes': 1, 'No': 0}}, inplace=True)
data.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)
data.replace({'Self_Employed': {'No': 0, 'Yes': 1}}, inplace=True)
data.replace({'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}}, inplace=True)
data.replace({'Education': {'Graduate': 0, 'Not Graduate': 1}}, inplace=True)

# Dependents Column
data['Dependents'] = data['Dependents'].replace('3+', 4).astype(int)

# Separating Features and Labels
X = data.drop(columns=["Loan_Status", "Loan_ID"])
Y = data["Loan_Status"]

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

# Model Training
model = svm.SVC(kernel='linear')

model.fit(X_train, Y_train)

# Training Accuracy
train_pred = model.predict(X_train)
train_score = accuracy_score(Y_train, train_pred)

print("Training Accuracy:", train_score)

# Testing Accuracy
test_pred = model.predict(X_test)
test_score = accuracy_score(Y_test, test_pred)

print("Testing Accuracy:", test_score)

# Predictive System
input_data = [
    0,      # Gender
    1,      # Married
    1,      # Dependents
    0,      # Education
    0,      # Self_Employed
    4583,   # ApplicantIncome
    1508.0, # CoapplicantIncome
    128.0,  # LoanAmount
    360.0,  # Loan_Amount_Term
    1.0,    # Credit_History
    0       # Property_Area
]

input_data_np = np.asarray(input_data)
input_data_reshaped = input_data_np.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 1:
    print("You are eligible for taking a loan.")
else:
    print("You are not eligible for the loan.")