import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("Data.csv", header=None)

# =========================
# EDA
# =========================
print("First 5 Rows:")
print(data.head())

print("\nShape:")
print(data.shape)

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nClass Distribution:")
print(data[60].value_counts())

# =========================
# FEATURE AND TARGET
# =========================
X = data.drop(columns=60, axis=1)
Y = data[60]

# =========================
# TRAIN TEST SPLIT
# =========================
split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(X, Y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]

# =========================
# PREPROCESSING
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# =========================
# TRAINING ACCURACY
# =========================
train_pred = model.predict(X_train)

train_accuracy = accuracy_score(Y_train, train_pred)

print("\nTraining Accuracy:")
print(train_accuracy)

# =========================
# TESTING ACCURACY
# =========================
test_pred = model.predict(X_test)

test_accuracy = accuracy_score(Y_test, test_pred)

print("\nTesting Accuracy:")
print(test_accuracy)

# =========================
# CONFUSION MATRIX
# =========================
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, test_pred))

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:")
print(classification_report(Y_test, test_pred))

# =========================
# PREDICTION SYSTEM
# =========================
input_data = (
    0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,
    0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,
    0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,
    0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,
    0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,
    0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,
    0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,
    0.0180,0.0084,0.0090,0.0032
)

input_df = pd.DataFrame([input_data])

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)

print("\nPrediction:", prediction[0])

if prediction[0] == "R":
    print("Object is Rock")
else:
    print("Object is Mine")