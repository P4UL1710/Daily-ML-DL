# 🌧️ Rainfall Prediction System using Random Forest

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Random Forest](https://img.shields.io/badge/RandomForest-Classifier-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Accurate rainfall prediction plays an important role in agriculture, disaster management, transportation, and weather forecasting.

In this project, a **Random Forest Classifier** is used to predict whether rainfall will occur based on atmospheric and weather-related parameters such as pressure, humidity, cloud cover, sunshine, and wind conditions.

The project includes data preprocessing, imbalance handling, feature selection, hyperparameter tuning, model evaluation, and a predictive system for real-world weather inputs.

---

## 🎯 Objectives

* Analyze weather-related data
* Handle missing values
* Address class imbalance
* Perform feature selection
* Build a Random Forest Classification model
* Optimize performance using GridSearchCV
* Evaluate model performance
* Create a rainfall prediction system

---

## 📂 Dataset

The dataset contains historical weather observations.

### Features

* Pressure
* Dew Point
* Humidity
* Cloud Cover
* Sunshine
* Wind Direction
* Wind Speed
* Maximum Temperature
* Minimum Temperature
* Average Temperature

### Target Variable

| Value | Meaning     |
| ----- | ----------- |
| 0     | No Rainfall |
| 1     | Rainfall    |

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Random Forest
* GridSearchCV
* Pickle

---

## ⚙️ Data Preprocessing

### Data Cleaning

* Removed unnecessary columns
* Standardized column names
* Handled missing values

### Missing Value Treatment

#### Wind Direction

```python id="0bwlrv"
mode()
```

#### Wind Speed

```python id="vt5m1d"
median()
```

---

## 📊 Exploratory Data Analysis

Performed:

✅ Distribution Analysis

✅ Count Plots

✅ Correlation Heatmaps

✅ Boxplots

### Correlation-Based Feature Selection

Removed highly correlated features:

```text id="xvsj95"
maxtemp
temparature
mintemp
```

to reduce redundancy and improve model performance.

---

## ⚖️ Handling Imbalanced Data

The rainfall classes were imbalanced.

Applied:

```python id="sm55d4"
resample()
```

to downsample the majority class and create a balanced dataset.

---

## 🤖 Machine Learning Model

### Random Forest Classifier

The project uses:

```python id="g6m8od"
RandomForestClassifier()
```

for classification.

### Hyperparameter Tuning

Used:

```python id="77kj2f"
GridSearchCV()
```

to find optimal values for:

* n_estimators
* max_depth
* max_features
* min_samples_split
* min_samples_leaf

---

## 📊 Workflow

### 1. Data Collection

Load rainfall dataset.

### 2. Data Cleaning

* Remove irrelevant columns
* Handle missing values

### 3. Exploratory Data Analysis

* Histograms
* Countplots
* Correlation Heatmaps
* Boxplots

### 4. Feature Engineering

* Remove highly correlated features
* Balance dataset

### 5. Train-Test Split

Use Stratified Sampling.

### 6. Hyperparameter Optimization

Apply GridSearchCV.

### 7. Model Training

Train optimized Random Forest model.

### 8. Model Evaluation

Evaluate using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* Cross Validation Score

### 9. Predictive System

Predict rainfall occurrence using custom weather conditions.

### 10. Model Serialization

Save trained model using Pickle.

---

## 🚀 Installation

Clone the repository:

```bash id="4jbh15"
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to project folder:

```bash id="9qlzk5"
cd Project_23_Rainfall_Prediction
```

Install dependencies:

```bash id="kiklb6"
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the project:

```bash id="7mqm84"
python main.py
```

---

## 📈 Model Performance

Evaluation Metrics:

* Accuracy Score
* Cross Validation Score
* Confusion Matrix
* Classification Report

The optimized Random Forest model achieves strong performance in predicting rainfall events based on atmospheric conditions.

---

## 🔮 Predictive System

Example Input:

```text id="u4pxr7"
Pressure       : 1015.9
Dew Point      : 19.9
Humidity       : 95
Cloud Cover    : 81
Sunshine       : 0.0
Wind Direction : 40.0
Wind Speed     : 13.7
```

Example Output:

```text id="ut2a6k"
Prediction Result: Rainfall
```

or

```text id="ik5oao"
Prediction Result: No Rainfall
```

---

## 💾 Model Saving

The trained model is saved using:

```python id="vk0snw"
pickle.dump()
```

Output File:

```text id="jknj1w"
rainfall_prediction_model.pkl
```

This allows future deployment without retraining.

---

## 📁 Project Structure

```text id="50j6mn"
Project_23_Rainfall_Prediction/
│
├── Rainfall.csv
├── main.py
├── rainfall_prediction_model.pkl
├── README.md
│
└── requirements.txt
```

---

## 🔮 Future Improvements

* XGBoost Classifier
* LightGBM
* Weather API Integration
* Real-Time Rainfall Forecasting
* Streamlit Deployment
* Feature Importance Analysis
* Ensemble Methods

---

## 🎓 Key Learnings

* Classification Problems
* Random Forest Classifier
* Hyperparameter Tuning
* GridSearchCV
* Handling Imbalanced Data
* Feature Selection
* Cross Validation
* Model Serialization

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
