# 🚢 Titanic Survival Prediction using Machine Learning

A Machine Learning project that predicts whether a passenger survived the Titanic disaster based on features such as age, gender, passenger class, fare, and family information.

---

## 📌 Project Overview

The Titanic Survival Prediction project is a binary classification problem where the goal is to predict passenger survival using historical Titanic passenger data.

This project covers:

- Data Loading and Exploration
- Data Cleaning and Preprocessing
- Handling Missing Values
- Feature Encoding
- Data Visualization
- Model Training using Logistic Regression
- Model Evaluation
- Predictive System for New Passenger Data

---

## 📂 Dataset

The dataset contains information about Titanic passengers, including:

| Feature | Description |
|----------|-------------|
| PassengerId | Unique passenger ID |
| Pclass | Ticket Class |
| Name | Passenger Name |
| Sex | Gender |
| Age | Age of Passenger |
| SibSp | Number of Siblings/Spouses aboard |
| Parch | Number of Parents/Children aboard |
| Ticket | Ticket Number |
| Fare | Ticket Fare |
| Cabin | Cabin Number |
| Embarked | Port of Embarkation |
| Survived | Survival Status (Target Variable) |

### Target Variable

- 0 → Did Not Survive
- 1 → Survived

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📊 Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns:
  - PassengerId
  - Name
  - Ticket
  - Cabin

- Handled Missing Values:
  - Age → Filled using Median
  - Embarked → Filled using Mode

- Encoded Categorical Features:
  - Male → 0
  - Female → 1
  - S → 0
  - C → 1
  - Q → 2

---

## 📈 Exploratory Data Analysis

Visualizations include:

- Survival Distribution
- Gender Distribution
- Correlation Analysis
- Feature Relationships

---

## 🤖 Machine Learning Model

### Logistic Regression

The Logistic Regression algorithm was used for classification because:

- Simple and efficient
- Works well for binary classification
- Easy to interpret

---

## 🚀 Model Training

The dataset was split into:

- Training Data: 80%
- Testing Data: 20%

Using:

```python
StratifiedShuffleSplit()
```

to maintain class balance.

---

## 📋 Model Evaluation

Performance was evaluated using:

- Accuracy Score

Example Output:

```text
Training Accuracy : 80%+
Testing Accuracy : 78%+
```

*(Results may vary slightly depending on preprocessing and random state.)*

---

## 🔮 Predictive System

The project includes a prediction system where custom passenger details can be entered to determine whether the passenger is likely to survive.

Example:

```python
input_data = [3, 0, 22, 1, 0, 7.25, 0]
```

Output:

```text
Passenger Survived
```

or

```text
Passenger Did Not Survive
```

---

## 📁 Project Structure

```
Titanic Survival Prediction using ml/
│
├── train.csv
├── main.ipynb
├── main.py
└── README.md
```

---

## 🎯 Key Learning Outcomes

- Data Cleaning
- Missing Value Treatment
- Feature Engineering
- Data Visualization
- Logistic Regression
- Model Evaluation
- Building Predictive Systems

---

## 📚 Future Improvements

- Random Forest Classifier
- XGBoost Classifier
- Hyperparameter Tuning
- Feature Engineering
- Web App Deployment using Streamlit

---

## ⭐ Project Status

✅ Completed

Part of my **Daily ML & Deep Learning Projects** repository where I build machine learning projects consistently to strengthen practical skills and create a strong portfolio.

---