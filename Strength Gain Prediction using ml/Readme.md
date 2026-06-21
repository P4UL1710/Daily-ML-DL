# 💪 Strength Gain Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Predicting workout results is a challenging task because strength gains depend on multiple factors such as age, gender, supplement usage, body weight, and training consistency.

In this project, an **XGBoost Regressor** is used to predict an individual's expected **Strength Gain (%)** based on supplement usage and fitness-related attributes.

The model learns patterns from historical supplement performance data and estimates future strength improvements for new users.

---

## 🎯 Objectives

* Perform data preprocessing and cleaning
* Encode categorical features
* Build a regression model using XGBoost
* Evaluate model performance using R² Score
* Predict expected strength gain percentage
* Create a predictive system for custom user inputs

---

## 📂 Dataset

The dataset contains information about supplement users and their fitness progress.

### Features

* Age
* Gender
* Supplement Type
* Duration of Usage
* Initial Weight
* Final Weight
* Primary Benefit

### Target Variable

* **Strength_Gain (%)**

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* XGBoost

---

## ⚙️ Data Preprocessing

### Categorical Encoding

#### Gender

| Category   | Encoded Value |
| ---------- | ------------- |
| Non-Binary | 0             |
| Male       | 1             |
| Female     | 2             |

#### Supplement

| Category             | Encoded Value |
| -------------------- | ------------- |
| Both                 | 0             |
| Mass Gainer          | 1             |
| Creatine Monohydrate | 2             |

#### Primary Benefit

Encoded using:

```python id="pq7zhk"
LabelEncoder()
```

### Target Transformation

The target column originally contains percentage values:

```text id="t6v95i"
25%
18%
32%
```

Converted into numerical values:

```python id="jj48wl"
25
18
32
```

for regression modeling.

---

## 🤖 Machine Learning Model

### XGBoost Regressor

The project uses:

```python id="ukwcyx"
XGBRegressor()
```

to learn the relationship between user characteristics and strength gain percentage.

---

## 📊 Workflow

### 1. Data Collection

Load supplement performance dataset.

### 2. Data Cleaning

* Remove unnecessary columns
* Handle percentage values

### 3. Feature Engineering

* Encode categorical variables
* Prepare target variable

### 4. Train-Test Split

Split dataset into:

* 80% Training Data
* 20% Testing Data

### 5. Model Training

Train XGBoost Regressor.

### 6. Model Evaluation

Evaluate performance using:

* R² Score

### 7. Predictive System

Accept user inputs and predict expected strength gain percentage.

---

## 🚀 Installation

Clone the repository:

```bash id="g6mzvz"
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to project folder:

```bash id="8m0e3v"
cd Project_21_Strength_Gain_Prediction
```

Install dependencies:

```bash id="6p1tyq"
pip install numpy pandas scikit-learn xgboost
```

Run the project:

```bash id="4r3zk4"
python main.py
```

---

## 📈 Model Performance

The model is evaluated using the **R² Score**, which measures how well the predicted strength gain values match the actual values.

```text id="dnw3mr"
Training R² Score : High
Testing R² Score  : Generalization Performance
```

A higher R² score indicates better predictive capability.

---

## 📁 Project Structure

```text id="dnj6xt"
Project_21_Strength_Gain_Prediction/
│
├── supplement_impact_data.csv
├── main.py
├── README.md
│
└── requirements.txt
```

---

## 🔮 Future Improvements

* Hyperparameter Tuning
* Feature Scaling Experiments
* Cross Validation
* Compare Multiple Regression Models
* Random Forest Regressor
* Linear Regression
* LightGBM
* CatBoost
* Streamlit Deployment

---

## 🎓 Key Learnings

* Regression Problems
* XGBoost Regressor
* Label Encoding
* Data Preprocessing
* Feature Engineering
* Model Evaluation using R² Score
* Building Predictive Systems

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
