# 🏥 Insurance Cost Prediction using Machine Learning

This project predicts medical insurance charges based on a person's demographic and health-related information using Machine Learning.

## 📌 Project Overview

The objective of this project is to build a regression model that can estimate insurance charges based on factors such as:

* Age
* Gender
* BMI (Body Mass Index)
* Number of Children
* Smoking Status
* Region

The project includes data preprocessing, exploratory data analysis, model training, evaluation, and a predictive system for real-world inputs.

---

## 📊 Dataset

The dataset contains the following features:

| Feature  | Description                              |
| -------- | ---------------------------------------- |
| age      | Age of the insured person                |
| sex      | Gender (Male/Female)                     |
| bmi      | Body Mass Index                          |
| children | Number of dependents covered             |
| smoker   | Smoking status                           |
| region   | Residential region                       |
| charges  | Medical insurance cost (Target Variable) |

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

---

## 🔍 Project Workflow

### 1. Data Collection

* Load the insurance dataset using Pandas.

### 2. Data Preprocessing

* Check missing values.
* Encode categorical variables.
* Prepare data for model training.

### 3. Exploratory Data Analysis (EDA)

* Distribution plots
* Correlation analysis
* Feature understanding

### 4. Model Training

* Split data into training and testing sets.
* Train a Linear Regression model.

### 5. Model Evaluation

* Evaluate performance using R² Score.

### 6. Predictive System

* Accept user input.
* Process and encode data.
* Predict insurance charges.

---

## 📈 Model Used

### Linear Regression

Linear Regression is used to predict continuous insurance charges based on input features.

---

## 🚀 How to Run

### Clone the Repository

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
cd Daily-ML-DL
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python main.py
```

---

## 🧪 Sample Prediction

Input:

```python
[37, "female", 27.74, 3, "no", "northwest"]
```

Output:

```text
Predicted Insurance Charge: ₹XXXXX.XX
```

---

## 📂 Project Structure

```text
Insurance-Cost-Prediction/
│
├── insurance.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## 🎯 Results

* Successfully trained a Machine Learning model for insurance charge prediction.
* Built a predictive system for real-time user inputs.
* Achieved reliable performance using Linear Regression.

---

## 📚 Learning Outcomes

* Data Preprocessing
* Feature Encoding
* Exploratory Data Analysis
* Regression Modeling
* Model Evaluation
* Building Predictive Systems

---

## 👨‍💻 Author

Pawan

Building Machine Learning & Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving the repository a star!
