# 🏦 Loan Prediction System

A Machine Learning project that predicts whether a loan application will be approved or not based on applicant information such as income, education, credit history, marital status, and other factors.

## 🚀 Project Overview

The goal of this project is to build a classification model that can predict loan approval status using historical loan application data.

This project demonstrates:

* Data Cleaning & Preprocessing
* Handling Missing Values
* Feature Encoding
* Exploratory Data Analysis (EDA)
* Model Training using Support Vector Machine (SVM)
* Model Evaluation
* Loan Approval Prediction

---

## 📊 Dataset Features

The dataset contains the following attributes:

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area
* Loan Status (Target Variable)

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```bash
Loan-Prediction-System/
│
├── Data.csv
├── main.py
├── requirements.txt
├── README.md
└── loan_prediction.ipynb
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to the project directory:

```bash
cd Daily-ML-DL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🧠 Machine Learning Workflow

1. Load Dataset
2. Handle Missing Values
3. Encode Categorical Features
4. Split Data into Training and Testing Sets
5. Train SVM Classifier
6. Evaluate Model Performance
7. Predict Loan Approval Status

---

## 📈 Model Performance

The model is evaluated using Accuracy Score on both training and testing datasets.

Metrics used:

* Training Accuracy
* Testing Accuracy

---

## 🔮 Sample Prediction

Input applicant details:

```python
[0, 1, 1, 0, 0, 4583, 1508.0, 128.0, 360.0, 1.0, 0]
```

Output:

```text
You are eligible for taking a loan.
```

---

## 🎯 Learning Outcomes

Through this project, I learned:

* Data preprocessing techniques
* Feature encoding
* Classification using SVM
* Model evaluation and validation
* Building an end-to-end Machine Learning pipeline

---

## ⭐ Connect With Me

If you found this project useful, feel free to star the repository and connect with me on LinkedIn.

#MachineLearning #DataScience #Python #ScikitLearn #SVM #LoanPrediction
