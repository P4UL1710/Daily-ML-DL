# 💳 Credit Card Fraud Detection System

A Machine Learning project that detects fraudulent credit card transactions using Logistic Regression. The model is trained on historical transaction data and classifies transactions as either legitimate or fraudulent.

---

## 🚀 Project Overview

Credit card fraud is a major concern in the financial industry. This project uses Machine Learning techniques to identify fraudulent transactions and help improve transaction security.

The dataset contains transactions made by credit card holders, where:

* **0** → Legitimate Transaction
* **1** → Fraudulent Transaction

Due to the highly imbalanced nature of the dataset, undersampling is used to create a balanced dataset before model training.

---

## 📊 Features

✅ Data Loading and Exploration

✅ Data Preprocessing

✅ Handling Imbalanced Data using Undersampling

✅ Feature and Target Separation

✅ Stratified Train-Test Split

✅ Logistic Regression Model Training

✅ Model Evaluation using Accuracy Score

✅ Fraud Prediction System

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn

---

## 📂 Project Structure

```bash
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Move to the project directory:

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

## 📈 Machine Learning Workflow

1. Load the Dataset
2. Analyze Class Distribution
3. Handle Class Imbalance
4. Create Balanced Dataset
5. Split Data into Training and Testing Sets
6. Train Logistic Regression Model
7. Evaluate Model Performance
8. Predict Fraudulent Transactions

---

## 📊 Model Used

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification problems. In this project, it predicts whether a transaction is:

* Legitimate (0)
* Fraudulent (1)

---

## 🎯 Results

The model successfully classifies credit card transactions as legitimate or fraudulent based on transaction features.

Performance is evaluated using:

* Training Accuracy
* Testing Accuracy

---

## 🔮 Future Improvements

* Random Forest Classifier
* XGBoost Classifier
* SMOTE for Class Balancing
* Hyperparameter Tuning
* Deployment using Flask or Streamlit

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

### ⭐ If you found this project useful, consider giving it a star!
