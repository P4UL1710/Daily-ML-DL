# 🩺 Breast Cancer Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection plays a crucial role in improving survival rates and treatment effectiveness.

This project uses **Machine Learning** to classify tumors as **Benign (Non-Cancerous)** or **Malignant (Cancerous)** based on diagnostic measurements from the Breast Cancer Wisconsin Dataset.

---

## 🎯 Objectives

* Perform data cleaning and preprocessing
* Convert categorical labels into numerical values
* Split data using Stratified Sampling
* Train a Logistic Regression model
* Evaluate model performance
* Build a predictive system for new patient data

---

## 📂 Dataset

**Dataset:** Breast Cancer Wisconsin Dataset

### Features Include:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

### Target Variable:

* **0 → Benign**
* **1 → Malignant**

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn

---

## 📊 Workflow

### 1. Data Collection

Load the breast cancer dataset into a Pandas DataFrame.

### 2. Data Cleaning

* Remove unnecessary columns
* Handle categorical values

### 3. Data Preprocessing

* Separate features and target variable
* Convert diagnosis labels:

  * B → 0
  * M → 1

### 4. Train-Test Split

Use **StratifiedShuffleSplit** to maintain class distribution.

### 5. Model Training

Train a **Logistic Regression** classifier.

### 6. Model Evaluation

Calculate:

* Training Accuracy
* Testing Accuracy

### 7. Predictive System

Provide tumor measurements and predict whether the tumor is:

* Benign
* Malignant

---

## 🚀 Installation

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git

cd Daily-ML-DL
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

Run the project:

```bash
python main.py
```

---

## 📈 Results

The Logistic Regression model achieves high classification accuracy on both training and testing datasets, making it effective for breast cancer diagnosis prediction.

---

## 📁 Project Structure

```text
Breast-Cancer-Prediction/
│
├── data.csv
├── main.py
├── README.md
│
└── requirements.txt
```

---

## 🔮 Future Improvements

* Add Feature Scaling
* Hyperparameter Tuning
* Deploy using Streamlit
* Create a Web-Based Prediction System
* Compare with Random Forest and XGBoost

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning & Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
