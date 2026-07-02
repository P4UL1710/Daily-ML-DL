# 💳 Credit Card Fraud Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Credit card fraud is one of the most significant challenges in the financial industry, where fraudulent transactions can lead to substantial financial losses. In this project, a **Deep Learning-based Artificial Neural Network (ANN)** is developed to classify credit card transactions as **Legitimate** or **Fraudulent**.

The project covers the complete machine learning pipeline, including data preprocessing, handling class imbalance, feature scaling, neural network training, model evaluation, and a predictive system for real-world transaction classification.

---

# 🎯 Objectives

* Analyze credit card transaction data
* Handle highly imbalanced datasets
* Perform feature scaling
* Build an Artificial Neural Network (ANN)
* Train and evaluate the deep learning model
* Detect fraudulent transactions
* Create a predictive system
* Save the trained model for future deployment

---

# 📂 Dataset

The project uses the **Credit Card Fraud Detection Dataset**.

### Features

* Time
* V1 – V28 (PCA Transformed Features)
* Amount

### Target Variable

| Class | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* TensorFlow
* Keras
* Scikit-Learn

---

# ⚙️ Data Preprocessing

### Data Cleaning

* Load dataset
* Inspect missing values
* Analyze class distribution

### Handling Imbalanced Dataset

Since fraudulent transactions represent only a very small percentage of the dataset, **Random Undersampling** is applied to create a balanced training dataset.

### Feature Selection

Removed:

* Time

Retained:

* V1–V28
* Amount

### Feature Scaling

Applied **StandardScaler** to normalize feature values before training the neural network.

---

# 🧠 Deep Learning Architecture

```text
Input Layer (29 Features)
        │
        ▼
Dense Layer (32 Neurons, ReLU)
        │
        ▼
Dense Layer (16 Neurons, ReLU)
        │
        ▼
Output Layer (1 Neuron, Sigmoid)
```

---

# 🤖 Model Configuration

### Hidden Layers

* Dense (32, ReLU)
* Dense (16, ReLU)

### Output Layer

* Dense (1, Sigmoid)

### Compilation

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metric: Accuracy

---

# 📊 Workflow

### 1. Data Collection

Load the credit card transaction dataset.

### 2. Exploratory Data Analysis

* Dataset overview
* Missing value analysis
* Class distribution visualization

### 3. Data Preprocessing

* Handle class imbalance
* Feature selection
* Feature scaling

### 4. Train-Test Split

Use **Stratified Shuffle Split** to preserve class distribution.

### 5. Model Building

Build an Artificial Neural Network using TensorFlow/Keras.

### 6. Model Training

Train the ANN on balanced transaction data.

### 7. Model Evaluation

Evaluate the model using:

* Accuracy Score
* Confusion Matrix
* Classification Report

### 8. Predictive System

Predict whether a custom transaction is **Legitimate** or **Fraudulent**.

### 9. Model Saving

Save the trained model for future deployment.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to the project

```bash
cd Project_26_Credit_Card_Fraud_Detection_DL
```

Install dependencies

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

Run the project

```bash
python main.py
```

---

# 📈 Model Performance

Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report

The trained neural network successfully learns patterns from transaction data to classify legitimate and fraudulent transactions.

---

# 🔮 Predictive System

### Example Input

```text
Transaction Features:
V1 ... V28
Amount = 123.50
```

### Example Output

```text
Prediction : Legitimate Transaction
```

or

```text
Prediction : Fraudulent Transaction
```

---

# 💾 Model Saving

The trained model is saved as:

```text
credit_card_fraud_detection_dl.keras
```

This allows future deployment without retraining the model.

---

# 📁 Project Structure

```text
Project_26_Credit_Card_Fraud_Detection_DL/
│
├── creditcard.csv
├── main.py
├── credit_card_fraud_detection_dl.keras
├── README.md
└── requirements.txt
```

---

# 🔮 Future Improvements

* SMOTE Oversampling
* Class Weight Balancing
* Dropout Regularization
* Batch Normalization
* Hyperparameter Tuning
* Autoencoders for Anomaly Detection
* Streamlit Deployment

---

# 🎓 Key Learnings

* Deep Learning Fundamentals
* Artificial Neural Networks (ANN)
* Binary Classification
* Fraud Detection
* Feature Scaling
* Handling Imbalanced Data
* Stratified Sampling
* Model Evaluation
* TensorFlow & Keras

---

# 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong AI portfolio.

⭐ If you found this project useful, consider giving it a star.
