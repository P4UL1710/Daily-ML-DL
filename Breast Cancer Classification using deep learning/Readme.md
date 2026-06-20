# 🧠 Breast Cancer Classification using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Breast cancer is one of the most common forms of cancer worldwide, and early diagnosis can significantly improve treatment outcomes.

In this project, an **Artificial Neural Network (ANN)** is built using **TensorFlow and Keras** to classify breast tumors as **Benign** or **Malignant** based on diagnostic measurements from the Breast Cancer Wisconsin Dataset.

The project demonstrates the complete Deep Learning workflow, including data preprocessing, feature scaling, model training, evaluation, visualization, and prediction on new patient data.

---

## 🎯 Objectives

* Perform data cleaning and preprocessing
* Apply feature scaling using StandardScaler
* Build an Artificial Neural Network (ANN)
* Train the model using TensorFlow/Keras
* Evaluate classification performance
* Visualize training and validation metrics
* Create a predictive system for new patient data

---

## 📂 Dataset

**Dataset:** Breast Cancer Wisconsin Dataset

### Features Include

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

### Target Classes

| Class | Description |
| ----- | ----------- |
| 0     | Benign      |
| 1     | Malignant   |

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* TensorFlow
* Keras

---

## 🏗️ Deep Learning Architecture

```text
Input Layer (30 Features)
        │
        ▼
Dense Layer (20 Neurons, ReLU)
        │
        ▼
Output Layer (2 Neurons, Sigmoid)
```

### Model Configuration

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Evaluation Metric: Accuracy
* Epochs: 10

---

## 📊 Workflow

### 1. Data Collection

Load the Breast Cancer dataset.

### 2. Data Cleaning

* Remove unnecessary columns
* Encode diagnosis labels

### 3. Data Preprocessing

* Split features and target
* Perform Stratified Train-Test Split

### 4. Feature Scaling

* StandardScaler normalization

### 5. Model Building

* Build ANN using TensorFlow/Keras
* Add Dense layers

### 6. Model Training

* Train using training dataset
* Validate using validation split

### 7. Performance Visualization

* Accuracy vs Epochs
* Loss vs Epochs

### 8. Model Evaluation

* Test dataset accuracy

### 9. Predictive System

* Predict tumor class for custom patient data

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to project folder:

```bash
cd Project_20_Breast_Cancer_ANN
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Run the project:

```bash
python main.py
```

---

## 📈 Model Performance

The Artificial Neural Network successfully learns patterns from breast cancer diagnostic data and achieves high classification accuracy on unseen test samples.

The training process is visualized using:

* Accuracy Curves
* Validation Accuracy Curves
* Training Loss Curves
* Validation Loss Curves

---

## 📁 Project Structure

```text
Project_20_Breast_Cancer_ANN/
│
├── data.csv
├── main.py
├── README.md
│
└── requirements.txt
```

---

## 🔮 Future Improvements

* Increase network depth
* Add Dropout layers
* Hyperparameter tuning
* Cross-validation
* Compare ANN with:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Deploy using Streamlit

---

## 🎓 Key Learnings

* Neural Network Fundamentals
* TensorFlow & Keras Workflow
* Feature Scaling Importance
* ANN Architecture Design
* Model Evaluation Techniques
* Binary Classification using Deep Learning

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
