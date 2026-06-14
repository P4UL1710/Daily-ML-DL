# 🧠 Parkinson's Disease Prediction System

A Machine Learning project that predicts whether a person is affected by Parkinson's Disease using biomedical voice measurements. The model is trained using Support Vector Machine (SVM) and achieves high classification accuracy after data preprocessing and feature scaling.

---

## 📌 Project Overview

Parkinson's Disease is a progressive neurological disorder that affects movement and speech. Early detection can help in timely medical intervention.

This project uses machine learning techniques to analyze voice measurement features and predict whether a person has Parkinson's Disease.

---

## 🚀 Features

* Data Preprocessing and Cleaning
* Exploratory Data Analysis (EDA)
* Feature Scaling using StandardScaler
* Stratified Train-Test Splitting
* Support Vector Machine (SVM) Classifier
* Model Evaluation using Accuracy Score
* Predictive System for New Patient Data

---

## 📂 Dataset

The dataset contains biomedical voice measurements from individuals with and without Parkinson's Disease.

### Target Variable

| Column | Description                          |
| ------ | ------------------------------------ |
| status | 0 = Healthy, 1 = Parkinson's Disease |

### Sample Features

* MDVP:Fo(Hz)
* MDVP:Fhi(Hz)
* MDVP:Flo(Hz)
* Jitter Measurements
* Shimmer Measurements
* NHR
* HNR
* RPDE
* DFA
* PPE

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Support Vector Machine (SVM)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/parkinsons-disease-prediction.git
```

Move into the project directory:

```bash
cd parkinsons-disease-prediction
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

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Selection
4. Train-Test Split
5. Feature Scaling
6. Model Training using SVM
7. Model Evaluation
8. Prediction on New Data

---

## 📈 Model Performance

The model is evaluated using:

* Training Accuracy
* Testing Accuracy

```python
from sklearn.metrics import accuracy_score
```

---

## 🔮 Example Prediction

```python
prediction = model.predict(input_data_scaled)

if prediction[0] == 0:
    print("The person is Healthy.")
else:
    print("The person has Parkinson's Disease.")
```

---

## 📁 Project Structure

```text
Parkinsons-Disease-Prediction/
│
├── parkinsons.csv
├── main.py
├── README.md
└── requirements.txt
```

---

## 🎯 Future Improvements

* Hyperparameter Tuning
* Model Deployment using Streamlit
* Flask/FastAPI Integration
* Cross-Validation
* Feature Importance Analysis

---

## 👨‍💻 Author

Pawan

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong AI portfolio.

⭐ If you found this project useful, consider giving it a star.
