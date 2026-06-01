# 🚀 Rock vs Mine Prediction using Machine Learning

This project uses the Sonar dataset to classify whether an object detected by sonar signals is a **Rock (R)** or a **Mine (M)**.

## 📌 Project Overview

Sonar signals are bounced off objects underwater and the reflected signals are recorded. Using these 60 numerical features, a Machine Learning model predicts whether the object is:

- 🪨 Rock (R)
- 💣 Mine (M)

This is a Binary Classification problem solved using Logistic Regression.

---

## 📊 Dataset Information

- Dataset: Sonar Dataset
- Total Samples: 208
- Features: 60
- Target Classes:
  - R → Rock
  - M → Mine

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset Shape
- Dataset Information
- Statistical Summary
- Missing Value Analysis
- Class Distribution Analysis

```python
data.shape
data.info()
data.describe()
data.isnull().sum()
data[60].value_counts()
```

---

## ⚙️ Data Preprocessing

### Feature and Target Separation

```python
X = data.drop(columns=60, axis=1)
Y = data[60]
```

### Train-Test Split

Stratified Sampling was used to maintain class balance.

```python
StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
```

### Feature Scaling

```python
StandardScaler()
```

---

## 🤖 Model Used

### Logistic Regression

```python
model = LogisticRegression(max_iter=1000)
```

Why Logistic Regression?

- Simple and efficient
- Works well on binary classification problems
- Easy to interpret
- Good baseline model

---

## 📈 Model Evaluation

Evaluation Metrics:

- Accuracy Score
- Confusion Matrix
- Classification Report

```python
accuracy_score()
confusion_matrix()
classification_report()
```

### Results

| Metric | Score |
|----------|----------|
| Training Accuracy | ~83.7% |
| Testing Accuracy | ~85.7% |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn

---

## 📂 Project Structure

```
Rock-vs-Mine-Prediction/
│
├── Data.csv
├── rock_vs_mine.py
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### Clone Repository

```bash
git clone <repository-url>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python rock_vs_mine.py
```

---

## 🎯 Future Improvements

- Try Random Forest Classifier
- Try Support Vector Machine (SVM)
- Hyperparameter Tuning
- Cross Validation
- Deploy using Streamlit

---

## 👨‍💻 Author

Pawan

Machine Learning & Deep Learning Enthusiast 🚀

Currently learning:
- Machine Learning
- Deep Learning
- NLP
- Transformers
- MLOps

---

⭐ If you found this project useful, consider giving it a star.