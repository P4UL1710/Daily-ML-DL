# 🍷 Wine Quality Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a wine is of good quality or bad quality using Machine Learning.

The model is trained on the Wine Quality Dataset and uses a Random Forest Classifier to classify wines based on their physicochemical properties.

---

## 📊 Dataset Information

The dataset contains various chemical properties of red wine.

### Features

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

### Target Variable

Wine Quality

Converted into Binary Classification:

- Good Quality Wine = 1 (Quality ≥ 7)
- Bad Quality Wine = 0 (Quality < 7)

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 🚀 Workflow

### 1. Data Collection
- Loaded the Wine Quality Dataset.

### 2. Exploratory Data Analysis (EDA)
- Dataset inspection
- Distribution analysis
- Correlation heatmap
- Feature relationship visualization

### 3. Data Preprocessing
- Feature and target separation
- Binary target creation

### 4. Train-Test Split
- StratifiedShuffleSplit
- 80% Training Data
- 20% Testing Data

### 5. Model Training
- Random Forest Classifier

### 6. Model Evaluation
- Training Accuracy
- Testing Accuracy

### 7. Prediction System
- Predicts whether a wine is good or bad based on input chemical properties.

---

## 📈 Results

The Random Forest Classifier achieved strong performance on the Wine Quality dataset.

Evaluation Metric:
- Accuracy Score

---

## 📂 Project Structure

```text
Wine-Quality-Prediction
│
├── winequality-red.csv
├── main.py
├── main.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Wine-Quality-Prediction.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python main.py
```

---

## 🎯 Key Learnings

- Data preprocessing techniques
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Stratified sampling
- Random Forest Classifier
- Model evaluation using Accuracy Score

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Compare multiple ML algorithms
- Feature importance analysis
- Build a Streamlit web application
- Deploy the model online

---

## 👨‍💻 Author

**Pawan**

Documenting my Machine Learning and Deep Learning journey by building projects consistently and sharing my learnings publicly.

⭐ Feel free to explore the repository and provide feedback.