# 🩺 Diabetes Prediction System Using Machine Learning

## 📌 Project Overview

This project predicts whether a person is diabetic or non-diabetic using Machine Learning techniques. The model is trained on the Pima Indians Diabetes Dataset and uses a Support Vector Machine (SVM) classifier for prediction.

The project covers the complete Machine Learning workflow, including data preprocessing, feature scaling, train-test splitting, model training, evaluation, and prediction.

---

## 📊 Dataset Information

The dataset contains medical information about patients and the target variable `Outcome`.

### Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target Variable

- **0** → Non-Diabetic
- **1** → Diabetic

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Support Vector Machine (SVM)

---

## 🚀 Machine Learning Workflow

### 1. Data Collection
- Loaded the dataset using Pandas.

### 2. Data Preprocessing
- Checked dataset information.
- Separated features and target variable.
- Standardized features using StandardScaler.

### 3. Train-Test Split
- Used StratifiedShuffleSplit to preserve class distribution.
- Training Data: 80%
- Testing Data: 20%

### 4. Model Training
- Trained a Support Vector Machine (SVM) classifier with a linear kernel.

### 5. Model Evaluation
- Calculated training and testing accuracy using Accuracy Score.

### 6. Prediction System
- Built a predictive system that classifies a patient as diabetic or non-diabetic based on input values.

---

## 📂 Project Structure

```text
Diabetes-Prediction-System-Using-ML
│
├── diabetes.csv
├── main.py
├── main.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Diabetes-Prediction-System-Using-ML.git
```

### Navigate to Project Directory

```bash
cd Diabetes-Prediction-System-Using-ML
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

## 📈 Results

The model is evaluated using Accuracy Score on both training and testing datasets.

Example Output:

```text
Training Accuracy: 0.78

Testing Accuracy: 0.77

Prediction Result:
The person is Non-Diabetic
```

---

## 🎯 Key Learnings

- Data preprocessing improves model performance.
- Feature scaling is important for SVM models.
- Stratified splitting helps maintain balanced datasets.
- SVM is effective for binary classification tasks.

---

## 🔮 Future Improvements

- Perform detailed Exploratory Data Analysis (EDA).
- Compare multiple Machine Learning algorithms.
- Hyperparameter tuning.
- Build a Streamlit web application.
- Deploy the model online.

---

## 👨‍💻 Author

**Pawan**

Currently building and sharing Machine Learning & Deep Learning projects daily to strengthen practical AI skills.

---

⭐ If you found this project useful, consider giving it a star on GitHub!