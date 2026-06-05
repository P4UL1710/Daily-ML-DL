# 🏠 House Price Prediction System

A Machine Learning project that predicts California house prices using various regression algorithms and selects the best-performing model based on evaluation metrics.

---

## 📌 Project Overview

This project uses the California Housing dataset to predict house prices based on features such as:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

The project includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing Pipeline
- Model Training
- Model Evaluation
- Model Comparison
- Model Serialization using Joblib

---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- Joblib

---

## 📂 Project Structure

```text
House-Prediction-System/
│
├── housing.csv
├── main.ipynb
├── main.py
├── house_price_pipeline.pkl
├── output.png
├── Income Cat distribution.png
├── requirements.txt
└── README.md
```

---

## 📊 Data Preprocessing

The following preprocessing steps were applied:

### Missing Value Handling

- Median Imputation for numerical columns

### Feature Scaling

- StandardScaler

### Categorical Encoding

- OneHotEncoder

### Pipeline

- Scikit-Learn Pipeline
- ColumnTransformer

---

## 🤖 Models Trained

### 1. Linear Regression

Used as a baseline model.

### 2. Random Forest Regressor

Ensemble learning algorithm using multiple decision trees.

### 3. XGBoost Regressor

Gradient boosting algorithm that achieved the best performance.

---

## 📈 Model Evaluation Metrics

The models were evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

### Model Comparison

| Model | RMSE | MAE | R² Score |
|---------|---------:|---------:|---------:|
| XGBoost | 47612.81 | 31606.35 | 0.827 |
| Random Forest | 48941.70 | 31628.40 | 0.817 |
| Linear Regression | 70059.19 | 50670.49 | 0.625 |

### Best Model

🏆 **XGBoost Regressor**

- Lowest RMSE
- Lowest MAE
- Highest R² Score

---

## 💾 Saving the Model

The best-performing model is saved using Joblib.

```python
import joblib

joblib.dump(final_pipeline, "house_price_pipeline.pkl")
```

---

## ▶️ Running the Project

### Clone Repository

```bash
git clone <your-repository-url>
cd House-Prediction-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training Script

```bash
python main.py
```

---

## 🔮 Making Predictions

```python
import joblib
import pandas as pd

model = joblib.load("house_price_pipeline.pkl")

sample = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41],
    'total_rooms': [880],
    'total_bedrooms': [129],
    'population': [322],
    'households': [126],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
})

prediction = model.predict(sample)

print(prediction)
```

---

## 📷 Visualizations

The project includes:

- Income Category Distribution
- Correlation Analysis
- Model Comparison
- Actual vs Predicted Results

---

## 🎯 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Streamlit Web Application
- Flask/FastAPI Deployment
- Docker Containerization
- Cloud Deployment

---

## 👨‍💻 Author
Paul
Developed as a Machine Learning Regression Project using the California Housing Dataset.
