# 🛒 BigMart Sales Prediction System

A Machine Learning project that predicts the sales of products across different BigMart outlets based on product characteristics and store information.

---

## 📌 Project Overview

The objective of this project is to build a regression model capable of predicting the sales of products in various BigMart stores.

The project involves:

- Data Cleaning
- Missing Value Treatment
- Feature Engineering
- Data Preprocessing using Pipelines
- Categorical Encoding using One-Hot Encoding
- Model Building using XGBoost Regressor
- Model Evaluation using R² Score
- Predictive System for New Data

---

## 📂 Dataset Features

| Feature | Description |
|----------|-------------|
| Item_Identifier | Unique Product ID |
| Item_Weight | Weight of Product |
| Item_Fat_Content | Fat Content of Product |
| Item_Visibility | Visibility Percentage |
| Item_Type | Category of Product |
| Item_MRP | Maximum Retail Price |
| Outlet_Identifier | Unique Outlet ID |
| Outlet_Establishment_Year | Outlet Establishment Year |
| Outlet_Size | Size of Outlet |
| Outlet_Location_Type | Tier of Outlet Location |
| Outlet_Type | Type of Outlet |
| Item_Outlet_Sales | Target Variable |

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- XGBoost

---

## 🔍 Data Preprocessing

### Missing Value Handling

#### Item Weight
- Filled using Mean Imputation

#### Outlet Size
- Filled using Mode based on Outlet Type

### Feature Processing

- Numerical Features → SimpleImputer
- Categorical Features → OneHotEncoder
- Combined using ColumnTransformer

---

## ⚙️ Model Pipeline

```python
Numerical Pipeline
        │
        ▼
SimpleImputer

Categorical Pipeline
        │
        ▼
SimpleImputer
        │
        ▼
OneHotEncoder

        │
        ▼

ColumnTransformer
        │
        ▼

XGBoost Regressor
```

---

## 🚀 Model Training

The dataset was split into:

- 80% Training Data
- 20% Testing Data

Model Used:

```python
XGBRegressor()
```

---

## 📊 Model Performance

| Metric | Score |
|----------|----------|
| Train R² Score | 0.71 |
| Test R² Score | 0.50 |

---

## 🎯 Predictive System

The project includes a predictive system where users can input product and outlet details to estimate expected sales.

Example Input:

```python
[
    'FDL48',
    19.35,
    'Regular',
    0.082601537,
    'Baking Goods',
    50.1034,
    'OUT018',
    2009,
    'Medium',
    'Tier 3',
    'Supermarket Type2'
]
```

Example Output:

```python
Predicted Sales: 735.42
```

---

## 📁 Project Structure

```text
BigMart-Sales-Prediction/
│
├── Train.csv
├── main.py
├── BigMart_Sales_Prediction.ipynb
├── README.md
└── requirements.txt
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/BigMart-Sales-Prediction.git
```

Move into project directory:

```bash
cd BigMart-Sales-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## 📈 Future Improvements

- Hyperparameter Tuning
- Feature Engineering
- Cross Validation
- Model Deployment using Streamlit
- Model Serialization using Joblib
- Performance Optimization

---

## 👨‍💻 Author

**Pawan**

Building Machine Learning & Deep Learning Projects Daily 🚀