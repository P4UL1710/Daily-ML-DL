# 🚗 Car Price Prediction System

A Machine Learning project that predicts the selling price of a used car based on various features such as manufacturing year, present price, kilometers driven, fuel type, seller type, transmission type, and ownership history.

## 📌 Project Overview

The objective of this project is to build a regression model capable of estimating the market price of a used car. This can help buyers and sellers make informed decisions based on historical vehicle data.

## 🎯 Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Categorical Feature Encoding
* Feature Selection
* Train-Test Split
* Linear Regression Model Training
* Model Evaluation using R² Score
* Price Prediction for New Inputs
* Data Visualization using Matplotlib and Seaborn

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

## 📂 Dataset Features

| Feature       | Description                     |
| ------------- | ------------------------------- |
| Car_Name      | Name of the Car                 |
| Year          | Manufacturing Year              |
| Selling_Price | Selling Price (Target Variable) |
| Present_Price | Current Ex-showroom Price       |
| Kms_Driven    | Total Kilometers Driven         |
| Fuel_Type     | Petrol / Diesel / CNG           |
| Seller_Type   | Dealer / Individual             |
| Transmission  | Manual / Automatic              |
| Owner         | Number of Previous Owners       |

## ⚙️ Project Workflow

1. Load and inspect the dataset.
2. Perform data preprocessing and handle categorical variables.
3. Split the dataset into training and testing sets.
4. Train a Linear Regression model.
5. Evaluate model performance using R² Score.
6. Visualize Actual vs Predicted Prices.
7. Predict selling prices for new car data.

## 📊 Model Performance

The model is evaluated using:

* R² Score on Training Data
* R² Score on Testing Data

Higher R² values indicate better predictive performance.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/car-price-prediction.git
```

Navigate to the project folder:

```bash
cd car-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## 📸 Sample Output

```text
Training R² Score: 0.88
Testing R² Score: 0.84

Predicted Selling Price: ₹4,75,000
```

## 📈 Future Improvements

* Random Forest Regressor
* XGBoost Regressor
* Hyperparameter Tuning
* Model Deployment using Streamlit
* Real-time Car Price Prediction Web App

## 👨‍💻 Author

Built as part of my Daily Machine Learning & Deep Learning Project Challenge.

If you found this project useful, feel free to ⭐ the repository.
