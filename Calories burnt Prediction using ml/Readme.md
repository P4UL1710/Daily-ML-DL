# Calories Burnt Prediction using XGBoost

This project predicts the number of calories burned during exercise using machine learning. The model is trained using exercise and calorie datasets and utilizes the XGBoost Regressor for high prediction accuracy.

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Correlation heatmap visualization
- Gender encoding
- Train-test data splitting
- XGBoost regression model training
- Model evaluation using R² Score
- Calories burned prediction for custom input values

## Dataset

The project uses two datasets:

- `exercise.csv`
- `calories.csv`

These datasets are merged to create the final training dataset.

### Dataset Columns

| Column | Description |
|----------|-------------|
| User_ID | Unique user identifier |
| Gender | Male/Female |
| Age | Age of user |
| Height | Height (cm) |
| Weight | Weight (kg) |
| Duration | Exercise duration (minutes) |
| Heart_Rate | Average heart rate |
| Body_Temp | Body temperature |
| Calories | Calories burned (Target Variable) |

## Project Structure

```
project/
│
├── main.py
├── exercise.csv
├── calories.csv
├── requirements.txt
└── README.md
```

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd calories-burnt-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Running the Project

```bash
python main.py
```

## Machine Learning Workflow

1. Load datasets
2. Merge exercise and calorie data
3. Perform exploratory data analysis
4. Encode categorical features
5. Generate correlation heatmap
6. Split data into training and testing sets
7. Train XGBoost Regressor
8. Evaluate model performance
9. Predict calories burned for new input data

## Example Prediction

Input:

```python
[0, 27, 154.0, 58.0, 10.0, 81.0, 39.8]
```

Where:

```text
Gender      = Female (0)
Age         = 27
Height      = 154 cm
Weight      = 58 kg
Duration    = 10 min
Heart Rate  = 81 bpm
Body Temp   = 39.8 °C
```

Output:

```text
Predicted Calories Burned: XX.XX
```

## Model Used

### XGBoost Regressor

XGBoost is a gradient boosting framework that provides:

- High prediction accuracy
- Fast training speed
- Regularization to reduce overfitting
- Efficient handling of large datasets

## Evaluation Metric

The model performance is measured using:

### R² Score

```text
R² = 1.0  → Perfect prediction
R² = 0.0  → No predictive power
```

Training and testing scores are displayed after model training.

## Visualization

The project generates:

- Age distribution plot
- Correlation heatmap between features

These visualizations help understand the dataset and feature relationships.

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Model serialization using Pickle/Joblib
- Deployment using Flask or FastAPI
- Interactive web application

## License

This project is intended for educational and learning purposes.