# Fake News Detection Using Machine Learning

## Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify news articles as Fake or Real.

The model is trained using the Fake News and True News datasets and utilizes:

- Text Preprocessing
- Porter Stemming
- TF-IDF Vectorization
- Logistic Regression

The system can take a news article as input and predict whether it is likely Fake or Real.

---

## Dataset

Dataset consists of two files:

- Fake.csv
- True.csv

After combining:

- Total Samples: 44,898

Target Labels:

- 0 → Fake News
- 1 → Real News

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn

---

## Machine Learning Pipeline

1. Load Dataset
2. Merge Fake and True News
3. Text Cleaning
4. Stopword Removal
5. Porter Stemming
6. TF-IDF Vectorization
7. Train-Test Split
8. Logistic Regression
9. Model Evaluation
10. News Prediction

---

## Features

- Automatic preprocessing
- TF-IDF feature extraction
- Logistic Regression classifier
- Interactive prediction system
- Confidence score output

---

## Installation

Install dependencies:

```bash
pip install pandas numpy nltk scikit-learn
```

---

## Project Structure

```text
Fake-News-Detection/
│
├── Fake.csv
├── True.csv
├── main.py
├── README.md
```

---

## Run the Project

```bash
python main.py
```

---

## Example

Input:

```text
Scientists have discovered a new renewable energy source capable of generating electricity with zero emissions.
```

Output:

```text
The news is likely REAL (97.82% confidence)
```

---

## Model Performance

Accuracy achieved on test data:

```text
99.44%
```

---

## Future Improvements

- Streamlit Web Application
- Flask API
- BERT-Based Fake News Detection
- Model Deployment
- Explainable AI Integration

---

## Author

Pawan

Machine Learning & NLP Project