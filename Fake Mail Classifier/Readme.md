# 📧 Fake Mail Classifier using Machine Learning

A Machine Learning project that classifies emails as **Spam** or **Ham (Legitimate)** using **Natural Language Processing (NLP)** and **Logistic Regression**.

This project converts email text into numerical features using **TF-IDF Vectorization** and then trains a classification model to detect unwanted spam emails.

---

## 🚀 Project Overview

Email spam detection is one of the most common applications of Machine Learning and NLP.

In this project:

- Email messages are preprocessed
- Text data is converted into numerical vectors using TF-IDF
- A Logistic Regression model is trained on labeled email data
- The model predicts whether a new email is Spam or Ham

---

## 📂 Dataset

The dataset contains two columns:

| Column | Description |
|----------|-------------|
| Category | Spam or Ham |
| Message | Email Text |

Example:

| Category | Message |
|-----------|----------|
| ham | I'm reaching home soon |
| spam | Congratulations! You've won a prize |

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 🔍 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Handling Missing Values
4. Label Encoding
5. Train-Test Split
6. Feature Extraction using TF-IDF
7. Model Training using Logistic Regression
8. Model Evaluation
9. Predictive System

---

## 📊 Feature Extraction

Text data cannot be directly used by Machine Learning models.

TF-IDF Vectorization converts text into numerical feature vectors while preserving the importance of words.

```python
vectorizer = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)
```

---

## 🤖 Model Used

### Logistic Regression

Logistic Regression is a supervised classification algorithm widely used for binary classification tasks such as:

- Spam Detection
- Sentiment Analysis
- Disease Prediction
- Fraud Detection

---

## 📈 Model Performance

The model achieves high accuracy on both training and testing datasets.

Metrics used:

- Accuracy Score
- Prediction System

---

## 💻 Running the Project

### Clone Repository

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

### Navigate to Project Folder

```bash
cd "Fake Mail Classifier"
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

### Run Project

```bash
python main.py
```

---

## 📌 Example Prediction

Input:

```text
Congratulations! You have won ₹50,000. Click here to claim.
```

Output:

```text
📩 Spam Email
```

Input:

```text
Hey, let's meet tomorrow at 5 PM.
```

Output:

```text
✅ Ham (Legitimate) Email
```

---

## 📁 Project Structure

```text
Fake Mail Classifier/
│
├── mail_data.csv
├── main.py
└── README.md
```

---

## 🎯 Key Learning Outcomes

- Natural Language Processing Basics
- Text Preprocessing
- TF-IDF Vectorization
- Logistic Regression
- Spam Detection Systems
- Machine Learning Pipeline Development

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily as part of the **Daily-ML-DL** challenge.

⭐ If you found this project useful, consider giving it a star.