# 📰 Fake News Detection using Deep Learning (LSTM)

## 📌 Project Overview

Fake news has become a major challenge in the digital age, spreading misinformation across social media and news platforms.

In this project, a **Deep Learning-based Fake News Detection System** is developed using **Word Embeddings** and **Long Short-Term Memory (LSTM)** networks to classify news articles as **Fake** or **Real**.

The model learns semantic relationships between words and captures contextual information from news headlines to make accurate predictions.

---

## 🎯 Objectives

* Perform text preprocessing
* Apply Natural Language Processing (NLP) techniques
* Generate word embeddings
* Build an LSTM-based Deep Learning model
* Classify news articles as Fake or Real
* Evaluate model performance
* Create a predictive system for custom news headlines

---

## 📂 Dataset

The dataset contains news headlines/articles labeled as:

| Label | Meaning   |
| ----- | --------- |
| 0     | Real News |
| 1     | Fake News |

### Features

* News Title
* News Content (if available)

### Target

* Fake News Classification

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow
* Keras
* NLTK
* Scikit-Learn

---

## ⚙️ NLP Pipeline

### Text Cleaning

* Remove special characters
* Convert text to lowercase

### Tokenization

Convert text into individual words.

### Stopword Removal

Remove frequently occurring words that add little meaning.

Example:

```text
the
is
was
and
```

### Stemming

Reduce words to their root form.

Example:

```text
running → run
played → play
studies → studi
```

### One-Hot Encoding

Convert words into integer representations.

### Padding

Ensure all sequences have the same length.

### Word Embedding

Transform words into dense vector representations using:

```python
Embedding()
```

---

## 🧠 Deep Learning Architecture

```text
Input Text
     │
     ▼
One-Hot Encoding
     │
     ▼
Padding
     │
     ▼
Embedding Layer
     │
     ▼
LSTM Layer (100 Units)
     │
     ▼
Dense Layer
     │
     ▼
Sigmoid Output
```

---

## 🤖 Model Configuration

### Embedding Layer

Learns semantic word representations.

### LSTM Layer

Captures sequence and contextual information from text.

### Output Layer

Binary classification:

```text
0 → Real News
1 → Fake News
```

### Compilation

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metric: Accuracy

---

## 📊 Workflow

### 1. Data Collection

Load fake news dataset.

### 2. Text Preprocessing

* Cleaning
* Tokenization
* Stopword Removal
* Stemming

### 3. Feature Engineering

* One-Hot Encoding
* Sequence Padding

### 4. Word Embedding

Convert text into dense vectors.

### 5. Model Building

Build LSTM Neural Network.

### 6. Model Training

Train model on news data.

### 7. Model Evaluation

Evaluate using:

* Accuracy
* Confusion Matrix
* Classification Report

### 8. Predictive System

Predict whether a custom news headline is fake or real.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to project folder:

```bash
cd Project_24_Fake_News_Detection_LSTM
```

Install dependencies:

```bash
pip install numpy pandas tensorflow nltk scikit-learn
```

Run the project:

```bash
python main.py
```

---

## 📈 Model Performance

Evaluation Metrics:

* Accuracy Score
* Confusion Matrix
* Classification Report

The LSTM model successfully learns textual patterns and semantic relationships to classify fake and real news articles.

---

## 🔮 Predictive System

Example Input:

```text
Scientists discover a new treatment that cures cancer completely
```

Example Output:

```text
Prediction: Real News
```

or

```text
Prediction: Fake News
```

---

## 📁 Project Structure

```text
Project_24_Fake_News_Detection_LSTM/
│
├── train.csv
├── main.py
├── README.md
│
└── requirements.txt
```

---

## 🔮 Future Improvements

* Bidirectional LSTM
* GRU Networks
* Pretrained Word2Vec Embeddings
* GloVe Embeddings
* BERT-based Fake News Detection
* Transformer Architectures
* Streamlit Deployment

---

## 🎓 Key Learnings

* Natural Language Processing (NLP)
* Text Preprocessing
* Word Embeddings
* Sequence Modeling
* Long Short-Term Memory (LSTM)
* Binary Text Classification
* Deep Learning for NLP

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
