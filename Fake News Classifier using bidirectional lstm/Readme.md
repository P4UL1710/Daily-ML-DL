# 📰 Fake News Detection using Bidirectional LSTM

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![NLP](https://img.shields.io/badge/NLP-Natural_Language_Processing-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Detecting fake news is one of the most important Natural Language Processing (NLP) applications in today's digital world. Unlike a traditional LSTM, a **Bidirectional LSTM (BiLSTM)** processes text sequences in both forward and backward directions, allowing the model to capture richer contextual information.

In this project, a **Bidirectional LSTM** combined with **Word Embeddings** is used to classify news articles as **Fake** or **Real** based on their textual content.

---

# 🎯 Objectives

* Perform text preprocessing
* Clean and normalize textual data
* Apply NLP techniques
* Generate word embeddings
* Build a Bidirectional LSTM model
* Detect fake news articles
* Evaluate classification performance
* Build a predictive system for custom news headlines

---

# 📂 Dataset

The dataset contains labeled news articles.

### Features

* News Title
* News Content (if available)

### Target

| Label | Meaning   |
| ----- | --------- |
| 0     | Real News |
| 1     | Fake News |

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow
* Keras
* NLTK
* Scikit-Learn

---

# ⚙️ NLP Pipeline

### Data Cleaning

* Remove punctuation
* Remove special characters
* Convert text to lowercase

### Stopword Removal

Remove common English stopwords.

### Stemming

Reduce words to their root form.

Example:

```
running → run
studies → studi
playing → play
```

### One-Hot Encoding

Convert words into integer representations.

### Padding

Ensure all text sequences have equal length.

### Word Embedding

Generate dense vector representations using the Keras Embedding layer.

---

# 🧠 Deep Learning Architecture

```
Input Text
      │
      ▼
Text Cleaning
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
Bidirectional LSTM
      │
      ▼
Dense Layer
      │
      ▼
Sigmoid Output
```

---

# 🔄 Why Bidirectional LSTM?

A standard LSTM only processes text from:

```
Left ➜ Right
```

A Bidirectional LSTM processes text in both directions:

```
Left ➜ Right
Right ➜ Left
```

This enables the model to understand context from both previous and upcoming words, often improving performance in NLP classification tasks.

---

# 🤖 Model Configuration

### Embedding Layer

Converts words into dense vector representations.

### Bidirectional LSTM

Learns contextual information from both directions of a sentence.

### Output Layer

Binary Classification

```
0 → Real News
1 → Fake News
```

### Compilation

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metric: Accuracy

---

# 📊 Workflow

### 1. Data Collection

Load fake news dataset.

### 2. Text Preprocessing

* Cleaning
* Lowercasing
* Stopword Removal
* Stemming

### 3. Feature Engineering

* One-Hot Encoding
* Sequence Padding

### 4. Word Embedding

Generate dense word vectors.

### 5. Build Bidirectional LSTM

Construct the neural network.

### 6. Model Training

Train the model on labeled news articles.

### 7. Model Evaluation

Evaluate using:

* Accuracy Score
* Confusion Matrix
* Classification Report

### 8. Predictive System

Predict whether custom news headlines are Fake or Real.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to the project

```bash
cd Project_25_Fake_News_Detection_Bidirectional_LSTM
```

Install dependencies

```bash
pip install numpy pandas tensorflow nltk scikit-learn
```

Run the project

```bash
python main.py
```

---

# 📈 Model Performance

Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report

The Bidirectional LSTM learns contextual relationships from both directions of a sentence, resulting in improved language understanding compared to a standard LSTM.

---

# 🔮 Predictive System

### Example Input

```
Scientists announce breakthrough treatment for Alzheimer's disease.
```

### Example Output

```
Prediction : Real News
```

or

```
Prediction : Fake News
```

---

# 📁 Project Structure

```
Project_25_Fake_News_Detection_Bidirectional_LSTM/
│
├── data.csv
├── main.py
├── README.md
├── fake_news_bilstm_model.keras
│
└── requirements.txt
```

---

# 🔮 Future Improvements

* Attention Mechanism
* GRU Networks
* Word2Vec Embeddings
* GloVe Embeddings
* BERT
* RoBERTa
* Transformer-based Fake News Detection
* Streamlit Deployment

---

# 🎓 Key Learnings

* Natural Language Processing
* Text Preprocessing
* Word Embeddings
* Sequence Modeling
* Bidirectional LSTM
* Binary Text Classification
* Deep Learning with TensorFlow
* Context-aware Language Understanding

---

# 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
