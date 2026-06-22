# 🔢 MNIST Handwritten Digit Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Handwritten digit recognition is one of the most fundamental problems in Computer Vision and Deep Learning.

In this project, an **Artificial Neural Network (ANN)** is built using **TensorFlow and Keras** to classify handwritten digits from **0 to 9** using the famous **MNIST Dataset**.

The model learns patterns from thousands of handwritten digit images and predicts the correct digit for unseen images with high accuracy.

---

## 🎯 Objectives

* Load and preprocess the MNIST dataset
* Normalize image pixel values
* Build an Artificial Neural Network (ANN)
* Train the model using TensorFlow/Keras
* Evaluate model performance
* Visualize predictions using a confusion matrix
* Create a predictive system for handwritten digit recognition

---

## 📂 Dataset

### MNIST Dataset

The dataset contains grayscale images of handwritten digits.

#### Training Data

* 60,000 Images
* Image Size: 28 × 28 pixels

#### Testing Data

* 10,000 Images
* Image Size: 28 × 28 pixels

### Classes

| Label | Digit |
| ----- | ----- |
| 0     | 0     |
| 1     | 1     |
| 2     | 2     |
| 3     | 3     |
| 4     | 4     |
| 5     | 5     |
| 6     | 6     |
| 7     | 7     |
| 8     | 8     |
| 9     | 9     |

---

## 🛠️ Technologies Used

* Python
* NumPy
* Matplotlib
* Seaborn
* TensorFlow
* Keras
* Scikit-Learn

---

## 🏗️ Neural Network Architecture

```text
Input Image (28 × 28)
          │
          ▼
Flatten Layer
          │
          ▼
Dense Layer (50 Neurons, ReLU)
          │
          ▼
Dense Layer (50 Neurons, ReLU)
          │
          ▼
Output Layer (10 Neurons)
```

---

## ⚙️ Data Preprocessing

### Image Normalization

Pixel values are scaled from:

```text
0 - 255
```

to:

```text
0 - 1
```

using:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

This improves model convergence and training stability.

---

## 🤖 Deep Learning Model

### Artificial Neural Network (ANN)

Model Components:

* Flatten Layer
* Dense Layer (50 Neurons)
* Dense Layer (50 Neurons)
* Output Layer (10 Classes)

### Compilation Settings

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metric: Accuracy

---

## 📊 Workflow

### 1. Data Collection

Load MNIST dataset from Keras.

### 2. Data Preprocessing

* Normalize pixel values
* Prepare training and testing datasets

### 3. Model Building

Construct ANN using TensorFlow/Keras.

### 4. Model Training

Train the network on 60,000 handwritten digit images.

### 5. Model Evaluation

Evaluate accuracy on unseen test images.

### 6. Prediction

Predict handwritten digits from test images.

### 7. Visualization

Generate confusion matrix for performance analysis.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git
```

Navigate to project folder:

```bash
cd Project_22_MNIST_Digit_Classification
```

Install dependencies:

```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn
```

Run the project:

```bash
python main.py
```

---

## 📈 Model Performance

The Artificial Neural Network achieves high classification accuracy on the MNIST test dataset.

Evaluation Metrics:

* Test Accuracy
* Confusion Matrix
* Class-wise Predictions

The confusion matrix helps visualize prediction performance across all digit classes.

---

## 🔮 Predictive System

The project includes a predictive system that:

✅ Randomly selects an image from the test dataset

✅ Displays the handwritten digit

✅ Predicts the digit using the trained ANN

✅ Compares predicted and actual values

Example Output:

```text
==================================================
The model predicts the digit is: 7
Actual digit is: 7
==================================================
```

---

## 📁 Project Structure

```text
Project_22_MNIST_Digit_Classification/
│
├── main.py
├── README.md
│
└── requirements.txt
```

---

## 🔮 Future Improvements

* Convolutional Neural Networks (CNN)
* Custom Digit Image Prediction
* Streamlit Deployment
* Real-Time Digit Recognition
* Data Augmentation
* Transfer Learning Experiments

---

## 🎓 Key Learnings

* Deep Learning Fundamentals
* Artificial Neural Networks
* TensorFlow & Keras
* Image Classification
* Computer Vision Basics
* Confusion Matrix Analysis
* Multi-Class Classification

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
