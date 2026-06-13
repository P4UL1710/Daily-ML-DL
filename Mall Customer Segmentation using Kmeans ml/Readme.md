# 🛍️ Mall Customer Segmentation using K-Means Clustering

This project applies the **K-Means Clustering** algorithm to segment mall customers into distinct groups based on their **Annual Income** and **Spending Score**. Customer segmentation helps businesses understand customer behavior and create targeted marketing strategies.

---

## 📌 Project Overview

Customer segmentation is a key business strategy used to identify groups of customers with similar purchasing behavior. In this project, K-Means Clustering is used to automatically group customers into clusters without predefined labels.

The model analyzes:

- Annual Income (k$)
- Spending Score (1–100)

and identifies different customer segments.

---

## 📂 Dataset

The dataset used is **Mall Customers Dataset** containing:

| Feature | Description |
|----------|-------------|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income of the customer |
| Spending Score (1-100) | Spending behavior score assigned by the mall |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📊 Project Workflow

1. Import required libraries
2. Load and explore the dataset
3. Select relevant features
4. Apply the Elbow Method to determine the optimal number of clusters
5. Train the K-Means Clustering model
6. Predict customer clusters
7. Visualize customer segments and cluster centroids

---

## 📈 Elbow Method

The Elbow Method is used to determine the optimal number of clusters by plotting:

- Number of Clusters
- WCSS (Within Cluster Sum of Squares)

The "elbow point" in the graph indicates the ideal number of clusters.

---

## 🎯 Results

The K-Means algorithm successfully segments customers into **5 distinct groups** based on their income and spending behavior.

These segments can be used for:

- Personalized marketing
- Customer retention strategies
- Targeted promotions
- Business decision making

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/P4UL1710/Daily-ML-DL.git