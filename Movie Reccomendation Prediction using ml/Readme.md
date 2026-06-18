# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

This project is a Content-Based Movie Recommendation System built using Machine Learning techniques. It recommends movies similar to a movie selected by the user by analyzing movie metadata such as genres, keywords, tagline, cast, and director.

The system uses TF-IDF Vectorization and Cosine Similarity to identify movies with similar characteristics and provide personalized recommendations.

---

## 🎯 Features

* Content-Based Recommendation System
* TF-IDF Text Vectorization
* Cosine Similarity Calculation
* Movie Title Matching using Difflib
* Top Similar Movie Recommendations
* Simple Command-Line Interface

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Difflib

---

## 📂 Dataset

The dataset contains movie information including:

* Title
* Genres
* Keywords
* Tagline
* Cast
* Director

These features are combined to generate movie recommendations.

---

## ⚙️ Workflow

1. Load Movie Dataset
2. Handle Missing Values
3. Combine Important Features
4. Convert Text into Numerical Features using TF-IDF
5. Compute Cosine Similarity Matrix
6. Take Movie Name from User
7. Find Closest Matching Movie
8. Recommend Similar Movies

---

## 🧠 Machine Learning Concepts Used

### TF-IDF Vectorization

Converts textual movie information into numerical feature vectors.

### Cosine Similarity

Measures similarity between movies based on their feature vectors.

### Content-Based Filtering

Recommends movies similar to the selected movie using movie attributes rather than user ratings.

---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/Movie-Recommendation-System.git

cd Movie-Recommendation-System

pip install -r requirements.txt

python main.py
```

---

## 📸 Example

Input:

```text
Enter the name of the movie: Avatar
```

Output:

```text
Movies Suggested for You:

1. Guardians of the Galaxy
2. John Carter
3. Star Trek
4. Interstellar
5. The Avengers
...
```

---

## 📈 Future Improvements

* Web Application using Flask/FastAPI
* Streamlit Interface
* User-Based Collaborative Filtering
* Hybrid Recommendation System
* Movie Posters and Ratings Integration

---

## 🎓 Learning Outcomes

Through this project, I learned:

* Feature Engineering
* Natural Language Processing Basics
* TF-IDF Vectorization
* Cosine Similarity
* Recommendation Systems
* Data Preprocessing Techniques

---

## 👨‍💻 Author

**Pawan Tiwari**

Building Machine Learning and Deep Learning projects daily to strengthen practical skills and create a strong portfolio.

⭐ If you found this project useful, consider giving it a star.
