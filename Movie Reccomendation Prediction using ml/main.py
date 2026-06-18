import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
data = pd.read_csv("movies.csv")

# Features used for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Fill missing values
for feature in selected_features:
    data[feature] = data[feature].fillna('')

# Combine selected features
combined_features = (
    data['genres'] + ' ' +
    data['keywords'] + ' ' +
    data['tagline'] + ' ' +
    data['cast'] + ' ' +
    data['director']
)

# Convert text data into feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

# User Input
movie_name = input("Enter the name of the movie: ")

# Create list of all movie titles
list_of_all_movies = data['title'].tolist()

# Find closest movie name
find_close_match = difflib.get_close_matches(
    movie_name,
    list_of_all_movies
)

if len(find_close_match) == 0:
    print("Movie not found in the database.")
    exit()

close_match = find_close_match[0]

# Get movie index
index_of_movie = data[data.title == close_match]['index'].values[0]

# Get similarity scores
similarity_score = list(enumerate(similarity[index_of_movie]))

# Sort movies based on similarity score
sorted_similar_movies = sorted(
    similarity_score,
    key=lambda x: x[1],
    reverse=True
)

# Display recommendations
print("\nMovies Suggested for You:\n")

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = data[data.index == index]['title'].values[0]

    if i < 30:
        print(f"{i}. {title_from_index}")
        i += 1