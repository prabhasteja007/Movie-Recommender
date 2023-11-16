import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to create the user-item matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    # Map Ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    # Map indices to IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Function to find similar movies using KNN
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

# Function to recommend movies for a user based on movie title
def recommend_movies_for_title(title, X, movies, movie_mapper, movie_inv_mapper, k=10):
    # Search for movie ID based on title
    filtered_movies = movies[movies['title'].str.contains(title, case=False)]

    if filtered_movies.empty:
        print(f"No movies found with a title containing '{title}'. Please try a different title.")
        return

    if len(filtered_movies) > 1:
        print("Multiple movies found. Please choose one:")
        for i, movie in enumerate(filtered_movies['title']):
            print(f"{i + 1}. {movie}")
        
        choice = int(input("Enter the number corresponding to your choice: "))
        if 1 <= choice <= len(filtered_movies):
            movie_id = filtered_movies.iloc[choice - 1]['movieId']
        else:
            print("Invalid choice. Recommending for the first match.")
            movie_id = filtered_movies['movieId'].iloc[0]
    else:
        movie_id = filtered_movies['movieId'].iloc[0]

    movie_titles = dict(zip(movies['movieId'], movies['title']))
    movie_genres = dict(zip(movies['movieId'], movies['genres']))

    similar_ids = find_similar_movies(movie_id, X, k)
    movie_title = movie_titles.get(movie_id, "Movie not found")
    movie_genre = movie_genres.get(movie_id, "Unknown")

    if movie_title == "Movie not found":
        print(f"Movie with title '{title}' not found.")
        return

    print(f"\nSince you watched '{movie_title}' ({movie_genre}), you might also like:")
    
    # Sort recommendations by relevance (based on the number of overlapping genres)
    recommendations = []
    for i in similar_ids:
        # Avoid displaying the same movie title
        if i != movie_id:
            recommended_title = movie_titles.get(i, "Movie not found")
            recommended_genre = movie_genres.get(i, "Unknown")
            overlapping_genres = set(movie_genre.split('|')) & set(recommended_genre.split('|'))
            relevance_score = len(overlapping_genres)
            
            # Only include movies with non-zero relevance score
            if relevance_score > 0:
                recommendations.append((recommended_title, recommended_genre, relevance_score))
    
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    # Display recommendations in a tabular format
    recommendation_df = pd.DataFrame(recommendations, columns=['Recommended Movie', 'Genre', 'Relevance Score'])
    
    # Correct formatting of movie titles
    recommendation_df['Recommended Movie'] = recommendation_df['Recommended Movie'].apply(lambda x: x.split(' (')[0])

    print(recommendation_df.to_markdown(index=False))

  
# Main code
# loading rating dataset
ratings = pd.read_csv("ratings.csv")

# loading movie dataset
movies = pd.read_csv("movies.csv")

# Create the user-item matrix
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Get user input for movie title
user_title_input = input("Enter the title of the movie you watched: ")

# Recommend movies for the selected movie title
recommend_movies_for_title(user_title_input, X, movies, movie_mapper, movie_inv_mapper, k=20)


