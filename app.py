from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pro import create_matrix, recommend_movies_for_title

app = Flask(__name__)

# Load your datasets and create the user-item matrix here
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None

    if request.method == 'POST':
        user_title_input = request.form['user_title_input']
        recommendations = recommend_movies_for_title(user_title_input, X, movies, movie_mapper, movie_inv_mapper, k=20)

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

