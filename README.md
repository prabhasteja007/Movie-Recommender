Movie Recommendation System
This is a movie recommendation system built using Flask, a popular Python web framework. The system allows users to input a movie title and receive personalized movie recommendations based on collaborative filtering and machine learning algorithms.
Features

Allows users to input a movie title and receive a list of 20 recommended movies.
Utilizes a user-item matrix created from movie rating data to generate the recommendations.
Provides a simple and intuitive web interface for the users to interact with the system.

Getting Started
To run the movie recommendation system locally, follow these steps:

Clone the repository:
https://github.com/prabhasteja007/Movie-Recommender.git

Navigate to the project directory:
cd movie-recommendation-system

Install the required dependencies:
pip install -r requirements.txt

Run the Flask application:
python app.py

Open your web browser and go to http://localhost:5000 to access the movie recommendation system.

Dependencies
The project uses the following dependencies:

Flask
Pandas
NumPy
SciPy
Scikit-learn

Make sure to install these dependencies before running the application.
Customization
You can customize the recommendation system by modifying the app.py file. Here are some areas you can explore:

Implement different recommendation algorithms (e.g., content-based filtering, hybrid approaches)
Enhance the user interface by modifying the index.html template
Integrate additional data sources (e.g., movie metadata, user profiles) to improve the recommendations
