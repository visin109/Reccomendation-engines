from flask import Flask, render_template, request

from movie_recommendation import (get_similar_movies, item_similarity_df,movie_rates, similar_movie_rated_movies,user_based_recommendations,target_user, cosine_similarity, pivot_table1,similar_genre_movies)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_feedback = request.form['feedback']
        
        if user_feedback == 'like':
            # Call item_based_recommendations function with the liked movie
            movie_title = request.form['movie']
            return render_template('item_recommendations.html',movie_title = movie_title)
        elif user_feedback == 'dislike':
            # Get top 10 movies with higher ratings from the dataset
            top_rated_movies = movie_rates.sort_values(by='rating', ascending=False)['title'].head(10)
            return render_template('item_recommendations.html', top_rated_movies=top_rated_movies)
    
    # Default case: Get user-based recommendations and render the index template
    recommended_movies=user_based_recommendations(target_user, cosine_similarity, pivot_table1)  # Replace this with your function
    return render_template('index.html', recommended_movies=recommended_movies.tolist())

@app.route('/item_based_recommendations', methods=['POST'])
def item_based_recommendations():
    user_feedback = request.form['feedback']  # Get user feedback from the form
    movie_title = request.args.get('movie')  # Get the movie title from the query parameter
    
    if user_feedback == 'like':
        return render_template('like_feedback.html', movie_title=movie_title)
    elif user_feedback == 'dislike':
        return render_template('dislike_feedback.html', movie_title=movie_title)

@app.route('/item_based_recommendations', methods=['GET'])
def item_based_recommendations_default():
    return render_template('item_recommendations.html', similar_movies=[], top_rated_movies=[])

@app.route('/similar_movies_recommendation',methods=['GET', 'POST'])
def similar_movies_recommendation():
    movie_title = request.args.get('movie')
    # Call the item-based recommendation function using the selected movie
    similar_movies = get_similar_movies(item_similarity_df, movie_title, similar_movie_rated_movies)
    return render_template('item_recommendations.html', similar_movies=similar_movies.tolist(), movie_title= movie_title)
@app.route('/top_rated_movies', methods=['GET', 'POST'])
def top_rated_movies():
    movie_title = request.args.get('movie')
    top_rated_movies =movie_rates.sort_values(by='rating', ascending=False)['title'].drop_duplicates().head(10)
    return render_template('item_recommendations.html',top_rated_movies=top_rated_movies.tolist(),movie_title=movie_title)

@app.route('/same_genre_recommendation', methods=['GET','POST'])
def same_genre_recommendation():
    user_feedback = request.form['feedback']
    movie_title = request.args.get('movie')
    # Call the genre-based recommendation function using the selected movie's genre
    if user_feedback == 'like':
        return render_template('like_feedback.html', movie_title=movie_title)
    elif user_feedback == 'dislike':
        return render_template('dislike_feedback.html', movie_title=movie_title)
    # if user_feedback == 'like':
    #     genre_movies =similar_genre_movies(movie_title,movie_rates)  # Implement this function for genre-based recommendations
    #     return render_template('genre_recommendations.html', genre_movies=genre_movies.tolist(),movie_title=movie_title)

    # elif user_feedback == 'dislike':
    #     top_rated_movies =movie_rates.sort_values(by='rating', ascending=False)['title'].drop_duplicates().head(10)
    #     return render_template('item_recommendations.html', top_rated_movies=top_rated_movies)
    # return render_template('item_recommendations.html', similar_movies=[], top_rated_movies=[])

@app.route('/similar_genres_recommendation',methods=['GET', 'POST'])
def similar_genres_recommendation():
    movie_title = request.args.get('movie')
    genre_movies =similar_genre_movies(movie_title,movie_rates)  # Implement this function for genre-based recommendations
    return render_template('genre_recommendations.html', genre_movies=genre_movies.tolist(),movie_title=movie_title)


if __name__ == '__main__':
    app.run(debug=True) 