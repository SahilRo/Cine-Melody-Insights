from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load movie recommendation models and data
movies_dict = pickle.load(open('movie_list_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Load music recommendation models and data
df = pickle.load(open('df (3).pkl', 'rb'))
similarity_song = pickle.load(open('similarity_song.pkl', 'rb'))

# Movie Recommendation Function
def recommend_movie(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [(movies.iloc[i[0]].title, movies.iloc[i[0]].Sentiment_Classification) for i in movies_list]
    return recommended_movies

# Music Recommendation Function
def recommend_song(song):
    song_index = df[df['song'] == song].index[0]
    distances = similarity_song[song_index]
    song_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_songs = [(df.iloc[i[0]].song, df.iloc[i[0]].genre) for i in song_list]
    return recommended_songs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        selected_song = request.form.get('song')

        movie_recommendations = []
        song_recommendations = []

        if selected_movie:
            movie_recommendations = recommend_movie(selected_movie)

        if selected_song:
            song_recommendations = recommend_song(selected_song)

        return render_template('index.html', movies=movies['title'].values, songs=df['song'].values,
                               movie_recs=movie_recommendations, song_recs=song_recommendations)

    return render_template('index.html', movies=movies['title'].values, songs=df['song'].values)

if __name__ == "__main__":
    app.run(debug=True)
