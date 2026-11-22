"""
CODSOFT TASK 4 — MOVIE RECOMMENDATION SYSTEM
Run:
    pip install pandas scikit-learn
    python Task4_Recommendation_System.py

This content-based recommender uses TF-IDF and cosine similarity.
The small sample dataset is included in this file for easy GitHub use.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES = [
    ("The Matrix", "Action Sci-Fi", "A hacker discovers reality is simulated and joins a rebellion."),
    ("Inception", "Sci-Fi Thriller", "A thief enters dreams to steal and plant ideas."),
    ("Interstellar", "Sci-Fi Drama", "Explorers travel through space searching for a new home."),
    ("The Dark Knight", "Action Crime", "A masked hero faces a criminal mastermind in a city."),
    ("Toy Story", "Animation Adventure", "Toys come to life and learn about friendship."),
    ("Finding Nemo", "Animation Adventure", "A clownfish crosses the ocean to find his missing son."),
    ("The Notebook", "Romance Drama", "Two people fall in love and face challenges over many years."),
    ("Titanic", "Romance Drama", "A romance unfolds aboard a famous ocean liner."),
    ("The Avengers", "Action Sci-Fi", "A team of heroes joins forces to protect Earth."),
    ("Jurassic Park", "Adventure Sci-Fi", "Scientists bring prehistoric creatures back to life.")
]

def build_recommender():
    df = pd.DataFrame(MOVIES, columns=["title", "genres", "description"])
    df["features"] = df["genres"] + " " + df["description"]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["features"])
    similarity = cosine_similarity(matrix)
    return df, similarity

def recommend(title, df, similarity, count=5):
    matches = df.index[df["title"].str.casefold() == title.casefold()].tolist()
    if not matches:
        return None
    index = matches[0]
    ranked = sorted(enumerate(similarity[index]), key=lambda item: item[1], reverse=True)
    return [(df.iloc[i]["title"], score) for i, score in ranked if i != index][:count]

def main():
    df, similarity = build_recommender()
    print("MOVIE RECOMMENDATION SYSTEM")
    print("Movies:", ", ".join(df["title"]))
    while True:
        title = input("\nEnter a movie title (or 'exit'): ").strip()
        if title.lower() in {"exit", "quit"}:
            break
        results = recommend(title, df, similarity)
        if results is None:
            print("Movie not found. Please choose a title from the list.")
        else:
            print(f"Recommended movies for {title}:")
            for movie, score in results:
                print(f"  {movie} — similarity: {score:.2f}")

if __name__ == "__main__":
    main()
