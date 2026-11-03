
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class ContentBasedRecommender:
    def __init__(self):
        # Load the movie dataset
        self.movies = pd.read_csv(DATA_DIR / "movies.csv")

        # Handle missing genre values
        self.movies["genres"] = (
            self.movies["genres"]
            .fillna("")
            .replace("(no genres listed)", "")
        )

        # Convert genre text into numerical features
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True
        )

        self.movie_vectors = self.vectorizer.fit_transform(
            self.movies["genres"].str.replace(
                "|", " ", regex=False
            )
        )

        # Map each title to its row index
        self.title_to_index = {
            title: index
            for index, title in enumerate(self.movies["title"])
        }

    def recommend(self, movie_title, top_n=10):
        # Validate the requested input
        if not isinstance(movie_title, str) or not movie_title.strip():
            raise ValueError("Please enter a valid movie title.")

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        movie_title = movie_title.strip()

        # Find the exact movie title
        if movie_title not in self.title_to_index:
            matches = [
                title
                for title in self.title_to_index
                if movie_title.lower() in title.lower()
            ]

            if matches:
                return {
                    "message": "Exact title not found. Possible matches:",
                    "matches": matches[:10]
                }

            return {
                "message": "No matching movie title was found.",
                "matches": []
            }

        # Get the numerical vector for the selected movie
        movie_index = self.title_to_index[movie_title]
        selected_vector = self.movie_vectors[movie_index]

        # Calculate similarity against all movies on demand
        scores = cosine_similarity(
            selected_vector,
            self.movie_vectors
        ).ravel()

        # Rank movies from most to least similar
        ranked_indices = scores.argsort()[::-1]

        # Exclude the selected movie itself
        ranked_indices = [
            index
            for index in ranked_indices
            if index != movie_index
        ]

        # Select the top recommendations
        top_indices = ranked_indices[:top_n]

        results = self.movies.iloc[top_indices][
            ["movieId", "title", "genres"]
        ].copy()

        results["similarity_score"] = scores[top_indices]

        return results.reset_index(drop=True)


if __name__ == "__main__":
    recommender = ContentBasedRecommender()

    movie_to_search = "Toy Story (1995)"
    recommendations = recommender.recommend(
        movie_to_search,
        top_n=10
    )

    print(f"\nRecommendations for: {movie_to_search}")
    print("=" * 75)

    if isinstance(recommendations, dict):
        print(recommendations["message"])
        for title in recommendations["matches"]:
            print("-", title)
    else:
        print(recommendations.to_string(index=False))


