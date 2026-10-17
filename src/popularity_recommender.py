
from pathlib import Path

import pandas as pd


# Locate the project and dataset folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_data():
    """Load movie and rating data."""
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")



    return movies, ratings



def calculate_popularity(movies, ratings):
    """Calculate weighted popularity scores for movies."""

    # Calculate the average rating and number of ratings
    movie_stats = ratings.groupby("movieId")["rating"].agg(
        average_rating="mean",
        rating_count="count"
    ).reset_index()

    # Calculate the average rating across the dataset
    global_average = ratings["rating"].mean()

    # Use the median rating count as our threshold
    minimum_votes = movie_stats["rating_count"].median()

    # Calculate weighted ratings
    movie_stats["weighted_rating"] = (
        (movie_stats["rating_count"] /
         (movie_stats["rating_count"] + minimum_votes))
        * movie_stats["average_rating"]
        +
        (minimum_votes /
         (movie_stats["rating_count"] + minimum_votes))
        * global_average
    )

    # Add movie titles and genres
    movie_stats = movie_stats.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    # Sort by weighted rating, highest first
    movie_stats = movie_stats.sort_values(
        by="weighted_rating",
        ascending=False
    )

    return movie_stats


def recommend_popular(movies, ratings, top_n=10):
    """Return the top N popular movies."""

    ranked_movies = calculate_popularity(movies, ratings)

    return ranked_movies[
        ["movieId", "title", "genres",
         "average_rating", "rating_count", "weighted_rating"]
    ].head(top_n)


if __name__ == "__main__":
    movies, ratings = load_data()

    recommendations = recommend_popular(
        movies,
        ratings,
        top_n=10
    )

    print("\nTOP 10 POPULAR MOVIES")
    print("=" * 80)
    print(recommendations.to_string(index=False))

    print("\nOverall average rating:", round(ratings["rating"].mean(), 3))
