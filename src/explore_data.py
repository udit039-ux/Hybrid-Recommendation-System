
from pathlib import Path

import pandas as pd

# Find the project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# Load the datasets
movies = pd.read_csv(DATA_DIR / "movies.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")
links = pd.read_csv(DATA_DIR / "links.csv")

# 1. Display dataset sizes
print("\n--- DATASET SIZES ---")
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Tags:", tags.shape)
print("Links:", links.shape)

# 2. Display sample data
print("\n--- MOVIE DATA ---")
print(movies.head())

print("\n--- RATING DATA ---")
print(ratings.head())

# 3. Count unique users and movies
print("\n--- BASIC STATISTICS ---")
print("Unique users:", ratings["userId"].nunique())
print("Unique rated movies:", ratings["movieId"].nunique())
print("Total ratings:", len(ratings))

# 4. Check missing values
print("\n--- MISSING VALUES ---")
print("Movies:\n", movies.isnull().sum())
print("Ratings:\n", ratings.isnull().sum())

# 5. Check duplicate rows
print("\n--- DUPLICATES ---")
print("Duplicate movie rows:", movies.duplicated().sum())
print("Duplicate rating rows:", ratings.duplicated().sum())

# 6. Find the most frequently rated movies
movie_counts = (
    ratings.groupby("movieId")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

popular_movies = movie_counts.rename("rating_count").reset_index()

popular_movies = popular_movies.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

print("\n--- TOP 10 MOST RATED MOVIES ---")
print(popular_movies[["title", "rating_count"]].to_string(index=False))
